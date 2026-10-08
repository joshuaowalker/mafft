#include "mltaln.h"
#include "dp.h"
#include <limits.h>
#include <stdint.h>
#if defined(MAFFT_SVE)
#ifndef LSVE_MINLANES
#define LSVE_MINLANES 8 /* the SVE fill is used for vectors of at least this many int32 lanes */
#endif
#endif

#define DEBUG 0
#define DEBUG2 0
#define XXXXXXX    0
#define USE_PENALTY_EX  1


static TLS int localstop; // 060910

#if 1
static void match_calc_mtx( double **mtx, double *match, char **s1, char **s2, int i1, int lgth2 ) 
{
	char *seq2 = s2[0];
	double *doubleptr = mtx[(unsigned char)s1[0][i1]];

	while( lgth2-- )
		*match++ = doubleptr[(unsigned char)*seq2++];
}
#else
static void match_calc( double *match, char **s1, char **s2, int i1, int lgth2 )
{
	int j;

	for( j=0; j<lgth2; j++ )
		match[j] = amino_dis[(*s1)[i1]][(*s2)[j]];
}
#endif

#if 0
static void match_calc_bk( double *match, double **cpmx1, double **cpmx2, int i1, int lgth2, double **doublework, int **intwork, int initialize )
{
	int j, k, l;
	double scarr[nalphabets];
	double **cpmxpd = doublework;
	int **cpmxpdn = intwork;
	int count = 0;

	if( initialize )
	{
		for( j=0; j<lgth2; j++ )
		{
			count = 0;
			for( l=0; l<nalphabets; l++ )
			{
				if( cpmx2[l][j] )
				{
					cpmxpd[count][j] = cpmx2[l][j];
					cpmxpdn[count][j] = l;
					count++;
				}
			}
			cpmxpdn[count][j] = -1;
		}
	}

#ifdef HAVE_SCARR_FILL
	scarr_fill( scarr, n_dis, cpmx1, i1 );
#else
	for( l=0; l<nalphabets; l++ )
	{
		scarr[l] = 0.0;
		for( k=0; k<nalphabets; k++ )
			scarr[l] += n_dis[k][l] * cpmx1[k][i1];
	}
#endif
#if 0 /* �����Ȥ��Ȥ���doublework�Υ��������Ȥ�դˤ��� */
	{
		double *fpt, **fptpt, *fpt2;
		int *ipt, **iptpt;
		fpt2 = match;
		iptpt = cpmxpdn;
		fptpt = cpmxpd;
		while( lgth2-- )
		{
			*fpt2 = 0.0;
			ipt=*iptpt,fpt=*fptpt;
			while( *ipt > -1 )
				*fpt2 += scarr[*ipt++] * *fpt++;
			fpt2++,iptpt++,fptpt++;
		} 
	}
#else
	for( j=0; j<lgth2; j++ )
	{
		match[j] = 0.0;
		for( k=0; cpmxpdn[k][j]>-1; k++ )
			match[j] += scarr[cpmxpdn[k][j]] * cpmxpd[k][j];
	} 
#endif
}
#endif

#if defined(MAFFT_AVX2) || ( defined(MAFFT_A64) && !defined(__APPLE__) )
/*
 * Used by every AVX2 build, AVX-512 builds included: on Zen 4 (c7a) this fill made L-INS-i
 * about 7% faster than the AVX-512 prefix-scan fill that AVX-512 builds used before (opt5).
 * The integer DP is the same under every rounding class, so MAFFT_STOCK_FMA does not matter
 * here.  arm64 builds other than Apple's (Linux on Graviton) use it too, with a NEON
 * loop; the Apple build keeps the prefix-scan fill (its all-pairs stage runs on the GPU).
 *
 * The AVX2 integer fill (Lfill_int below) does not compute the traceback offsets themselves.  A
 * cell whose best move is a horizontal gap stores LMARK_H, a vertical gap LMARK_V, and every DP
 * row is kept (lr_*) instead of two alternating ones.  Ltracking only reads ijp along the one
 * path it follows, and for a marked cell asks lres_hk()/lres_vmp() for the offset, which replay
 * the original scalar rules over the stored rows -- integer arithmetic, so the offset is exactly
 * the one the full fill would have stored.  The markers lie below every real value (offsets are
 * at least -lgth2, localstop and warpbase are positive).
 */
#define LFILL_MARKS 1
#define LMARK_H INT_MIN
#define LMARK_V ( INT_MIN + 1 )
static TLS int *lr_raw = NULL;
static TLS size_t lr_cap = 0;
static TLS int *lr_base, lr_stride, lr_ext, lr_v0;
#define LROW( r ) ( lr_base + (size_t)( r ) * lr_stride )

/* hk[j] of row i: first k in [0, j-2] maximising prev[k] - k*ext (the first k of a tie), k = 0 for j < 2 */
static int lres_hk( int i, int j )
{
	int *prev = LROW( i-1 ), jj, tbest = prev[0], tbestk = 0;
	for( jj=2; jj<=j; jj++ )
	{
		int q = prev[jj-2] - ( jj-2 ) * lr_ext;
		if( q > tbest ) { tbest = q; tbestk = jj-2; }
	}
	return( tbestk );
}

/* vmp[j] as row i reads it: the vertical state started from row 0 and updated by rows 1 .. i-1 */
static int lres_vmp( int i, int j )
{
	int r, vm = ( j == 1 ) ? lr_v0 : LROW( 0 )[j-1], vmp = 0;
	for( r=1; r<i; r++ )
	{
		int p = LROW( r-1 )[j-1];
		if( p > vm ) { vm = p; vmp = r-1; }
		vm += lr_ext;
	}
	return( vmp );
}
#endif

static double Ltracking( double *lasthorizontalw, double *lastverticalw, 
						char **seq1, char **seq2, 
                        char **mseq1, char **mseq2, 
                        int **ijp, int *off1pt, int *off2pt, int endi, int endj,
						int *warpis, int *warpjs, int warpbase )
{
	int i, j, l, iin, jin, lgth1, lgth2, k, limk;
	int ifi=0, jfi=0; // by D.Mathog, a guess
//	char gap[] = "-";
	char *gap;
	gap = newgapstr;
	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );

#if 0
	for( i=0; i<lgth1; i++ ) 
	{
		fprintf( stderr, "lastverticalw[%d] = %f\n", i, lastverticalw[i] );
	}
#endif
 
    for( i=0; i<lgth1+1; i++ ) 
    {
        ijp[i][0] = localstop;
    }
    for( j=0; j<lgth2+1; j++ ) 
    {
        ijp[0][j] = localstop;
    }

	mseq1[0] += lgth1+lgth2;
	*mseq1[0] = 0;
	mseq2[0] += lgth1+lgth2;
	*mseq2[0] = 0;
	iin = endi; jin = endj;
	limk = lgth1+lgth2;
	for( k=0; k<=limk; k++ ) 
	{
		int ijv = ijp[iin][jin];
#ifdef LFILL_MARKS
		if( ijv == LMARK_H ) ijv = -( jin - lres_hk( iin, jin ) );
		else if( ijv == LMARK_V ) ijv = iin - lres_vmp( iin, jin );
#endif
		if( ijv >= warpbase )
		{
//			fprintf( stderr, "WARP!\n" );
			ifi = warpis[ijv-warpbase];
			jfi = warpjs[ijv-warpbase];
		}
		else if( ijv < 0 ) 
		{
			ifi = iin-1; jfi = jin+ijv;
		}
		else if( ijv > 0 )
		{
			ifi = iin-ijv; jfi = jin-1;
		}
		else
		{
			ifi = iin-1; jfi = jin-1;
		}


#if 1 // sentou de warp?
		if( ifi == -warpbase && jfi == -warpbase )
		{
			l = iin;
			while( --l >= 0 ) 
			{
				*--mseq1[0] = seq1[0][l];
				*--mseq2[0] = *gap;
				k++;
			}
			l= jin;
			while( --l >= 0 )
			{
				*--mseq1[0] = *gap;
				*--mseq2[0] = seq2[0][l];
				k++;
			}
			break;
		}
		else
#endif
		{
			l = iin - ifi;
			while( --l > 0 ) 
			{
				*--mseq1[0] = seq1[0][ifi+l];
				*--mseq2[0] = *gap;
				k++;
			}
			l= jin - jfi;
			while( --l > 0 )
			{
				*--mseq1[0] = *gap;
				*--mseq2[0] = seq2[0][jfi+l];
				k++;
			}
		}


		if( iin <= 0 || jin <= 0 ) break;
		*--mseq1[0] = seq1[0][ifi];
		*--mseq2[0] = seq2[0][jfi];
		if( ijp[ifi][jfi] == localstop ) break;
		k++;
		iin = ifi; jin = jfi;
	}
	if( ifi == -1 ) *off1pt = 0; else *off1pt = ifi;
	if( jfi == -1 ) *off2pt = 0; else *off2pt = jfi;

//	fprintf( stderr, "ifn = %d, jfn = %d\n", ifi, jfi );
//	fprintf( stderr, "\n" );
//	fprintf( stderr, "%s\n", mseq1[0] );
//	fprintf( stderr, "%s\n", mseq2[0] );


	return( 0.0 );
}


/*
 * Integer version of the L__align11 fill (trywarp == 0 only).
 *
 * When every score and penalty is integral (the default: n_dis * consweight_multi with
 * consweight_multi == 1.0, integer penalties, scoreoffset == 0), the double DP below only ever
 * holds integers well inside 2^53, so an int32 DP makes exactly the same comparisons and yields
 * the same ijp[][], maxwm and end point.  Every candidate for cell (i,j) is derived from row i-1,
 * so a whole row, including the prefix scan for the horizontal-gap state, is computed 4 cells at
 * a time with NEON int32 vectors (scalar code for the last few cells).
 *
 * Returns 0 (and does nothing) when the integral/range preconditions do not hold.
 */
#ifdef LFILL_MARKS
/*
 * Lfill_int for AVX2 (see LFILL_MARKS above): the same integer fill, 8 cells at a time, storing
 * LMARK_H / LMARK_V instead of gap offsets, so neither the first-index scan for the horizontal
 * state nor the vmp[] rows are needed.  Values, wm, the threshold clamp, ijp's 0 / lstop cells,
 * maxwm and the end point are computed exactly as in the original.  On Linux arm64 the same
 * loop runs 4 cells at a time with NEON, or svcntw() at a time with SVE when the vectors are at
 * least 256 bits wide (Graviton3); 128-bit SVE (Graviton4) uses the NEON loop.
 */
#if defined(MAFFT_A64)
#define LPROF_STRIDE ( ( lgth2 + 4 + 3 ) & ~3 ) /* each profile row 16-byte aligned */
#else
#define LPROF_STRIDE ( lgth2 + 4 )
#endif
static int Lfill_int( double **amino_dynamicmtx, double **n_dynamicmtx, double scoreoffset,
                      char *s1, char *s2, int lgth1, int lgth2, int **ijp, int lstop,
                      double *maxwmpt, int *endalipt, int *endaljpt )
{
	int i, j, c, k, maxabs = 0;
	int pen = penalty, ext = penalty_ex;
	double thr = -offset + scoreoffset * 600;
	int ithr, *prof[0x100], *profbuf, nprof = 0;
	int *prev, *cur, *vm, *hq, *wmrow, *vmraw;
#if defined(MAFFT_A64)
	int *kx, *kp, needwm, endpend = 0;
#endif
#if defined(MAFFT_SVE)
	/* lsve_nstep: 0 (use NEON), else the log2 of the lanes per SVE vector (2..4 handled) */
	int vl = (int)svcntw(), lsve_nstep = ( vl >= LSVE_MINLANES && vl <= 16 ) ? ( vl >= 16 ? 4 : vl >= 8 ? 3 : 2 ) : 0;
#endif
	int tbest, jj;
	int maxwm, endali = 0, endalj = 0, rowmax;
	unsigned char *u1 = (unsigned char *)s1, *u2 = (unsigned char *)s2;
	unsigned char used[0x100];
	size_t need;

	if( lgth1 < 1 || lgth2 < 1 ) return( 0 );
	if( thr != (double)(int)thr ) return( 0 );
	ithr = (int)thr;
	for( i=0; i<nalphabets; i++ ) for( j=0; j<nalphabets; j++ )
	{
		double v = n_dynamicmtx[i][j];
		if( v != (double)(int)v ) return( 0 );
		if( abs( (int)v ) > maxabs ) maxabs = abs( (int)v );
	}
	/* Characters outside the alphabet read entries that were never set from n_dynamicmtx. */
	memset( used, 0, sizeof( used ) );
	for( i=0; i<lgth1; i++ ) used[u1[i]] = 1;
	for( j=0; j<lgth2; j++ ) used[u2[j]] = 1;
	for( c=0; c<0x100; c++ ) if( used[c] )
	{
		for( k=0; k<0x100; k++ ) if( used[k] )
		{
			double v = amino_dynamicmtx[c][k];
			if( v != (double)(int)v || fabs( v ) > 1e6 ) return( 0 );
			if( abs( (int)v ) > maxabs ) maxabs = abs( (int)v );
		}
	}
	/* Keep every intermediate (including prev[k] - k*ext) far from int32 overflow. */
	{
		double bound = (double)( maxabs + abs( pen ) + abs( ext ) + abs( ithr ) + 1 ) * ( lgth1 + lgth2 + 4 ) * 2.0;
		if( bound > 5.0e8 ) return( 0 );
	}

	for( c=0; c<0x100; c++ ) prof[c] = NULL;
	for( c=0; c<0x100; c++ ) if( used[c] ) nprof++;
	profbuf = malloc( sizeof( int ) * nprof * LPROF_STRIDE );
	k = 0;
	for( c=0; c<0x100; c++ ) if( used[c] )
	{
		double *row = amino_dynamicmtx[c];
		prof[c] = profbuf + k * LPROF_STRIDE;
		for( j=0; j<lgth2; j++ ) prof[c][j] = (int)row[u2[j]];
		prof[c][lgth2] = 0;
		k++;
	}

	/* All rows, each with one slot in front (prev[-1] seeds the vector scan); rows start so that
	   cell 1 of every row is 32-byte aligned.  Kept between calls: a fresh 3 MB mapping per pair
	   costs more in page faults than the fill. */
	lr_stride = ( ( lgth2 + 9 ) + 7 ) & ~7;
	need = (size_t)( lgth1 + 1 ) * lr_stride + 16;
	/* The rows double the memory of ijp[][]; past 256 MB leave it to the double-precision fill. */
	if( need > ( (size_t)1 << 26 ) ) { free( profbuf ); return( 0 ); }
	if( need > lr_cap )
	{
		free( lr_raw );
		lr_raw = malloc( sizeof( int ) * need );
		if( lr_raw == NULL ) ErrorExit( "Cannot allocate the local alignment rows." );
		lr_cap = need;
	}
#ifdef MAFFT_POISON_TEST
	memset( lr_raw, 0xa5, sizeof( int ) * need );
#endif
#if defined(MAFFT_A64)
	/* the NEON loop starts at cell 4 (ijp rows come from malloc, 16-byte aligned): row cell 4 aligned too */
	lr_base = (int *)( ( (uintptr_t)( lr_raw + 8 ) & ~(uintptr_t)31 ) );
#else
	lr_base = (int *)( ( (uintptr_t)( lr_raw + 8 ) & ~(uintptr_t)31 ) ) + 7; /* lr_base + 1 is 32-byte aligned */
#endif
	lr_ext = ext;
	vmraw = malloc( sizeof( int ) * ( lgth2 + 16 ) );
#if defined(MAFFT_A64)
	vm    = (int *)( ( (uintptr_t)( vmraw + 8 ) & ~(uintptr_t)31 ) );
	/* kx[k] = k*ext, kp[k] = k*ext + ext + pen (horizontal gap scan offsets, read by the NEON loop) */
	kx = malloc( sizeof( int ) * 2 * ( lgth2 + 8 ) );
	kp = kx + lgth2 + 8;
	for( k=0; k<lgth2+8; k++ ) { kx[k] = k * ext; kp[k] = k * ext + ext + pen; }
#else
	vm    = (int *)( ( (uintptr_t)( vmraw + 8 ) & ~(uintptr_t)31 ) ) + 7;
#endif
	hq    = malloc( sizeof( int ) * ( lgth2 + 4 ) );
	wmrow = malloc( sizeof( int ) * ( lgth2 + 4 ) );
#ifdef MAFFT_POISON_TEST
	memset( vmraw, 0xa5, sizeof( int ) * ( lgth2 + 16 ) ); memset( hq, 0xa5, sizeof( int ) * ( lgth2 + 4 ) ); memset( wmrow, 0xa5, sizeof( int ) * ( lgth2 + 4 ) );
#endif

	/* row 0: currentw[j] = mtx[s1[0]][s2[j]];  m[j] = currentw[j-1] */
	cur = LROW( 0 );
	for( j=0; j<lgth2; j++ ) cur[j] = prof[u1[0]][j];
	cur[lgth2] = 0;
	for( j=1; j<=lgth2; j++ ) vm[j] = cur[j-1];
	lr_v0 = cur[0]; /* m[1] before row 1 overwrites cur[0] */

	maxwm = INT_MIN;
	for( i=1; i<=lgth1; i++ )
	{
		int *profrow = ( i < lgth1 ) ? prof[u1[i]] : NULL;
		int *ijrow = ijp[i];
		prev = LROW( i-1 ); cur = LROW( i );
		/* previousw[0] = initverticalw[i-1] = mtx[s2[0]][s1[i-1]] */
		prev[0] = (int)amino_dynamicmtx[u2[0]][u1[i-1]];
		/* H_j = (j-1)*ext + max_{k<=max(j-2,0)} (prev[k] - k*ext); prev[-1] stands for the k=0 term of H_1 */
		prev[-1] = prev[0] - ext;
		tbest = prev[0];

		rowmax = INT_MIN;
		if( !profrow ) profrow = prof[u1[0]]; /* last row: cur[] is never read again */
		j = 1;
#if defined(MAFFT_A64)
		/* Once maxwm > ithr, a row whose maximum beats it has rowmax > ithr, and its first cell
		   holding rowmax is the first j with cur[j] - profrow[j] == rowmax (a clamped cell gives
		   ithr there, below rowmax; any other gives its wm).  The vector loops then skip wmrow,
		   and the search is done once, after the last row, on the row that set maxwm. */
		needwm = ( maxwm <= ithr );
		if( lgth2 >= 7 )
		{
			/* Cells 1..3 in scalar code (the same rules as the remaining cells below), so that the
			   vector loop's stores into ijp, cur, vm and wmrow are 16-byte aligned. */
			for( ; j<4; j++ )
			{
				int p = prev[j-1], wm = p, ij = 0, g;
				if( j >= 2 )
				{
					int q = prev[j-2] - ( j-2 ) * ext;
					if( q > tbest ) tbest = q;
				}
				g = tbest + ( j-1 ) * ext + pen;
				if( g > wm ) { wm = g; ij = LMARK_H; }
				g = vm[j] + pen;
				if( g > wm ) { wm = g; ij = LMARK_V; }
				if( p > vm[j] ) vm[j] = p;
				vm[j] += ext;
				wmrow[j] = wm;
				rowmax = ( wm > rowmax ) ? wm : rowmax;
				if( wm < ithr ) { ij = lstop; wm = ithr; }
				ijrow[j] = ij;
				cur[j] = wm + profrow[j];
			}
		}
#if defined(MAFFT_SVE)
		if( j == 4 && lsve_nstep )
		{
			/* SVE, vector-length agnostic: svcntw() cells at a time (chosen only for vectors of at
			   least LSVE_MINLANES lanes; 128-bit SVE has the NEON loop's width and more work per
			   step).  The same operations as the NEON loop: the in-block prefix max shifts lanes up
			   with tbl (an out-of-range index reads 0) and a max predicated on the lanes that
			   received a value. */
			const svbool_t pt = svptrue_b32();
			const svint32_t vpen = svdup_n_s32( pen ), vext = svdup_n_s32( ext ), vthr = svdup_n_s32( ithr );
			const svint32_t vstop = svdup_n_s32( lstop ), vmarkh = svdup_n_s32( LMARK_H ), vmarkv = svdup_n_s32( LMARK_V );
			const svint32_t zero = svdup_n_s32( 0 );
			const svuint32_t lane = svindex_u32( 0, 1 ), last = svdup_n_u32( (unsigned)vl - 1 );
			const svuint32_t ix1 = svsub_n_u32_x( pt, lane, 1 ), ix2 = svsub_n_u32_x( pt, lane, 2 ), ix4 = svsub_n_u32_x( pt, lane, 4 ), ix8 = svsub_n_u32_x( pt, lane, 8 );
			const svbool_t ge1 = svcmpge_n_u32( pt, lane, 1 ), ge2 = svcmpge_n_u32( pt, lane, 2 ), ge4 = svcmpge_n_u32( pt, lane, 4 ), ge8 = svcmpge_n_u32( pt, lane, 8 );
			svint32_t vmax = svdup_n_s32( rowmax ), cv = svdup_n_s32( tbest );
#define LSVE_BLOCK( J, W ) \
			{ \
				svint32_t p = svld1_s32( pt, prev + (J) - 1 ); \
				svint32_t v = svsub_s32_x( pt, svld1_s32( pt, prev + (J) - 2 ), svld1_s32( pt, kx + (J) - 2 ) ); \
				svint32_t x, g, wm, ij, vmj; \
				svbool_t ch, cvt, clamp; \
				x = svmax_s32_m( ge1, v, svtbl_s32( v, ix1 ) ); \
				x = svmax_s32_m( ge2, x, svtbl_s32( x, ix2 ) ); \
				if( lsve_nstep > 2 ) x = svmax_s32_m( ge4, x, svtbl_s32( x, ix4 ) ); \
				if( lsve_nstep > 3 ) x = svmax_s32_m( ge8, x, svtbl_s32( x, ix8 ) ); \
				x = svmax_s32_x( pt, x, cv ); \
				cv = svtbl_s32( x, last ); \
				wm = p; \
				g = svadd_s32_x( pt, x, svld1_s32( pt, kp + (J) - 2 ) );   /* hq + pen */ \
				ch = svcmpgt_s32( pt, g, wm ); \
				wm = svmax_s32_x( pt, wm, g ); \
				vmj = svld1_s32( pt, vm + (J) ); \
				g = svadd_s32_x( pt, vmj, vpen ); \
				cvt = svcmpgt_s32( pt, g, wm ); \
				wm = svmax_s32_x( pt, wm, g ); \
				svst1_s32( pt, vm + (J), svadd_s32_x( pt, svmax_s32_x( pt, p, vmj ), vext ) ); \
				if( W ) svst1_s32( pt, wmrow + (J), wm ); \
				vmax = svmax_s32_x( pt, vmax, wm ); \
				clamp = svcmpgt_s32( pt, vthr, wm ); \
				ij = svsel_s32( ch, vmarkh, zero ); \
				ij = svsel_s32( cvt, vmarkv, ij ); \
				ij = svsel_s32( clamp, vstop, ij ); \
				wm = svmax_s32_x( pt, wm, vthr ); \
				svst1_s32( pt, ijrow + (J), ij ); \
				svst1_s32( pt, cur + (J), svadd_s32_x( pt, wm, svld1_s32( pt, profrow + (J) ) ) ); \
			}
			if( needwm )
				for( ; j+2*vl-1<=lgth2; j+=2*vl )
				{
					LSVE_BLOCK( j, 1 )
					LSVE_BLOCK( j+vl, 1 )
				}
			else
				for( ; j+2*vl-1<=lgth2; j+=2*vl )
				{
					LSVE_BLOCK( j, 0 )
					LSVE_BLOCK( j+vl, 0 )
				}
			for( ; j+vl-1<=lgth2; j+=vl )
				LSVE_BLOCK( j, needwm )
#undef LSVE_BLOCK
			rowmax = svmaxv_s32( pt, vmax );
			tbest = svlastb_s32( pt, cv );
		}
		else
#endif
		if( j == 4 )
		{
			/* 4 cells at a time, two blocks per iteration.  k*ext and k*ext+ext+pen come from tables
			   (loads, not vector adds).  ij goes through an empty asm so that the compiler keeps
			   the two selects instead of rebuilding them from and/or. */
			const int32x4_t vpen = vdupq_n_s32( pen ), vext = vdupq_n_s32( ext ), vthr = vdupq_n_s32( ithr );
			const int32x4_t vstop = vdupq_n_s32( lstop ), vmarkh = vdupq_n_s32( LMARK_H ), vmarkv = vdupq_n_s32( LMARK_V );
			const int32x4_t ninf = vdupq_n_s32( INT_MIN );
			int32x4_t vmax = vdupq_n_s32( rowmax ), cv = vdupq_n_s32( tbest );
#define LNEON_BLOCK( J, W ) \
			{ \
				int32x4_t p = vld1q_s32( prev + (J) - 1 ); \
				int32x4_t v = vsubq_s32( vld1q_s32( prev + (J) - 2 ), vld1q_s32( kx + (J) - 2 ) ); \
				int32x4_t x, g, wm, ij, vmj; \
				uint32x4_t ch, cvt, clamp; \
				x = vmaxq_s32( v, vextq_s32( ninf, v, 3 ) ); \
				x = vmaxq_s32( x, vextq_s32( ninf, x, 2 ) ); \
				x = vmaxq_s32( x, cv ); \
				cv = vdupq_laneq_s32( x, 3 ); \
				wm = p; \
				g = vaddq_s32( x, vld1q_s32( kp + (J) - 2 ) );   /* hq + pen */ \
				ch = vcgtq_s32( g, wm ); \
				wm = vmaxq_s32( wm, g ); \
				vmj = vld1q_s32( vm + (J) ); \
				g = vaddq_s32( vmj, vpen ); \
				cvt = vcgtq_s32( g, wm ); \
				wm = vmaxq_s32( wm, g ); \
				vst1q_s32( vm + (J), vaddq_s32( vmaxq_s32( p, vmj ), vext ) ); \
				if( W ) vst1q_s32( wmrow + (J), wm ); \
				vmax = vmaxq_s32( vmax, wm ); \
				clamp = vcgtq_s32( vthr, wm ); \
				ij = vandq_s32( vreinterpretq_s32_u32( ch ), vmarkh ); \
				LNEON_OPAQUE( ij ); \
				ij = vbslq_s32( cvt, vmarkv, ij ); \
				LNEON_OPAQUE( ij ); \
				ij = vbslq_s32( clamp, vstop, ij ); \
				wm = vmaxq_s32( wm, vthr ); \
				vst1q_s32( ijrow + (J), ij ); \
				vst1q_s32( cur + (J), vaddq_s32( wm, vld1q_s32( profrow + (J) ) ) ); \
			}
#define LNEON_OPAQUE( r ) __asm__( "" : "+w"( r ) )
			if( needwm )
				for( ; j+7<=lgth2; j+=8 )
				{
					LNEON_BLOCK( j, 1 )
					LNEON_BLOCK( j+4, 1 )
				}
			else
				for( ; j+7<=lgth2; j+=8 )
				{
					LNEON_BLOCK( j, 0 )
					LNEON_BLOCK( j+4, 0 )
				}
			for( ; j+3<=lgth2; j+=4 )
				LNEON_BLOCK( j, needwm )
#undef LNEON_BLOCK
#undef LNEON_OPAQUE
			rowmax = vmaxvq_s32( vmax );
			tbest = vgetq_lane_s32( cv, 0 );
		}
#else
		{
			/* prefix max with in-lane byte shifts (vpslldq) and the fill OR-ed in, one vpermd for the
			   half carry; blends are and/xor (vpblendvb is two port-5 uops on Haswell/Broadwell). */
#define AVX2_BLEND( a, b, m ) _mm256_xor_si256( (a), _mm256_and_si256( _mm256_xor_si256( (a), (b) ), (m) ) )
			const __m256i vpen = _mm256_set1_epi32( pen ), vext = _mm256_set1_epi32( ext ), vthr = _mm256_set1_epi32( ithr );
			const __m256i vstop = _mm256_set1_epi32( lstop ), vmarkh = _mm256_set1_epi32( LMARK_H ), vmarkv = _mm256_set1_epi32( LMARK_V );
			const __m256i ninf = _mm256_set1_epi32( INT_MIN );
			const __m256i fill1 = _mm256_setr_epi32( INT_MIN, 0, 0, 0, INT_MIN, 0, 0, 0 );
			const __m256i fill2 = _mm256_setr_epi32( INT_MIN, INT_MIN, 0, 0, INT_MIN, INT_MIN, 0, 0 );
			const __m256i idx3 = _mm256_set1_epi32( 3 ), last = _mm256_set1_epi32( 7 );
			const __m256i ext8 = _mm256_set1_epi32( 8 * ext );
			__m256i vmax = ninf, cv = ninf;
			/* lane l of the block at j stands for k = j-2+l */
			__m256i kext = _mm256_mullo_epi32( _mm256_setr_epi32( -1, 0, 1, 2, 3, 4, 5, 6 ), vext );
			__m256i kext1pen = _mm256_add_epi32( kext, _mm256_set1_epi32( ext + pen ) );
			for( ; j+7<=lgth2; j+=8 )
			{
				__m256i p = _mm256_loadu_si256( (__m256i *)( prev + j - 1 ) );
				__m256i v = _mm256_sub_epi32( _mm256_loadu_si256( (__m256i *)( prev + j - 2 ) ), kext );
				__m256i x, g, wm, ch, cvt, clamp, ij, vmj;
				x = _mm256_max_epi32( v, _mm256_or_si256( _mm256_bslli_epi128( v, 4 ), fill1 ) );
				x = _mm256_max_epi32( x, _mm256_or_si256( _mm256_bslli_epi128( x, 8 ), fill2 ) );
				x = _mm256_max_epi32( x, _mm256_blend_epi32( ninf, _mm256_permutevar8x32_epi32( x, idx3 ), 0xf0 ) );
				x = _mm256_max_epi32( x, cv );
				cv = _mm256_permutevar8x32_epi32( x, last );

				wm = p;
				g = _mm256_add_epi32( x, kext1pen );   /* hq + pen */
				ch = _mm256_cmpgt_epi32( g, wm );
				wm = _mm256_max_epi32( wm, g );
				vmj = _mm256_load_si256( (__m256i *)( vm + j ) );
				g = _mm256_add_epi32( vmj, vpen );
				cvt = _mm256_cmpgt_epi32( g, wm );
				wm = _mm256_max_epi32( wm, g );
				_mm256_store_si256( (__m256i *)( vm + j ), _mm256_add_epi32( _mm256_max_epi32( p, vmj ), vext ) );
				_mm256_storeu_si256( (__m256i *)( wmrow + j ), wm );
				vmax = _mm256_max_epi32( vmax, wm );
				clamp = _mm256_cmpgt_epi32( vthr, wm );
				ij = _mm256_and_si256( ch, vmarkh );
				ij = AVX2_BLEND( ij, vmarkv, cvt );
				ij = AVX2_BLEND( ij, vstop, clamp );
				wm = _mm256_max_epi32( wm, vthr );
				_mm256_storeu_si256( (__m256i *)( ijrow + j ), ij );
				_mm256_store_si256( (__m256i *)( cur + j ), _mm256_add_epi32( wm, _mm256_loadu_si256( (__m256i *)( profrow + j ) ) ) );
				kext = _mm256_add_epi32( kext, ext8 );
				kext1pen = _mm256_add_epi32( kext1pen, ext8 );
			}
#undef AVX2_BLEND
			{
				__m128i h = _mm_max_epi32( _mm256_castsi256_si128( vmax ), _mm256_extracti128_si256( vmax, 1 ) );
				h = _mm_max_epi32( h, _mm_shuffle_epi32( h, 0x4e ) );
				h = _mm_max_epi32( h, _mm_shuffle_epi32( h, 0xb1 ) );
				rowmax = _mm_cvtsi128_si32( h );
			}
			if( j > 1 ) tbest = _mm256_cvtsi256_si32( cv );
		}
#endif
		/* the remaining cells */
		for( jj=j; jj<=lgth2; jj++ )
		{
			if( jj >= 2 )
			{
				int q = prev[jj-2] - ( jj-2 ) * ext;
				if( q > tbest ) tbest = q;
			}
			hq[jj] = tbest + ( jj-1 ) * ext;
		}
		for( ; j<=lgth2; j++ )
		{
			int p = prev[j-1];
			int wm = p, ij = 0, g;
			g = hq[j] + pen;
			if( g > wm ) { wm = g; ij = LMARK_H; }
			g = vm[j] + pen;
			if( g > wm ) { wm = g; ij = LMARK_V; }
			if( p > vm[j] ) vm[j] = p;
			vm[j] += ext;
			wmrow[j] = wm;
			rowmax = ( wm > rowmax ) ? wm : rowmax;
			if( wm < ithr ) { ij = lstop; wm = ithr; }
			ijrow[j] = ij;
			cur[j] = wm + profrow[j];
		}
#if defined(MAFFT_A64)
		if( rowmax > maxwm && !needwm )
		{
			maxwm = rowmax; endali = i; endpend = 1;
		}
		else
#endif
		if( rowmax > maxwm )
		{
#if defined(MAFFT_A64)
			endpend = 0;
			/* first cell holding rowmax (it is always present): 16 cells per test, then 4 */
			int32x4_t vr = vdupq_n_s32( rowmax );
			j = 1;
			for( ; j+15<=lgth2; j+=16 )
				if( vmaxvq_u32( vorrq_u32( vorrq_u32( vceqq_s32( vld1q_s32( wmrow + j ), vr ), vceqq_s32( vld1q_s32( wmrow + j + 4 ), vr ) ),
				                           vorrq_u32( vceqq_s32( vld1q_s32( wmrow + j + 8 ), vr ), vceqq_s32( vld1q_s32( wmrow + j + 12 ), vr ) ) ) ) ) break;
			for( ; j+3<=lgth2; j+=4 )
				if( vmaxvq_u32( vceqq_s32( vld1q_s32( wmrow + j ), vr ) ) ) break;
#else
			/* first cell holding rowmax, 8 at a time (it is always present) */
			__m256i vr = _mm256_set1_epi32( rowmax );
			j = 1;
			for( ; j+7<=lgth2; j+=8 )
			{
				int km = _mm256_movemask_ps( _mm256_castsi256_ps( _mm256_cmpeq_epi32( _mm256_loadu_si256( (__m256i *)( wmrow + j ) ), vr ) ) );
				if( km ) { j += __builtin_ctz( (unsigned)km ); break; }
			}
#endif
			for( ; wmrow[j] != rowmax; j++ )
				;
			maxwm = rowmax; endali = i; endalj = j;
		}
		if( i < lgth1 ) cur[0] = (int)amino_dynamicmtx[u2[0]][u1[i]]; /* currentw[0] = initverticalw[i] */
	}

#if defined(MAFFT_A64)
	if( endpend )
	{
		int *r = LROW( endali ), *pr = ( endali < lgth1 ) ? prof[u1[endali]] : prof[u1[0]];
		for( j=1; r[j] - pr[j] != maxwm; j++ )
			;
		endalj = j;
	}
#endif
	free( profbuf ); free( vmraw ); free( hq ); free( wmrow );
#if defined(MAFFT_A64)
	free( kx );
#endif
	*maxwmpt = (double)maxwm;
	*endalipt = endali;
	*endaljpt = endalj;
	return( 1 );
}
#if defined(MAFFT_AVX512X)
/*
 * Lfill_kb: the marker fill above in a striped layout (Farrar's), with ijp kept as one byte per
 * cell (0 diagonal, KB_H for LMARK_H, KB_V for LMARK_V, KB_STOP for lstop) instead of an int.
 *
 * Column c = j-1 of a row (c = 0 .. lgth2-1, padded to 16*S) lives in lane c / S, position c % S,
 * S = ceil( lgth2 / 16 ).  Cell j reads prev[j-1] (its own position in the previous row) and the
 * horizontal state H_j = (j-1)*ext + max_{k <= max(j-2,0)} T_k, T_k = prev[k] - k*ext.  In this
 * layout that maximum is the larger of a running maximum along the cell's own lane (positions
 * before it) and a carry: the maxima of the whole lanes below it (for lane 0, T_0 itself, which is
 * the k = 0 term cell 1 uses).  The lane maxima come from the previous row's pass, so a row is one
 * pass of plain vertical max/add operations with no prefix scan inside the loop.  Padding columns
 * (c >= lgth2) sit above every real column, and the scan only carries to higher columns, so they
 * never reach a real cell; they are kept out of the row maximum by a mask.
 *
 * The rest is Lfill_int's arithmetic per cell, unchanged.  Every row is kept (striped, row i's
 * value for column c, i.e. prev[c] as row i+1 reads it, at KB_RV( i, c )) for the marker replay,
 * which kb_hk() and kb_vmp() do exactly as lres_hk() and lres_vmp() do over Lfill_int's rows.
 * The first cell holding a row's maximum is the first position holding it in the lowest lane
 * that has it (lanes are in column order).  Ltracking_kb() is Ltracking() reading kb_ijv().
 */
/* 64-byte aligned allocation, released with aligned_free_64() */
static void *aligned_alloc_64( size_t n )
{
	char *raw = malloc( n + 64 + sizeof( void * ) ), *p;
	if( raw == NULL ) ErrorExit( "Cannot allocate memory." );
	p = (char *)( ( (uintptr_t)( raw + sizeof( void * ) ) + 63 ) & ~(uintptr_t)63 );
	( (void **)p )[-1] = raw;
	return( p );
}
static void aligned_free_64( void *p ) { if( p ) free( ( (void **)p )[-1] ); }
#define KB_H 1
#define KB_V 2
#define KB_STOP 3
static TLS unsigned char *kb_raw = NULL;
static TLS int *kb_rows = NULL;
static TLS size_t kb_cap = 0, kb_rcap = 0;
static TLS int kb_S, kb_ext, kb_v0;
#define KB_RSTRIDE ( (size_t)( kb_S + 1 ) * 16 )
/* row r's value at column c (striped vector c % S, lane c / S; vector 0 holds the shifted values) */
#define KB_RV( r, c ) ( kb_rows[(size_t)( r ) * KB_RSTRIDE + (size_t)( ( c ) % kb_S ) * 16 + ( c ) / kb_S] )
#define KB_KIND( i, c ) ( kb_raw[(size_t)( i ) * kb_S * 16 + (size_t)( ( c ) % kb_S ) * 16 + ( c ) / kb_S] )

static int Lfill_kb( double **amino_dynamicmtx, double **n_dynamicmtx, double scoreoffset,
                     char *s1, char *s2, int lgth1, int lgth2,
                     double *maxwmpt, int *endalipt, int *endaljpt )
{
	int i, j, c, k, l, s, S, maxabs = 0;
	int pen = penalty, ext = penalty_ex;
	double thr = -offset + scoreoffset * 600;
	int ithr, *prof[0x100], *profbuf, nprof = 0, *vmraw, *vm, *wmrow;
	int maxwm, endali = 0, endalj = 0, rowmax;
	unsigned char *u1 = (unsigned char *)s1, *u2 = (unsigned char *)s2;
	unsigned char used[0x100];
	size_t need, needkb, rs;
	__m512i lanebase, ltn;

	if( lgth1 < 1 || lgth2 < 1 ) return( 0 );
	if( thr != (double)(int)thr ) return( 0 );
	ithr = (int)thr;
	for( i=0; i<nalphabets; i++ ) for( j=0; j<nalphabets; j++ )
	{
		double v = n_dynamicmtx[i][j];
		if( v != (double)(int)v ) return( 0 );
		if( abs( (int)v ) > maxabs ) maxabs = abs( (int)v );
	}
	memset( used, 0, sizeof( used ) );
	for( i=0; i<lgth1; i++ ) used[u1[i]] = 1;
	for( j=0; j<lgth2; j++ ) used[u2[j]] = 1;
	for( c=0; c<0x100; c++ ) if( used[c] )
	{
		for( k=0; k<0x100; k++ ) if( used[k] )
		{
			double v = amino_dynamicmtx[c][k];
			if( v != (double)(int)v || fabs( v ) > 1e6 ) return( 0 );
			if( abs( (int)v ) > maxabs ) maxabs = abs( (int)v );
		}
	}
	{
		double bound = (double)( maxabs + abs( pen ) + abs( ext ) + abs( ithr ) + 1 ) * ( lgth1 + lgth2 + 4 ) * 2.0;
		if( bound > 5.0e8 ) return( 0 );
	}

	S = kb_S = ( lgth2 + 15 ) / 16;
	rs = KB_RSTRIDE;
	need = (size_t)( lgth1 + 1 ) * rs;
	needkb = (size_t)( lgth1 + 1 ) * S * 16;
	if( need > ( (size_t)1 << 26 ) ) return( 0 );
	if( need > kb_rcap )
	{
		aligned_free_64( kb_rows );
		kb_rows = aligned_alloc_64( sizeof( int ) * need );
		kb_rcap = need;
	}
	if( needkb > kb_cap )
	{
		free( kb_raw );
		kb_raw = malloc( needkb );
		if( kb_raw == NULL ) ErrorExit( "Cannot allocate the local alignment traceback." );
		kb_cap = needkb;
	}
#ifdef MAFFT_POISON_TEST
	memset( kb_rows, 0xa5, sizeof( int ) * need ); memset( kb_raw, 0xa5, needkb );
#endif

	/* striped profiles: cell j = c+1 adds prof[.][j] (prof[.][lgth2] = 0; padding 0) */
	for( c=0; c<0x100; c++ ) prof[c] = NULL;
	for( c=0; c<0x100; c++ ) if( used[c] ) nprof++;
	profbuf = aligned_alloc_64( sizeof( int ) * nprof * S * 16 );
	k = 0;
	for( c=0; c<0x100; c++ ) if( used[c] )
	{
		double *row = amino_dynamicmtx[c];
		prof[c] = profbuf + (size_t)k * S * 16;
		for( s=0; s<S; s++ ) for( l=0; l<16; l++ )
		{
			int jj = l * S + s + 1;
			prof[c][s*16+l] = ( jj < lgth2 ) ? (int)row[u2[jj]] : 0;
		}
		k++;
	}
	vmraw = aligned_alloc_64( sizeof( int ) * S * 16 * 2 );
	vm = vmraw; wmrow = vmraw + S * 16;

	/* row 0: R_0[c] = mtx[s1[0]][s2[c]] for c < lgth2 (0 in the padding), in the shifted layout
	   (vector s holds columns l*S+s); vm[j] = R_0[j-1] */
	{
		int *r0 = kb_rows;
		for( s=0; s<S; s++ ) for( l=0; l<16; l++ )
		{
			int cc = l * S + s;
			r0[s*16+l] = ( cc < lgth2 ) ? (int)amino_dynamicmtx[u1[0]][u2[cc]] : 0;
		}
		memcpy( vm, r0, sizeof( int ) * S * 16 );
		kb_v0 = r0[0]; /* m[1] before row 1 overwrites R_0[0] */
	}
	kb_ext = ext;

	lanebase = _mm512_mullo_epi32( _mm512_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ), _mm512_set1_epi32( S ) );
	/* lane maxima of row 0's T (padding included: it only reaches padding) */
	{
		__m512i cext = _mm512_mullo_epi32( lanebase, _mm512_set1_epi32( ext ) );
		ltn = _mm512_set1_epi32( INT_MIN );
		for( s=0; s<S; s++ )
		{
			if( s > 0 ) ltn = _mm512_max_epi32( ltn, _mm512_sub_epi32( _mm512_load_si512( (void *)( kb_rows + s*16 ) ), cext ) );
			cext = _mm512_add_epi32( cext, _mm512_set1_epi32( ext ) );
		}
	}

	maxwm = INT_MIN;
	for( i=1; i<=lgth1; i++ )
	{
		int *prev = kb_rows + (size_t)( i-1 ) * rs, *cur = kb_rows + (size_t)i * rs;
		int *profrow = ( i < lgth1 ) ? prof[u1[i]] : prof[u1[0]]; /* last row: cur[] is never read again */
		unsigned char *kbrow = kb_raw + (size_t)i * S * 16;
		int special = (int)amino_dynamicmtx[u2[0]][u1[i-1]]; /* prev[0] = initverticalw[i-1] */
		const __m512i vpen = _mm512_set1_epi32( pen ), vext = _mm512_set1_epi32( ext ), vthr = _mm512_set1_epi32( ithr );
		const __m512i vkh = _mm512_set1_epi32( KB_H ), vkv = _mm512_set1_epi32( KB_V ), vks = _mm512_set1_epi32( KB_STOP );
		const __m512i ninf = _mm512_set1_epi32( INT_MIN ), one = _mm512_set1_epi32( 1 ), vn = _mm512_set1_epi32( lgth2 );
		__m512i r0, t0, lt, carry, e, cext, cextpen, vmax, pos;
		/* vector 0 of the previous row: its columns l*S, i.e. the previous lane's last cell, and
		   prev[0] in lane 0 */
		if( i == 1 ) r0 = _mm512_mask_mov_epi32( _mm512_load_si512( (void *)prev ), 1, _mm512_set1_epi32( special ) );
		else r0 = _mm512_alignr_epi32( _mm512_load_si512( (void *)( prev + (size_t)S * 16 ) ), _mm512_set1_epi32( special ), 15 );
		_mm512_store_si512( (void *)prev, r0 );
		/* lane maxima of T over the previous row, then the carry: the maximum over the lanes below,
		   and T_0 for lane 0 */
		cext = _mm512_mullo_epi32( lanebase, vext );
		t0 = _mm512_sub_epi32( r0, cext );
		lt = _mm512_max_epi32( ltn, t0 );
		carry = _mm512_alignr_epi32( lt, ninf, 15 );
		carry = _mm512_max_epi32( carry, _mm512_alignr_epi32( carry, ninf, 15 ) );
		carry = _mm512_max_epi32( carry, _mm512_alignr_epi32( carry, ninf, 14 ) );
		carry = _mm512_max_epi32( carry, _mm512_alignr_epi32( carry, ninf, 12 ) );
		carry = _mm512_max_epi32( carry, _mm512_alignr_epi32( carry, ninf, 8 ) );
		carry = _mm512_mask_mov_epi32( carry, 1, _mm512_set1_epi32( special ) ); /* T_0 = prev[0] */

		e = ninf;
		cextpen = _mm512_add_epi32( cext, vpen );
		vmax = ninf;
		pos = lanebase;
		ltn = ninf;
		for( s=0; s<S; s++ )
		{
			__m512i a = _mm512_load_si512( (void *)( prev + s*16 ) ), vmj = _mm512_load_si512( (void *)( vm + s*16 ) );
			__m512i t = _mm512_sub_epi32( a, cext ), m, g, wm, ij, v;
			__mmask16 ch, cvt, clamp;
			m = _mm512_max_epi32( e, carry );
			e = _mm512_max_epi32( e, t );
			g = _mm512_add_epi32( m, cextpen );    /* H + pen */
			ch = _mm512_cmpgt_epi32_mask( g, a );
			wm = _mm512_max_epi32( a, g );
			g = _mm512_add_epi32( vmj, vpen );
			cvt = _mm512_cmpgt_epi32_mask( g, wm );
			wm = _mm512_max_epi32( wm, g );
			_mm512_store_si512( (void *)( vm + s*16 ), _mm512_add_epi32( _mm512_max_epi32( a, vmj ), vext ) );
			_mm512_store_si512( (void *)( wmrow + s*16 ), wm );
			vmax = _mm512_mask_max_epi32( vmax, _mm512_cmplt_epi32_mask( pos, vn ), vmax, wm ); /* real columns only */
			clamp = _mm512_cmpgt_epi32_mask( vthr, wm );
			ij = _mm512_maskz_mov_epi32( ch, vkh );
			ij = _mm512_mask_mov_epi32( ij, cvt, vkv );
			ij = _mm512_mask_mov_epi32( ij, clamp, vks );
			_mm_store_si128( (__m128i *)( kbrow + s*16 ), _mm512_cvtepi32_epi8( ij ) );
			wm = _mm512_max_epi32( wm, vthr );
			v = _mm512_add_epi32( wm, _mm512_load_si512( (void *)( profrow + s*16 ) ) );
			_mm512_store_si512( (void *)( cur + (size_t)( s+1 ) * 16 ), v );
			cext = _mm512_add_epi32( cext, vext );
			cextpen = _mm512_add_epi32( cextpen, vext );
			/* next row's T at column c+1, same lane except the last position (lane above, added
			   with vector 0 at the next row's start) */
			if( s < S-1 ) ltn = _mm512_max_epi32( ltn, _mm512_sub_epi32( v, cext ) );
			pos = _mm512_add_epi32( pos, one );
		}
		rowmax = _mm512_reduce_max_epi32( vmax );
		if( rowmax > maxwm )
		{
			/* the first column holding it: the lowest lane that has it (lanes are in column order),
			   then its first position there */
			__m512i vr = _mm512_set1_epi32( rowmax );
			unsigned any = 0;
			int ln, c0;
			pos = lanebase;
			for( s=0; s<S; s++ )
			{
				any |= _mm512_mask_cmpeq_epi32_mask( _mm512_cmplt_epi32_mask( pos, vn ), _mm512_load_si512( (void *)( wmrow + s*16 ) ), vr );
				pos = _mm512_add_epi32( pos, one );
			}
			ln = __builtin_ctz( any );
			for( s=0; wmrow[s*16+ln] != rowmax; s++ )
				;
			c0 = ln * S + s;
			maxwm = rowmax; endali = i; endalj = c0 + 1;
		}
	}

	aligned_free_64( profbuf ); aligned_free_64( vmraw );
	*maxwmpt = (double)maxwm;
	*endalipt = endali;
	*endaljpt = endalj;
	return( 1 );
}

/* lres_hk() and lres_vmp() over the striped rows */
static int kb_hk( int i, int j )
{
	int jj, tbest = KB_RV( i-1, 0 ), tbestk = 0;
	for( jj=2; jj<=j; jj++ )
	{
		int q = KB_RV( i-1, jj-2 ) - ( jj-2 ) * kb_ext;
		if( q > tbest ) { tbest = q; tbestk = jj-2; }
	}
	return( tbestk );
}
static int kb_vmp( int i, int j )
{
	int r, vm = ( j == 1 ) ? kb_v0 : KB_RV( 0, j-1 ), vmp = 0;
	for( r=1; r<i; r++ )
	{
		int p = KB_RV( r-1, j-1 );
		if( p > vm ) { vm = p; vmp = r-1; }
		vm += kb_ext;
	}
	return( vmp );
}

/* ijp[i][j] as Ltracking resolves it after Lfill_int (row 0 and column 0 hold localstop) */
static int kb_ijv( int i, int j, int lstop )
{
	if( i <= 0 || j <= 0 ) return( lstop );
	switch( KB_KIND( i, j-1 ) )
	{
		case KB_H: return( -( j - kb_hk( i, j ) ) );
		case KB_V: return( i - kb_vmp( i, j ) );
		case KB_STOP: return( lstop );
		default: return( 0 );
	}
}
#define KB_ISSTOP( i, j ) ( ( i ) <= 0 || ( j ) <= 0 || KB_KIND( i, ( j )-1 ) == KB_STOP )

static void Ltracking_kb( char **seq1, char **seq2, char **mseq1, char **mseq2,
                            int *off1pt, int *off2pt, int endi, int endj, int lstop,
                            int *warpis, int *warpjs, int warpbase )
{
	int l, iin, jin, lgth1, lgth2, k, limk;
	int ifi=0, jfi=0;
	char *gap = newgapstr;
	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );

	mseq1[0] += lgth1+lgth2;
	*mseq1[0] = 0;
	mseq2[0] += lgth1+lgth2;
	*mseq2[0] = 0;
	iin = endi; jin = endj;
	limk = lgth1+lgth2;
	for( k=0; k<=limk; k++ )
	{
		int ijv = kb_ijv( iin, jin, lstop );
		if( ijv >= warpbase )
		{
			ifi = warpis[ijv-warpbase];
			jfi = warpjs[ijv-warpbase];
		}
		else if( ijv < 0 )
		{
			ifi = iin-1; jfi = jin+ijv;
		}
		else if( ijv > 0 )
		{
			ifi = iin-ijv; jfi = jin-1;
		}
		else
		{
			ifi = iin-1; jfi = jin-1;
		}

		if( ifi == -warpbase && jfi == -warpbase )
		{
			l = iin;
			while( --l >= 0 )
			{
				*--mseq1[0] = seq1[0][l];
				*--mseq2[0] = *gap;
				k++;
			}
			l= jin;
			while( --l >= 0 )
			{
				*--mseq1[0] = *gap;
				*--mseq2[0] = seq2[0][l];
				k++;
			}
			break;
		}
		else
		{
			l = iin - ifi;
			while( --l > 0 )
			{
				*--mseq1[0] = seq1[0][ifi+l];
				*--mseq2[0] = *gap;
				k++;
			}
			l= jin - jfi;
			while( --l > 0 )
			{
				*--mseq1[0] = *gap;
				*--mseq2[0] = seq2[0][jfi+l];
				k++;
			}
		}

		if( iin <= 0 || jin <= 0 ) break;
		*--mseq1[0] = seq1[0][ifi];
		*--mseq2[0] = seq2[0][jfi];
		if( KB_ISSTOP( ifi, jfi ) ) break;
		k++;
		iin = ifi; jin = jfi;
	}
	if( ifi == -1 ) *off1pt = 0; else *off1pt = ifi;
	if( jfi == -1 ) *off2pt = 0; else *off2pt = jfi;
}
#endif
#else
static int Lfill_int( double **amino_dynamicmtx, double **n_dynamicmtx, double scoreoffset,
                      char *s1, char *s2, int lgth1, int lgth2, int **ijp, int lstop,
                      double *maxwmpt, int *endalipt, int *endaljpt )
{
	int i, j, c, k, maxabs = 0, ok = 1;
	int pen = penalty, ext = penalty_ex;
	double thr = -offset + scoreoffset * 600;
	int ithr, *prof[0x100], *profbuf, nprof = 0;
	int *prev, *cur, *vm, *vmp, *hq, *hk, *wmrow, *pbuf1, *pbuf2;
	int tbest, tbestk, jj;
	int maxwm, endali = 0, endalj = 0, rowmax;
	unsigned char *u1 = (unsigned char *)s1, *u2 = (unsigned char *)s2;
	unsigned char used[0x100];

	if( lgth1 < 1 || lgth2 < 1 ) return( 0 );
	if( thr != (double)(int)thr ) return( 0 );
	ithr = (int)thr;
	for( i=0; i<nalphabets; i++ ) for( j=0; j<nalphabets; j++ )
	{
		double v = n_dynamicmtx[i][j];
		if( v != (double)(int)v ) return( 0 );
		if( abs( (int)v ) > maxabs ) maxabs = abs( (int)v );
	}
	/* Characters outside the alphabet read entries that were never set from n_dynamicmtx. */
	memset( used, 0, sizeof( used ) );
	for( i=0; i<lgth1; i++ ) used[u1[i]] = 1;
	for( j=0; j<lgth2; j++ ) used[u2[j]] = 1;
	for( c=0; c<0x100; c++ ) if( used[c] )
	{
		for( k=0; k<0x100; k++ ) if( used[k] )
		{
			double v = amino_dynamicmtx[c][k];
			if( v != (double)(int)v || fabs( v ) > 1e6 ) return( 0 );
			if( abs( (int)v ) > maxabs ) maxabs = abs( (int)v );
		}
	}
	/* Keep every intermediate (including prev[k] - k*ext) far from int32 overflow. */
	{
		double bound = (double)( maxabs + abs( pen ) + abs( ext ) + abs( ithr ) + 1 ) * ( lgth1 + lgth2 + 4 ) * 2.0;
		if( bound > 5.0e8 ) return( 0 );
	}

	for( c=0; c<0x100; c++ ) prof[c] = NULL;
	for( c=0; c<0x100; c++ ) if( used[c] ) nprof++;
	profbuf = malloc( sizeof( int ) * nprof * ( lgth2 + 4 ) );
	k = 0;
	for( c=0; c<0x100; c++ ) if( used[c] )
	{
		double *row = amino_dynamicmtx[c];
		prof[c] = profbuf + k * ( lgth2 + 4 );
		for( j=0; j<lgth2; j++ ) prof[c][j] = (int)row[u2[j]];
		prof[c][lgth2] = 0;
		k++;
	}

	/* one slot in front of prev[]/cur[]: prev[-1] seeds the vector scan */
	pbuf1 = calloc( lgth2 + 5, sizeof( int ) ); prev = pbuf1 + 1;
	pbuf2 = calloc( lgth2 + 5, sizeof( int ) ); cur = pbuf2 + 1;
	vm    = malloc( sizeof( int ) * ( lgth2 + 4 ) );
	vmp   = malloc( sizeof( int ) * ( lgth2 + 4 ) );
	hq    = malloc( sizeof( int ) * ( lgth2 + 4 ) );
	hk    = malloc( sizeof( int ) * ( lgth2 + 4 ) );
	wmrow = malloc( sizeof( int ) * ( lgth2 + 4 ) );

	/* row 0: currentw[j] = mtx[s1[0]][s2[j]];  m[j] = currentw[j-1], mp[j] = 0 */
	for( j=0; j<lgth2; j++ ) cur[j] = prof[u1[0]][j];
	cur[lgth2] = 0;
	for( j=1; j<=lgth2; j++ ) { vm[j] = cur[j-1]; vmp[j] = 0; }

	maxwm = INT_MIN;
	for( i=1; i<=lgth1; i++ )
	{
		int *t = prev; prev = cur; cur = t;
		int *profrow = ( i < lgth1 ) ? prof[u1[i]] : NULL;
		int *ijrow = ijp[i];
		int negi1 = i - 1;
		/* previousw[0] = initverticalw[i-1] = mtx[s2[0]][s1[i-1]] */
		prev[0] = (int)amino_dynamicmtx[u2[0]][u1[i-1]];

		/* Horizontal state before cell j:  H_j = (j-1)*ext + max_{k<=max(j-2,0)} (prev[k] - k*ext),
		   mpi_j = first k attaining it (the original updates only on strict '>').  Computed
		   inside the loop below: the first index attaining a running max is the last strict
		   record, so both are prefix maxima, taken 4 lanes at a time with a carry.  prev[-1] is
		   set so that lane k=-1 of the first block stands for the k=0 term of H_1. */
		prev[-1] = prev[0] - ext;
		tbest = prev[0]; tbestk = 0;

		rowmax = INT_MIN;
		if( !profrow ) profrow = prof[u1[0]]; /* last row: cur[] is never read again */
		j = 1;
#if defined(MAFFT_A64)
		{
			int32x4_t vpen = vdupq_n_s32( pen ), vext = vdupq_n_s32( ext ), vthr = vdupq_n_s32( ithr );
			int32x4_t vstop = vdupq_n_s32( lstop ), vi = vdupq_n_s32( i ), vi1 = vdupq_n_s32( negi1 );
			int32x4_t vmax = vdupq_n_s32( INT_MIN ), vfour = vdupq_n_s32( 4 );
			int32x4_t vj = { 1, 2, 3, 4 };
			int32x4_t ninf = vdupq_n_s32( INT_MIN ), none = vdupq_n_s32( -1 );
			int32x4_t cv = ninf, ck = vdupq_n_s32( 0 );
			int32x4_t kv = { -1, 0, 1, 2 };
			int32x4_t kext = vmulq_s32( kv, vext ), kext1 = vaddq_s32( kext, vext ), ext4 = vmulq_s32( vfour, vext );
			for( ; j+3<=lgth2; j+=4 )
			{
				int32x4_t p = vld1q_s32( prev + j - 1 );
				int32x4_t hqv, hkv;
				{
					int32x4_t v = vsubq_s32( vld1q_s32( prev + j - 2 ), kext );
					int32x4_t x = vmaxq_s32( v, vextq_s32( ninf, v, 3 ) );
					int32x4_t e, r, t;
					x = vmaxq_s32( x, vextq_s32( ninf, x, 2 ) );
					x = vmaxq_s32( x, cv );
					e = vextq_s32( cv, x, 3 );
					r = vbslq_s32( vcgtq_s32( v, e ), kv, none );
					t = vmaxq_s32( r, vextq_s32( none, r, 3 ) );
					t = vmaxq_s32( t, vextq_s32( none, t, 2 ) );
					t = vmaxq_s32( t, ck );
					hqv = vaddq_s32( x, kext1 );
					hkv = t;
					cv = vdupq_laneq_s32( x, 3 );
					ck = vdupq_laneq_s32( t, 3 );
					kv = vaddq_s32( kv, vfour );
					kext = vaddq_s32( kext, ext4 );
					kext1 = vaddq_s32( kext1, ext4 );
				}
				int32x4_t g, wm, ij, m, vmj, vmpj;
				uint32x4_t c;
				wm = p;
				ij = vdupq_n_s32( 0 );
				g = vaddq_s32( hqv, vpen );
				c = vcgtq_s32( g, wm );
				wm = vmaxq_s32( wm, g );
				ij = vbslq_s32( c, vsubq_s32( hkv, vj ), ij );
				vmj = vld1q_s32( vm + j );
				vmpj = vld1q_s32( vmp + j );
				g = vaddq_s32( vmj, vpen );
				c = vcgtq_s32( g, wm );
				wm = vmaxq_s32( wm, g );
				ij = vbslq_s32( c, vsubq_s32( vi, vmpj ), ij );
				c = vcgtq_s32( p, vmj );
				m = vmaxq_s32( p, vmj );
				vst1q_s32( vm + j, vaddq_s32( m, vext ) );
				vst1q_s32( vmp + j, vbslq_s32( c, vi1, vmpj ) );
				vst1q_s32( wmrow + j, wm );
				vmax = vmaxq_s32( vmax, wm );
				c = vcltq_s32( wm, vthr );
				ij = vbslq_s32( c, vstop, ij );
				wm = vmaxq_s32( wm, vthr );
				vst1q_s32( ijrow + j, ij );
				vst1q_s32( cur + j, vaddq_s32( wm, vld1q_s32( profrow + j ) ) );
				vj = vaddq_s32( vj, vfour );
			}
			rowmax = vmaxvq_s32( vmax );
			if( j > 1 ) { tbest = vgetq_lane_s32( cv, 0 ); tbestk = vgetq_lane_s32( ck, 0 ); }
		}
#elif defined(MAFFT_SSE41)
		/* The NEON block above, lane for lane: palignr for vextq, pblendvb for vbslq. */
		{
			__m128i vpen = _mm_set1_epi32( pen ), vext = _mm_set1_epi32( ext ), vthr = _mm_set1_epi32( ithr );
			__m128i vstop = _mm_set1_epi32( lstop ), vi = _mm_set1_epi32( i ), vi1 = _mm_set1_epi32( negi1 );
			__m128i vmax = _mm_set1_epi32( INT_MIN ), vfour = _mm_set1_epi32( 4 );
			__m128i vj = _mm_setr_epi32( 1, 2, 3, 4 );
			__m128i ninf = _mm_set1_epi32( INT_MIN ), none = _mm_set1_epi32( -1 );
			__m128i cv = ninf, ck = _mm_setzero_si128();
			__m128i kv = _mm_setr_epi32( -1, 0, 1, 2 );
			__m128i kext = _mm_mullo_epi32( kv, vext ), kext1 = _mm_add_epi32( kext, vext ), ext4 = _mm_mullo_epi32( vfour, vext );
			for( ; j+3<=lgth2; j+=4 )
			{
				__m128i p = _mm_loadu_si128( (__m128i *)( prev + j - 1 ) );
				__m128i hqv, hkv;
				{
					__m128i v = _mm_sub_epi32( _mm_loadu_si128( (__m128i *)( prev + j - 2 ) ), kext );
					__m128i x = _mm_max_epi32( v, _mm_alignr_epi8( v, ninf, 12 ) );
					__m128i e, r, t;
					x = _mm_max_epi32( x, _mm_alignr_epi8( x, ninf, 8 ) );
					x = _mm_max_epi32( x, cv );
					e = _mm_alignr_epi8( x, cv, 12 );
					r = _mm_blendv_epi8( none, kv, _mm_cmpgt_epi32( v, e ) );
					t = _mm_max_epi32( r, _mm_alignr_epi8( r, none, 12 ) );
					t = _mm_max_epi32( t, _mm_alignr_epi8( t, none, 8 ) );
					t = _mm_max_epi32( t, ck );
					hqv = _mm_add_epi32( x, kext1 );
					hkv = t;
					cv = _mm_shuffle_epi32( x, 0xff );
					ck = _mm_shuffle_epi32( t, 0xff );
					kv = _mm_add_epi32( kv, vfour );
					kext = _mm_add_epi32( kext, ext4 );
					kext1 = _mm_add_epi32( kext1, ext4 );
				}
				__m128i g, wm, ij, m, vmj, vmpj, c;
				wm = p;
				ij = _mm_setzero_si128();
				g = _mm_add_epi32( hqv, vpen );
				c = _mm_cmpgt_epi32( g, wm );
				wm = _mm_max_epi32( wm, g );
				ij = _mm_blendv_epi8( ij, _mm_sub_epi32( hkv, vj ), c );
				vmj = _mm_loadu_si128( (__m128i *)( vm + j ) );
				vmpj = _mm_loadu_si128( (__m128i *)( vmp + j ) );
				g = _mm_add_epi32( vmj, vpen );
				c = _mm_cmpgt_epi32( g, wm );
				wm = _mm_max_epi32( wm, g );
				ij = _mm_blendv_epi8( ij, _mm_sub_epi32( vi, vmpj ), c );
				c = _mm_cmpgt_epi32( p, vmj );
				m = _mm_max_epi32( p, vmj );
				_mm_storeu_si128( (__m128i *)( vm + j ), _mm_add_epi32( m, vext ) );
				_mm_storeu_si128( (__m128i *)( vmp + j ), _mm_blendv_epi8( vmpj, vi1, c ) );
				_mm_storeu_si128( (__m128i *)( wmrow + j ), wm );
				vmax = _mm_max_epi32( vmax, wm );
				c = _mm_cmpgt_epi32( vthr, wm );
				ij = _mm_blendv_epi8( ij, vstop, c );
				wm = _mm_max_epi32( wm, vthr );
				_mm_storeu_si128( (__m128i *)( ijrow + j ), ij );
				_mm_storeu_si128( (__m128i *)( cur + j ), _mm_add_epi32( wm, _mm_loadu_si128( (__m128i *)( profrow + j ) ) ) );
				vj = _mm_add_epi32( vj, vfour );
			}
			vmax = _mm_max_epi32( vmax, _mm_shuffle_epi32( vmax, 0x4e ) );
			vmax = _mm_max_epi32( vmax, _mm_shuffle_epi32( vmax, 0xb1 ) );
			rowmax = _mm_cvtsi128_si32( vmax );
			if( j > 1 ) { tbest = _mm_cvtsi128_si32( cv ); tbestk = _mm_cvtsi128_si32( ck ); }
		}
#endif
		/* scan for the remaining cells */
		for( jj=j; jj<=lgth2; jj++ )
		{
			if( jj >= 2 )
			{
				int q = prev[jj-2] - ( jj-2 ) * ext;
				if( q > tbest ) { tbest = q; tbestk = jj-2; }
			}
			hq[jj] = tbest + ( jj-1 ) * ext;
			hk[jj] = tbestk;
		}
		for( ; j<=lgth2; j++ )
		{
			int p = prev[j-1];
			int wm = p, ij = 0, g;
			g = hq[j] + pen;
			if( g > wm ) { wm = g; ij = -( j - hk[j] ); }
			g = vm[j] + pen;
			if( g > wm ) { wm = g; ij = i - vmp[j]; }
			if( p > vm[j] ) { vm[j] = p; vmp[j] = negi1; }
			vm[j] += ext;
			wmrow[j] = wm;
			rowmax = ( wm > rowmax ) ? wm : rowmax;
			if( wm < ithr ) { ij = lstop; wm = ithr; }
			ijrow[j] = ij;
			cur[j] = wm + profrow[j];
		}
		if( rowmax > maxwm )
		{
			j = 1;
#if defined(MAFFT_SSE41)
			{
				__m128i vr = _mm_set1_epi32( rowmax );
				for( ; j+3<=lgth2; j+=4 )
				{
					int k = _mm_movemask_ps( _mm_castsi128_ps( _mm_cmpeq_epi32( _mm_loadu_si128( (__m128i *)( wmrow + j ) ), vr ) ) );
					if( k ) { j += __builtin_ctz( (unsigned)k ); break; }
				}
			}
#endif
			for( ; wmrow[j] != rowmax; j++ )
				;
			maxwm = rowmax; endali = i; endalj = j;
		}
		if( i < lgth1 ) cur[0] = (int)amino_dynamicmtx[u2[0]][u1[i]]; /* currentw[0] = initverticalw[i] */
	}

	free( profbuf ); free( pbuf1 ); free( pbuf2 ); free( vm ); free( vmp ); free( hq ); free( hk ); free( wmrow );
	*maxwmpt = (double)maxwm;
	*endalipt = endali;
	*endaljpt = endalj;
	return( 1 );
}
#endif

double L__align11( double **n_dynamicmtx, double scoreoffset, char **seq1, char **seq2, int alloclen, int *off1pt, int *off2pt )
/* score no keisan no sai motokaraaru gap no atukai ni mondai ga aru */
{
//	int k;
	int i, j;
	int lasti, lastj;                      /* outgap == 0 -> lgth1, outgap == 1 -> lgth1+1 */
	int lgth1, lgth2;
	int resultlen;
	double wm = 0.0;   /* int ?????? */
	double g;
	double *currentw, *previousw;
#if 1
	double *wtmp;
	int *ijppt;
	double *mjpt, *prept, *curpt;
	int *mpjpt;
#endif
	static TLS double mi, *m;
	static TLS int **ijp;
	static TLS int mpi, *mp;
	static TLS double *w1, *w2;
	static TLS double *match;
	static TLS double *initverticalw;    /* kufuu sureba iranai */
	static TLS double *lastverticalw;    /* kufuu sureba iranai */
	static TLS char **mseq1;
	static TLS char **mseq2;
	static TLS char **mseq;
//	static TLS int **intwork;
//	static TLS double **doublework;
	static TLS int orlgth1 = 0, orlgth2 = 0;
	static TLS double **amino_dynamicmtx = NULL; // ??
	double maxwm;
	int endali = 0, endalj = 0; // by D.Mathog, a guess
//	int endali, endalj;
	double localthr = -offset + scoreoffset * 600; // 2013/12/13
	double localthr2 = -offset + scoreoffset * 600; // 2013/12/13
//	double localthr = -offset;
//	double localthr2 = -offset;
	double fpenalty = (double)penalty;
	double fpenalty_ex = (double)penalty_ex;
	double fpenalty_shift = (double)penalty_shift;
	double fpenalty_tmp; // atode kesu

	int *warpis = NULL;
	int *warpjs = NULL;
	int *warpi = NULL;
	int *warpj = NULL;
	int *prevwarpi = NULL;
	int *prevwarpj = NULL;
	double *wmrecords = NULL;
	double *prevwmrecords = NULL;
	int warpn = 0;
	int warpbase;
#if defined(MAFFT_AVX512X)
	int kbfill = 0;
#endif
	double curm = 0.0;
	double *wmrecordspt, *wmrecords1pt, *prevwmrecordspt;
	int *warpipt, *warpjpt;



	if( seq1 == NULL )
	{
		if( orlgth1 > 0 && orlgth2 > 0 )
		{
			orlgth1 = 0;
			orlgth2 = 0;
			free( mseq1 );
			free( mseq2 );
			FreeFloatVec( w1 );
			FreeFloatVec( w2 );
			FreeFloatVec( match );
			FreeFloatVec( initverticalw );
			FreeFloatVec( lastverticalw );

			FreeFloatVec( m );
			FreeIntVec( mp );

			FreeCharMtx( mseq );
			if( amino_dynamicmtx ) FreeDoubleMtx( amino_dynamicmtx ); amino_dynamicmtx = NULL;

		}
		return( 0.0 );
	}


	if( orlgth1 == 0 )
	{
		mseq1 = AllocateCharMtx( njob, 0 );
		mseq2 = AllocateCharMtx( njob, 0 );
	}


	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );


	warpbase = lgth1 + lgth2;
	warpis = NULL;
	warpjs = NULL;
	warpn = 0;
	if( trywarp )
	{
		wmrecords = AllocateFloatVec( lgth2+1 );
		warpi = AllocateIntVec( lgth2+1 );
		warpj = AllocateIntVec( lgth2+1 );
		prevwmrecords = AllocateFloatVec( lgth2+1 );
		prevwarpi = AllocateIntVec( lgth2+1 );
		prevwarpj = AllocateIntVec( lgth2+1 );
		for( i=0; i<lgth2+1; i++ ) prevwmrecords[i] = 0.0;
		for( i=0; i<lgth2+1; i++ ) wmrecords[i] = 0.0;
		for( i=0; i<lgth2+1; i++ ) prevwarpi[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) prevwarpj[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) warpi[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) warpj[i] = -warpbase;
	}


	if( lgth1 > orlgth1 || lgth2 > orlgth2 )
	{
		int ll1, ll2;

		if( orlgth1 > 0 && orlgth2 > 0 )
		{
			FreeFloatVec( w1 );
			FreeFloatVec( w2 );
			FreeFloatVec( match );
			FreeFloatVec( initverticalw );
			FreeFloatVec( lastverticalw );

			FreeFloatVec( m );
			FreeIntVec( mp );

			FreeCharMtx( mseq );
			if( amino_dynamicmtx ) FreeDoubleMtx( amino_dynamicmtx ); amino_dynamicmtx = NULL;


//			FreeFloatMtx( doublework );
//			FreeIntMtx( intwork );
		}

		ll1 = MAX( (int)(1.3*lgth1), orlgth1 ) + 100;
		ll2 = MAX( (int)(1.3*lgth2), orlgth2 ) + 100;

#if DEBUG
		fprintf( stderr, "\ntrying to allocate (%d+%d)xn matrices ... ", ll1, ll2 );
#endif

		w1 = AllocateFloatVec( ll2+2 );
		w2 = AllocateFloatVec( ll2+2 );
		match = AllocateFloatVec( ll2+2 );

		initverticalw = AllocateFloatVec( ll1+2 );
		lastverticalw = AllocateFloatVec( ll1+2 );

		m = AllocateFloatVec( ll2+2 );
		mp = AllocateIntVec( ll2+2 );

		mseq = AllocateCharMtx( njob, ll1+ll2 );


//		doublework = AllocateFloatMtx( nalphabets, MAX( ll1, ll2 )+2 ); 
//		intwork = AllocateIntMtx( nalphabets, MAX( ll1, ll2 )+2 ); 

#if DEBUG
		fprintf( stderr, "succeeded\n" );
#endif
		amino_dynamicmtx = AllocateDoubleMtx( 0x100, 0x100 );
		orlgth1 = ll1 - 100;
		orlgth2 = ll2 - 100;
	}

    for( i=0; i<nalphabets; i++) for( j=0; j<nalphabets; j++ )
		amino_dynamicmtx[(int)amino[i]][(int)amino[j]] = (double)n_dynamicmtx[i][j];


	mseq1[0] = mseq[0];
	mseq2[0] = mseq[1];


	if( orlgth1 > commonAlloc1 || orlgth2 > commonAlloc2 )
	{
		int ll1, ll2;

		if( commonAlloc1 && commonAlloc2 )
		{
			FreeIntMtx( commonIP );
		}

		ll1 = MAX( orlgth1, commonAlloc1 );
		ll2 = MAX( orlgth2, commonAlloc2 );

#if DEBUG
		fprintf( stderr, "\n\ntrying to allocate %dx%d matrices ... ", ll1+1, ll2+1 );
#endif

		commonIP = AllocateIntMtx( ll1+10, ll2+10 );

#if DEBUG
		fprintf( stderr, "succeeded\n\n" );
#endif

		commonAlloc1 = ll1;
		commonAlloc2 = ll2;
	}
	ijp = commonIP;

#if defined(MAFFT_AVX512X)
	if( !trywarp )
	{
		localstop = lgth1+lgth2+1;
		if( Lfill_kb( amino_dynamicmtx, n_dynamicmtx, scoreoffset, seq1[0], seq2[0], lgth1, lgth2, &maxwm, &endali, &endalj ) )
		{
			kbfill = 1;
			goto Lint_done;
		}
	}
#endif
	if( !trywarp )
	{
		localstop = lgth1+lgth2+1;
		if( Lfill_int( amino_dynamicmtx, n_dynamicmtx, scoreoffset, seq1[0], seq2[0], lgth1, lgth2, ijp, localstop, &maxwm, &endali, &endalj ) )
			goto Lint_done;
	}


#if 0
	for( i=0; i<lgth1; i++ ) 
		fprintf( stderr, "ogcp1[%d]=%f\n", i, ogcp1[i] );
#endif

	currentw = w1;
	previousw = w2;

	match_calc_mtx( amino_dynamicmtx, initverticalw, seq2, seq1, 0, lgth1 );

	match_calc_mtx( amino_dynamicmtx, currentw, seq1, seq2, 0, lgth2 );


	lasti = lgth2+1;
	for( j=1; j<lasti; ++j ) 
	{
		m[j] = currentw[j-1]; mp[j] = 0;
#if 0
		if( m[j] < localthr ) m[j] = localthr2;
#endif
	}

	lastverticalw[0] = currentw[lgth2-1];

	lasti = lgth1+1;

#if 0
fprintf( stderr, "currentw = \n" );
for( i=0; i<lgth1+1; i++ )
{
	fprintf( stderr, "%5.2f ", currentw[i] );
}
fprintf( stderr, "\n" );
fprintf( stderr, "initverticalw = \n" );
for( i=0; i<lgth2+1; i++ )
{
	fprintf( stderr, "%5.2f ", initverticalw[i] );
}
fprintf( stderr, "\n" );
#endif
#if DEBUG2
	fprintf( stderr, "\n" );
	fprintf( stderr, "       " );
	for( j=0; j<lgth2; j++ )
		fprintf( stderr, "%c     ", seq2[0][j] );
	fprintf( stderr, "\n" );
#endif

	localstop = lgth1+lgth2+1;
	maxwm = -999999999.9;
#if DEBUG2
	fprintf( stderr, "\n" );
	fprintf( stderr, "%c   ", seq1[0][0] );

	for( j=0; j<lgth2+1; j++ )
		fprintf( stderr, "%5.0f ", currentw[j] );
	fprintf( stderr, "\n" );
#endif

	for( i=1; i<lasti; i++ )
	{
		wtmp = previousw; 
		previousw = currentw;
		currentw = wtmp;

		previousw[0] = initverticalw[i-1];

		match_calc_mtx( amino_dynamicmtx, currentw, seq1, seq2, i, lgth2 );
#if DEBUG2
		fprintf( stderr, "%c   ", seq1[0][i] );
		fprintf( stderr, "%5.0f ", currentw[0] );
#endif

#if XXXXXXX
fprintf( stderr, "\n" );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
#if XXXXXXX
fprintf( stderr, "\n" );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
		currentw[0] = initverticalw[i];

		mi = previousw[0]; mpi = 0;

#if 0
		if( mi < localthr ) mi = localthr2;
#endif

		ijppt = ijp[i] + 1;
		mjpt = m + 1;
		prept = previousw;
		curpt = currentw + 1;
		mpjpt = mp + 1;
		lastj = lgth2+1;

		if( trywarp )
		{
			prevwmrecordspt = prevwmrecords;
			wmrecordspt = wmrecords+1;
			wmrecords1pt = wmrecords;
			warpipt = warpi + 1;
			warpjpt = warpj + 1;
		}
		for( j=1; j<lastj; j++ )
		{
			wm = *prept;
			*ijppt = 0;

#if 0
			fprintf( stderr, "%5.0f->", wm );
#endif
#if 0
			fprintf( stderr, "%5.0f?", g );
#endif
			if( (g=mi+fpenalty) > wm )
			{
				wm = g;
				*ijppt = -( j - mpi );
			}
			if( *prept > mi )
			{
				mi = *prept;
				mpi = j-1;
			}

#if USE_PENALTY_EX
			mi += fpenalty_ex;
#endif

#if 0 
			fprintf( stderr, "%5.0f?", g );
#endif
			if( (g=*mjpt+fpenalty) > wm )
			{
				wm = g;
				*ijppt = +( i - *mpjpt );
			}
			if( *prept > *mjpt )
			{
				*mjpt = *prept;
				*mpjpt = i-1;
			}
#if USE_PENALTY_EX
			*mjpt += fpenalty_ex;
#endif

			if( maxwm < wm )
			{
				maxwm = wm;
				endali = i;
				endalj = j;
			}
#if 1
			if( wm < localthr )
			{
//				fprintf( stderr, "stop i=%d, j=%d, curpt=%f, localthr = %f\n", i, j, *curpt, localthr );
				*ijppt = localstop;
				wm = localthr2;
			}
#endif
#if 0
			fprintf( stderr, "%5.0f ", *curpt );
#endif
#if 0
			fprintf( stderr, "wm (%d,%d) = %5.0f\n", i, j, wm );
//			fprintf( stderr, "%c-%c *ijppt = %d, localstop = %d\n", seq1[0][i], seq2[0][j], *ijppt, localstop );
#endif
			if( trywarp )
			{
				fpenalty_tmp = fpenalty_shift + fpenalty_ex * ( i - prevwarpi[j-1] + j - prevwarpj[j-1] );
//				fprintf( stderr, "fpenalty_shift = %f\n", fpenalty_tmp );

//				fprintf( stderr, "\n\n\nwarp to %c-%c (%d-%d) from %c-%c (%d-%d) ? prevwmrecords[%d] = %f + %f <- wm = %f\n", seq1[0][prevwarpi[j-1]], seq2[0][prevwarpj[j-1]], prevwarpi[j-1], prevwarpj[j-1], seq1[0][i], seq2[0][j], i, j, j, prevwmrecords[j-1], fpenalty_tmp, wm );
//				if( (g=prevwmrecords[j-1] + fpenalty_shift )> wm )
				if( ( g=*prevwmrecordspt++ + fpenalty_tmp )> wm ) // naka ha osokute kamawanai
				{
//					fprintf( stderr, "Yes! Warp!! from %d-%d (%c-%c) to  %d-%d (%c-%c) fpenalty_tmp = %f! warpn = %d\n", i, j, seq1[0][i], seq2[0][j-1], prevwarpi[j-1], prevwarpj[j-1],seq1[0][prevwarpi[j-1]], seq2[0][prevwarpj[j-1]], fpenalty_tmp, warpn );
					if( warpn && prevwarpi[j-1] == warpis[warpn-1] && prevwarpj[j-1] == warpjs[warpn-1] )
					{
						*ijppt = warpbase + warpn - 1;
					}
					else
					{
						*ijppt = warpbase + warpn;
						warpis = realloc( warpis, sizeof(int) * ( warpn+1 ) );
						warpjs = realloc( warpjs, sizeof(int) * ( warpn+1 ) );
						warpis[warpn] = prevwarpi[j-1];
						warpjs[warpn] = prevwarpj[j-1];
						warpn++;
					}
					wm = g;
				}
				else
				{
				}

				curm = *curpt + wm;
	
//				fprintf( stderr, "###### curm = %f at %c-%c, i=%d, j=%d\n", curm, seq1[0][i], seq2[0][j], i, j ); 
	
//				fprintf( stderr, "copy from i, j-1? %f > %f?\n", wmrecords[j-1], curm );
//				if( wmrecords[j-1] > wmrecords[j] )
				if( *wmrecords1pt > *wmrecordspt )
				{
//					fprintf( stderr, "yes\n" );
//					wmrecords[j] = wmrecords[j-1];
					*wmrecordspt = *wmrecords1pt;
//					warpi[j] = warpi[j-1];
//					warpj[j] = warpj[j-1];
					*warpipt  = *(warpipt-1);
					*warpjpt  = *(warpjpt-1);
//					fprintf( stderr, "warpi[j]=%d, warpj[j]=%d wmrecords[j] = %f\n", warpi[j], warpj[j], wmrecords[j] );
				}
//				else
//				{
//					fprintf( stderr, "no\n" );
//				}
	
//				fprintf( stderr, " curm = %f at %c-%c\n", curm, seq1[0][i], seq2[0][j] ); 
//				fprintf( stderr, " wmrecords[%d] = %f\n", j, wmrecords[j] ); 
//				fprintf( stderr, "replace?\n" );
	
//				if( curm > wmrecords[j] )
				if( curm > *wmrecordspt )
				{
//					fprintf( stderr, "yes at %d-%d (%c-%c), replaced warp: warpi[j]=%d, warpj[j]=%d warpn=%d, wmrecords[j] = %f -> %f\n", i, j, seq1[0][i], seq2[0][j], i, j, warpn, wmrecords[j], curm );
//					wmrecords[j] = curm;
					*wmrecordspt = curm;
//					warpi[j] = i;
//					warpj[j] = j;
					*warpipt = i;
					*warpjpt = j;
				}
//				else
//				{
//					fprintf( stderr, "No! warpi[j]=%d, warpj[j]=%d wmrecords[j] = %f\n", warpi[j], warpj[j], wmrecords[j] );
//				}
//				fprintf( stderr, "%d-%d (%c-%c) curm = %5.0f, wmrecords[j]=%f\n", i, j, seq1[0][i], seq2[0][j], curm, wmrecords[j] );
				wmrecordspt++;
				wmrecords1pt++;
				warpipt++;
				warpjpt++;
			}

			*curpt++ += wm;
			ijppt++;
			mjpt++;
			prept++;
			mpjpt++;
		}
#if DEBUG2
		fprintf( stderr, "\n" );
#endif

		lastverticalw[i] = currentw[lgth2-1];
		if( trywarp )
		{
			fltncpy( prevwmrecords, wmrecords, lastj );
			intncpy( prevwarpi, warpi, lastj );
			intncpy( prevwarpj, warpj, lastj );
		}

	}
//	fprintf( stderr, "\nwm = %f\n", wm );
	if( trywarp )
	{
//		if( warpn ) fprintf( stderr, "warpn = %d\n", warpn );
		free( wmrecords );
		free( prevwmrecords );
		free( warpi );
		free( warpj );
		free( prevwarpi );
		free( prevwarpj );
	}

#if 0
	fprintf( stderr, "maxwm = %f\n", maxwm );
	fprintf( stderr, "endali = %d\n", endali );
	fprintf( stderr, "endalj = %d\n", endalj );
#endif

Lint_done:
#if defined(MAFFT_AVX512X)
	if( kbfill ? KB_ISSTOP( endali, endalj ) : ( ijp[endali][endalj] == localstop ) )
#else
	if( ijp[endali][endalj] == localstop )
#endif
	{
		strcpy( seq1[0], "" );
		strcpy( seq2[0], "" );
		*off1pt = *off2pt = 0;
		fprintf( stderr, "maxwm <- 0.0 \n" );
		return( 0.0 );
	}
		
#if defined(MAFFT_AVX512X)
	if( kbfill )
		Ltracking_kb( seq1, seq2, mseq1, mseq2, off1pt, off2pt, endali, endalj, localstop, warpis, warpjs, warpbase );
	else
#endif
	Ltracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, off1pt, off2pt, endali, endalj, warpis, warpjs, warpbase );
	if( warpis ) free( warpis );
	if( warpjs ) free( warpjs );


	resultlen = strlen( mseq1[0] );
	if( alloclen < resultlen || resultlen > N )
	{
		fprintf( stderr, "alloclen=%d, resultlen=%d, N=%d\n", alloclen, resultlen, N );
		ErrorExit( "LENGTH OVER!\n" );
	}


	strcpy( seq1[0], mseq1[0] );
	strcpy( seq2[0], mseq2[0] );

#if 0
	fprintf( stderr, "wm=%f\n", wm );
	fprintf( stderr, ">\n%s\n", mseq1[0] );
	fprintf( stderr, ">\n%s\n", mseq2[0] );

	fprintf( stderr, "*off1pt = %d, *off2pt = %d\n", *off1pt, *off2pt );

	fprintf( stderr, "maxwm = %f\n", maxwm );
	fprintf( stderr, "   wm = %f\n",    wm );
#endif

	return( maxwm );
}


double L__align11_noalign( double **n_dynamicmtx, char **seq1, char **seq2 )
// warp mitaiou
{
//	int k;
	int i, j;
	int lasti, lastj;                      /* outgap == 0 -> lgth1, outgap == 1 -> lgth1+1 */
	int lgth1, lgth2;
//	int resultlen;
	double wm = 0.0;   /* int ?????? */
	double g;
	double *currentw, *previousw;
#if 1
	double *wtmp;
//	int *ijppt;
	double *mjpt, *prept, *curpt;
//	int *mpjpt;
#endif
	static TLS double mi, *m;
//	static TLS int **ijp;
//	static TLS int mpi, *mp;
	static TLS double *w1, *w2;
	static TLS double *match;
	static TLS double *initverticalw;    /* kufuu sureba iranai */
	static TLS double *lastverticalw;    /* kufuu sureba iranai */
//	static TLS char **mseq1;
//	static TLS char **mseq2;
//	static TLS char **mseq;
//	static TLS int **intwork;
//	static TLS double **doublework;
	static TLS int orlgth1 = 0, orlgth2 = 0;
	static TLS double **amino_dynamicmtx = NULL; // ??
	double maxwm;
//	int endali = 0, endalj = 0; // by D.Mathog, a guess
//	int endali, endalj;
	double localthr = -offset;
	double localthr2 = -offset;
//	double localthr = 100;
//	double localthr2 = 100;
	double fpenalty = (double)penalty;
	double fpenalty_ex = (double)penalty_ex;

	if( seq1 == NULL )
	{
		if( orlgth1 > 0 && orlgth2 > 0 )
		{
			orlgth1 = 0;
			orlgth2 = 0;
//			free( mseq1 );
//			free( mseq2 );
			FreeFloatVec( w1 );
			FreeFloatVec( w2 );
			FreeFloatVec( match );
			FreeFloatVec( initverticalw );
			FreeFloatVec( lastverticalw );

			FreeFloatVec( m );
//			FreeIntVec( mp );

//			FreeCharMtx( mseq );
			if( amino_dynamicmtx ) FreeDoubleMtx( amino_dynamicmtx ); amino_dynamicmtx = NULL;

		}
		return( 0.0 );
	}


//	if( orlgth1 == 0 )
//	{
//		mseq1 = AllocateCharMtx( njob, 0 );
//		mseq2 = AllocateCharMtx( njob, 0 );
//	}


	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );

	if( lgth1 > orlgth1 || lgth2 > orlgth2 )
	{
		int ll1, ll2;

		if( orlgth1 > 0 && orlgth2 > 0 )
		{
			FreeFloatVec( w1 );
			FreeFloatVec( w2 );
			FreeFloatVec( match );
			FreeFloatVec( initverticalw );
			FreeFloatVec( lastverticalw );

			FreeFloatVec( m );
//			FreeIntVec( mp );

//			FreeCharMtx( mseq );



//			FreeFloatMtx( doublework );
//			FreeIntMtx( intwork );
			if( amino_dynamicmtx ) FreeDoubleMtx( amino_dynamicmtx ); amino_dynamicmtx = NULL;
		}

		ll1 = MAX( (int)(1.3*lgth1), orlgth1 ) + 100;
		ll2 = MAX( (int)(1.3*lgth2), orlgth2 ) + 100;

#if DEBUG
		fprintf( stderr, "\ntrying to allocate (%d+%d)xn matrices ... ", ll1, ll2 );
#endif

		w1 = AllocateFloatVec( ll2+2 );
		w2 = AllocateFloatVec( ll2+2 );
		match = AllocateFloatVec( ll2+2 );

		initverticalw = AllocateFloatVec( ll1+2 );
		lastverticalw = AllocateFloatVec( ll1+2 );

		m = AllocateFloatVec( ll2+2 );
//		mp = AllocateIntVec( ll2+2 );

//		mseq = AllocateCharMtx( njob, ll1+ll2 );


//		doublework = AllocateFloatMtx( nalphabets, MAX( ll1, ll2 )+2 ); 
//		intwork = AllocateIntMtx( nalphabets, MAX( ll1, ll2 )+2 ); 

#if DEBUG
		fprintf( stderr, "succeeded\n" );
#endif
//		amino_dynamicmtx = AllocateDoubleMtx( 0x80, 0x80 ); 
		amino_dynamicmtx = AllocateDoubleMtx( 0x100, 0x100 );  // 2017/Nov.  constants.c no 'charsize' wo global hensuu nishita houga yoi?
		orlgth1 = ll1 - 100;
		orlgth2 = ll2 - 100;
	}

    for( i=0; i<nalphabets; i++) for( j=0; j<nalphabets; j++ )
		amino_dynamicmtx[(int)amino[i]][(int)amino[j]] = (double)n_dynamicmtx[i][j];



//	mseq1[0] = mseq[0];
//	mseq2[0] = mseq[1];


//	if( orlgth1 > commonAlloc1 || orlgth2 > commonAlloc2 )
//	{
//		int ll1, ll2;
//
//		if( commonAlloc1 && commonAlloc2 )
//		{
//			FreeIntMtx( commonIP );
//		}
//
//		ll1 = MAX( orlgth1, commonAlloc1 );
//		ll2 = MAX( orlgth2, commonAlloc2 );

#if DEBUG
//		fprintf( stderr, "\n\ntrying to allocate %dx%d matrices ... ", ll1+1, ll2+1 );
#endif

//		commonIP = AllocateIntMtx( ll1+10, ll2+10 );

#if DEBUG
//		fprintf( stderr, "succeeded\n\n" );
#endif

//		commonAlloc1 = ll1;
//		commonAlloc2 = ll2;
//	}
//	ijp = commonIP;


#if 0
	for( i=0; i<lgth1; i++ ) 
		fprintf( stderr, "ogcp1[%d]=%f\n", i, ogcp1[i] );
#endif

	currentw = w1;
	previousw = w2;

	match_calc_mtx( amino_dynamicmtx, initverticalw, seq2, seq1, 0, lgth1 );

	match_calc_mtx( amino_dynamicmtx, currentw, seq1, seq2, 0, lgth2 );


	lasti = lgth2+1;
	for( j=1; j<lasti; ++j ) 
	{
		m[j] = currentw[j-1]; 
//		mp[j] = 0;
#if 0
		if( m[j] < localthr ) m[j] = localthr2;
#endif
	}

	lastverticalw[0] = currentw[lgth2-1];

	lasti = lgth1+1;

#if 0
fprintf( stderr, "currentw = \n" );
for( i=0; i<lgth1+1; i++ )
{
	fprintf( stderr, "%5.2f ", currentw[i] );
}
fprintf( stderr, "\n" );
fprintf( stderr, "initverticalw = \n" );
for( i=0; i<lgth2+1; i++ )
{
	fprintf( stderr, "%5.2f ", initverticalw[i] );
}
fprintf( stderr, "\n" );
#endif
#if DEBUG2
	fprintf( stderr, "\n" );
	fprintf( stderr, "       " );
	for( j=0; j<lgth2; j++ )
		fprintf( stderr, "%c     ", seq2[0][j] );
	fprintf( stderr, "\n" );
#endif

	localstop = lgth1+lgth2+1;
	maxwm = -999999999.9;
#if DEBUG2
	fprintf( stderr, "\n" );
	fprintf( stderr, "%c   ", seq1[0][0] );

	for( j=0; j<lgth2+1; j++ )
		fprintf( stderr, "%5.0f ", currentw[j] );
	fprintf( stderr, "\n" );
#endif

	for( i=1; i<lasti; i++ )
	{
		wtmp = previousw; 
		previousw = currentw;
		currentw = wtmp;

		previousw[0] = initverticalw[i-1];

		match_calc_mtx( amino_dynamicmtx, currentw, seq1, seq2, i, lgth2 );
#if DEBUG2
		fprintf( stderr, "%c   ", seq1[0][i] );
		fprintf( stderr, "%5.0f ", currentw[0] );
#endif

#if XXXXXXX
fprintf( stderr, "\n" );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
#if XXXXXXX
fprintf( stderr, "\n" );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
		currentw[0] = initverticalw[i];

		mi = previousw[0]; 
//		mpi = 0;

#if 0
		if( mi < localthr ) mi = localthr2;
#endif

//		ijppt = ijp[i] + 1;
		mjpt = m + 1;
		prept = previousw;
		curpt = currentw + 1;
//		mpjpt = mp + 1;
		lastj = lgth2+1;
		for( j=1; j<lastj; j++ )
		{
			wm = *prept;
//			*ijppt = 0;

#if 0
			fprintf( stderr, "%5.0f->", wm );
#endif
#if 0
			fprintf( stderr, "%5.0f?", g );
#endif
			if( (g=mi+fpenalty) > wm )
			{
				wm = g;
//				*ijppt = -( j - mpi );
			}
			if( *prept > mi )
			{
				mi = *prept;
//				mpi = j-1;
			}

#if USE_PENALTY_EX
			mi += fpenalty_ex;
#endif

#if 0 
			fprintf( stderr, "%5.0f?", g );
#endif
			if( (g=*mjpt+fpenalty) > wm )
			{
				wm = g;
//				*ijppt = +( i - *mpjpt );
			}
			if( *prept > *mjpt )
			{
				*mjpt = *prept;
//				*mpjpt = i-1;
			}
#if USE_PENALTY_EX
			*mjpt += fpenalty_ex;
#endif

			if( maxwm < wm )
			{
				maxwm = wm;
//				endali = i;
//				endalj = j;
			}
#if 1
			if( wm < localthr )
			{
//				fprintf( stderr, "stop i=%d, j=%d, curpt=%f\n", i, j, *curpt );
//				*ijppt = localstop;
				wm = localthr2;
			}
#endif
#if 0
			fprintf( stderr, "%5.0f ", *curpt );
#endif
#if DEBUG2
			fprintf( stderr, "%5.0f ", wm );
//			fprintf( stderr, "%c-%c *ijppt = %d, localstop = %d\n", seq1[0][i], seq2[0][j], *ijppt, localstop );
#endif

			*curpt++ += wm;
//			ijppt++;
			mjpt++;
			prept++;
//			mpjpt++;
		}
#if DEBUG2
		fprintf( stderr, "\n" );
#endif

		lastverticalw[i] = currentw[lgth2-1];
	}


#if 0
	fprintf( stderr, "maxwm = %f\n", maxwm );
	fprintf( stderr, "endali = %d\n", endali );
	fprintf( stderr, "endalj = %d\n", endalj );
#endif


#if 0 // IRUKAMO!!!!
	if( ijp[endali][endalj] == localstop )
	{
		strcpy( seq1[0], "" );
		strcpy( seq2[0], "" );
		*off1pt = *off2pt = 0;
		fprintf( stderr, "maxwm <- 0.0 \n" );
		return( 0.0 );
	}
#else
	if( maxwm < localthr )
	{
		fprintf( stderr, "maxwm <- 0.0 \n" );
		return( 0.0 );
	}
#endif
		
//	Ltracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, off1pt, off2pt, endali, endalj );


//	resultlen = strlen( mseq1[0] );
//	if( alloclen < resultlen || resultlen > N )
//	{
//		fprintf( stderr, "alloclen=%d, resultlen=%d, N=%d\n", alloclen, resultlen, N );
//		ErrorExit( "LENGTH OVER!\n" );
//	}


//	strcpy( seq1[0], mseq1[0] );
//	strcpy( seq2[0], mseq2[0] );

#if 0
	fprintf( stderr, "wm=%f\n", wm );
	fprintf( stderr, ">\n%s\n", mseq1[0] );
	fprintf( stderr, ">\n%s\n", mseq2[0] );

	fprintf( stderr, "maxwm = %f\n", maxwm );
	fprintf( stderr, "   wm = %f\n",    wm );
#endif

	return( maxwm );
}
