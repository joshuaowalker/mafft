#include "mltaln.h"
#include "dp.h"
#if ( defined(__AVX512F__) || defined(__AVX2__) ) && !defined(__ARM_NEON)
#include <immintrin.h>
/* vector a*b+c rounded like MULADD */
#if MAFFT_STOCK_FMA
#define VMULADD(a,b,c) _mm512_fmadd_pd( (a), (b), (c) )
#else
#define VMULADD(a,b,c) _mm512_add_pd( _mm512_mul_pd( (a), (b) ), (c) )
#endif
#if defined(MAFFT_AVX2_PATHS)
#if MAFFT_STOCK_FMA
#define VMULADD4(a,b,c) _mm256_fmadd_pd( (a), (b), (c) )
#else
#define VMULADD4(a,b,c) _mm256_add_pd( _mm256_mul_pd( (a), (b) ), (c) )
#endif
/* 4 x 64-bit compare mask -> 4 x 32-bit mask */
#define PACKMASK4(c) _mm256_castsi256_si128( _mm256_permutevar8x32_epi32( _mm256_castpd_si256( c ), _mm256_setr_epi32( 0, 2, 4, 6, 0, 2, 4, 6 ) ) )
#endif
#endif

#define MACHIGAI 0
#define OUTGAP0TRY 1
#define DEBUG 0
#define XXXXXXX    0
#define USE_PENALTY_EX  0
#define FASTMATCHCALC 1

#if 0
static void st_OpeningGapCount( double *ogcp, int clus, char **seq, double *eff, int len )
{
	int i, j, gc, gb; 
	double feff;
	
	for( i=0; i<len; i++ ) ogcp[i] = 0.0;
	for( j=0; j<clus; j++ ) 
	{
		feff = (double)eff[j];
		gc = 0;
		for( i=0; i<len; i++ ) 
		{
			gb = gc;
			gc = ( seq[j][i] == '-' );
			{
				if( !gb *  gc ) ogcp[i] += feff;
			}
		}
	}
}

static void st_FinalGapCount( double *fgcp, int clus, char **seq, double *eff, int len )
{
	int i, j, gc, gb; 
	double feff;
	
	for( i=0; i<len; i++ ) fgcp[i] = 0.0;
	for( j=0; j<clus; j++ ) 
	{
		feff = (double)eff[j];
		gc = ( seq[j][0] == '-' );
		for( i=1; i<len+1; i++ ) 
		{
			gb = gc;
			gc = ( seq[j][i] == '-' );
			{
				if( gb * !gc ) fgcp[i-1] += feff;
			}
		}
	}
}
#endif


			
		

static TLS int impalloclen = 0;
static TLS double **impmtx = NULL;
static TLS int *improwlo = NULL, *improwhi = NULL;
static TLS int impclean = 0; /* 1: impmtx is zero outside [improwlo[i],improwhi[i]] */
double part_imp_match_out_sc( int i1, int j1 )
{
//	fprintf( stderr, "impalloclen = %d\n", impalloclen );
//	fprintf( stderr, "i1,j1=%d,%d -> impmtx=%f\n", i1, j1, impmtx[i1][j1] );
	return( impmtx[i1][j1] );
#if 0
	if( i1 == l1 || j1 == l2 ) return( 0.0 );
	return( impmtx[i1+start1][j1+start2] );
#endif
}
static void part_imp_match_out_vead_gapmap( double *imp, int i1, int lgth2, int start2, int *gapmap2 )
{
#if FASTMACHCALC
	double *pt = imp;
	int *gapmappt = gapmap2;
	while( lgth2-- )
		*pt++ += impmtx[i1][start2+*gapmappt++];
#else
	int j = 0;
#if defined(__AVX512F__)
	{
		double *row = impmtx[i1] + start2;
		for( ; j+7<lgth2; j+=8 )
		{
			__m512d g = _mm512_i32gather_pd( _mm256_loadu_si256( (__m256i *)( gapmap2 + j ) ), row, 8 );
			_mm512_storeu_pd( imp + j, _mm512_add_pd( _mm512_loadu_pd( imp + j ), g ) );
		}
	}
#endif
	for( ; j<lgth2; j++ )
	{
		imp[j] += impmtx[i1][start2+gapmap2[j]];
	}
#endif
}

static void part_imp_match_out_vead_tate_gapmap( double *imp, int j1, int lgth1, int start1, int *gapmap1 )
{
#if FASTMACHCALC
	double *pt = imp;
	int *gapmappt = gapmap1;
	while( lgth1-- )
		*pt++ = impmtx[start1+*gapmappt++][j1];
#else
	int i;
	for( i=0; i<lgth1; i++ )
	{
		imp[i] += impmtx[start1+gapmap1[i]][j1];
	}
#endif
}

#if 1
void part_imp_match_init_strict( double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2 )
{
//	int i, j, k1, k2, tmpint, start1, start2, end1, end2;
//	double effij;
//	double effij_kozo;
//	double effijx;
//	char *pt, *pt1, *pt2;
//	static TLS char *nocount1 = NULL;
//	static TLS char *nocount2 = NULL;
//	LocalHom *tmpptr;

	if( seq1 == NULL )
	{
		if( impmtx ) FreeFloatMtx( impmtx );
		impmtx = NULL;
		free( improwlo ); free( improwhi ); improwlo = improwhi = NULL; impclean = 0;
//		if( nocount1 ) free( nocount1 );
//		nocount1 = NULL;
//		if( nocount2 ) free( nocount2 );
//		nocount2 = NULL;
		
		return;
	}

	if( impalloclen < lgth1 + 2 || impalloclen < lgth2 + 2 )
	{
		if( impmtx ) FreeFloatMtx( impmtx );
//		if( nocount1 ) free( nocount1 );
//		if( nocount2 ) free( nocount2 );
		impalloclen = MAX( lgth1, lgth2 ) + 2;
		impmtx = AllocateFloatMtx( impalloclen, impalloclen );
//		nocount1 = AllocateCharVec( impalloclen );
//		nocount2 = AllocateCharVec( impalloclen );
		free( improwlo ); free( improwhi );
		improwlo = malloc( sizeof( int ) * impalloclen );
		improwhi = malloc( sizeof( int ) * impalloclen );
		impclean = 0;
	}

	/* Instead of zeroing lgth1 x lgth2 doubles on every call, zero the whole matrix once and
	   afterwards only the per-row column ranges written by the previous fillimp. */
	{
		int i, j;
		if( !impclean )
		{
			for( i=0; i<impalloclen; i++ ) for( j=0; j<impalloclen; j++ ) impmtx[i][j] = 0.0;
		}
		else
		{
			for( i=0; i<impalloclen; i++ ) for( j=improwlo[i]; j<=improwhi[i]; j++ ) impmtx[i][j] = 0.0;
		}
		for( i=0; i<impalloclen; i++ ) { improwlo[i] = impalloclen; improwhi[i] = -1; }
		impclean = 1;
	}
	fillimp_track( impmtx, imp, clus1, clus2, lgth1, lgth2, seq1, seq2, eff1, eff2, eff1_kozo, eff2_kozo, localhom, swaplist, forscore, orinum1, orinum2, improwlo, improwhi );
}
#else
#endif


void part_imp_rna( int nseq1, int nseq2, char **seq1, char **seq2, double *eff1, double *eff2, RNApair ***grouprna1, RNApair ***grouprna2, int *gapmap1, int *gapmap2, RNApair *additionalpair )
{
	foldrna( nseq1, nseq2, seq1, seq2, eff1, eff2, grouprna1, grouprna2, impmtx, gapmap1, gapmap2, additionalpair );
	impclean = 0; /* foldrna writes outside the tracked ranges */
}





/*
 * Columns of cpmx2 grouped by their ordered list of nonzero letters.  match_calc() adds, for each
 * column, scarr[l] * cpmx2[l][j] over the nonzero l in ascending order with fused multiply-adds;
 * within a group every column uses the same scarr[] scalars, so each group is a short vector loop
 * doing exactly the same fma sequence per column.
 */
static TLS int mc_len = -1, mc_alloc = 0, mc_ngrp = 0, mc_ready = 0;
static TLS int *mc_perm = NULL;      /* column at permuted position */
static TLS int *mc_gstart = NULL;    /* group g occupies positions [gstart[g], gstart[g+1]) */
static TLS int *mc_gcnt = NULL;      /* letters in group g */
static TLS int *mc_glet = NULL;      /* letters of group g at [goff[g], goff[g]+gcnt[g]) */
static TLS int *mc_goff = NULL;
static TLS double *mc_val = NULL;    /* value of the k-th letter of the column at position pos: mc_val[mc_vstart[g]+k*gsize+ (pos-gstart)] */
static TLS int *mc_vstart = NULL;
static TLS double *mc_scarr = NULL;

static void mc_build( double **cpmx2, int lgth2 )
{
	int j, l, g, k, n, cnt, *key, *colgrp, *keyoff, *keycnt, *fill;
	unsigned long long h, *hash;
	int nkey = 0, tabsize, *tab, keylen = 0;

	mc_ready = 0;
	if( mc_alloc < lgth2 + 2 )
	{
		free( mc_perm ); free( mc_gstart ); free( mc_gcnt ); free( mc_goff ); free( mc_vstart ); free( mc_glet ); free( mc_val );
		mc_alloc = lgth2 + 2;
		mc_perm = malloc( sizeof( int ) * mc_alloc );
		mc_gstart = malloc( sizeof( int ) * ( mc_alloc + 1 ) );
		mc_gcnt = malloc( sizeof( int ) * mc_alloc );
		mc_goff = malloc( sizeof( int ) * mc_alloc );
		mc_vstart = malloc( sizeof( int ) * mc_alloc );
		mc_glet = NULL; mc_val = NULL;
	}
	key = malloc( sizeof( int ) * ( lgth2 * nalphabets + 1 ) );
	colgrp = malloc( sizeof( int ) * ( lgth2 + 1 ) );
	keyoff = malloc( sizeof( int ) * ( lgth2 + 1 ) );
	keycnt = malloc( sizeof( int ) * ( lgth2 + 1 ) );
	hash = malloc( sizeof( unsigned long long ) * ( lgth2 + 1 ) );
	for( tabsize=64; tabsize < 2*lgth2+2; tabsize*=2 ) ;
	tab = malloc( sizeof( int ) * tabsize );
	for( k=0; k<tabsize; k++ ) tab[k] = -1;

	for( j=0; j<lgth2; j++ )
	{
		int *kj = key + keylen, slot;
		cnt = 0; h = 1469598103934665603ULL;
		for( l=0; l<nalphabets; l++ ) if( cpmx2[l][j] ) { kj[cnt++] = l; h = ( h ^ (unsigned long long)( l + 1 ) ) * 1099511628211ULL; }
		h = ( h ^ (unsigned long long)cnt ) * 1099511628211ULL;
		slot = (int)( h & (unsigned long long)( tabsize - 1 ) );
		for( ;; slot = ( slot + 1 ) & ( tabsize - 1 ) )
		{
			g = tab[slot];
			if( g < 0 )
			{
				g = nkey++;
				tab[slot] = g;
				keyoff[g] = keylen; keycnt[g] = cnt; hash[g] = h;
				keylen += cnt;
				break;
			}
			if( hash[g] == h && keycnt[g] == cnt && !memcmp( key + keyoff[g], kj, sizeof( int ) * cnt ) ) break;
		}
		colgrp[j] = g;
	}
	mc_ngrp = nkey;
	/* counting sort of columns by group, keeping ascending column order inside a group */
	for( g=0; g<=nkey; g++ ) mc_gstart[g] = 0;
	for( j=0; j<lgth2; j++ ) mc_gstart[colgrp[j]+1]++;
	for( g=0; g<nkey; g++ ) mc_gstart[g+1] += mc_gstart[g];
	fill = malloc( sizeof( int ) * ( nkey + 1 ) );
	for( g=0; g<nkey; g++ ) fill[g] = mc_gstart[g];
	for( j=0; j<lgth2; j++ ) mc_perm[fill[colgrp[j]]++] = j;
	free( mc_glet );
	mc_glet = malloc( sizeof( int ) * ( keylen + 1 ) );
	n = 0;
	for( g=0; g<nkey; g++ )
	{
		int gsize = mc_gstart[g+1] - mc_gstart[g];
		mc_gcnt[g] = keycnt[g];
		mc_goff[g] = keyoff[g];
		for( k=0; k<keycnt[g]; k++ ) mc_glet[keyoff[g]+k] = key[keyoff[g]+k];
		mc_vstart[g] = n;
		n += gsize * keycnt[g];
	}
	free( mc_val );
	mc_val = malloc( sizeof( double ) * ( n + 1 ) );
	for( g=0; g<nkey; g++ )
	{
		int gs = mc_gstart[g], gsize = mc_gstart[g+1] - gs;
		for( k=0; k<mc_gcnt[g]; k++ )
		{
			double *v = mc_val + mc_vstart[g] + (size_t)k * gsize;
			double *row = cpmx2[mc_glet[mc_goff[g]+k]];
			for( n=0; n<gsize; n++ ) v[n] = row[mc_perm[gs+n]];
		}
	}
	free( key ); free( colgrp ); free( keyoff ); free( keycnt ); free( hash ); free( tab ); free( fill );
	mc_len = lgth2;
	mc_ready = 1;
}

static void mc_match( double *match, double **cpmx1, int i1, int lgth2 )
{
	int g, k, l, j, n;
	double *scarr;
	static TLS int scalloc = 0;
	if( scalloc < nalphabets ) { free( mc_scarr ); scalloc = nalphabets; mc_scarr = malloc( sizeof( double ) * scalloc ); }
	scarr = mc_scarr;
	l = 0;
#if defined(__AVX512F__) && !defined(__ARM_NEON)
	/* 8 letters at a time; each letter's sum still runs over j in order */
	for( ; l+8<=nalphabets; l+=8 )
	{
		__m512d s = _mm512_setzero_pd();
		for( j=0; j<nalphabets; j++ )
			s = VMULADD( _mm512_loadu_pd( n_dis_consweight_multi[j] + l ), _mm512_set1_pd( cpmx1[j][i1] ), s );
		_mm512_storeu_pd( scarr + l, s );
	}
#elif defined(MAFFT_AVX2_PATHS)
	/* Only the letters present in column i1 contribute: s + (+-0) == s for every s the sum can
	   hold (it starts at +0 and a sum is never -0 under round-to-nearest), so skipping the zero
	   terms is exact (with FMA too: a*0 is an exact +-0).  The remaining terms are added in
	   ascending j, 4 letters at a time, each lane rounded like MULADD (VMULADD4). */
	{
		static TLS int *nzj = NULL, nzalloc = 0;
		static TLS double *nzc = NULL;
		int nz = 0;
		if( nzalloc < nalphabets ) { free( nzj ); free( nzc ); nzalloc = nalphabets; nzj = malloc( sizeof( int ) * nzalloc ); nzc = malloc( sizeof( double ) * nzalloc ); }
		for( j=0; j<nalphabets; j++ ) if( cpmx1[j][i1] != 0.0 ) { nzj[nz] = j; nzc[nz] = cpmx1[j][i1]; nz++; }
		for( ; l+4<=nalphabets; l+=4 )
		{
			__m256d s = _mm256_setzero_pd();
			for( k=0; k<nz; k++ )
				s = VMULADD4( _mm256_loadu_pd( n_dis_consweight_multi[nzj[k]] + l ), _mm256_set1_pd( nzc[k] ), s );
			_mm256_storeu_pd( scarr + l, s );
		}
		for( ; l<nalphabets; l++ )
		{
			double s = 0.0;
			for( k=0; k<nz; k++ )
				s = MULADD( n_dis_consweight_multi[nzj[k]][l], nzc[k], s );
			scarr[l] = s;
		}
	}
#endif
	for( ; l<nalphabets; l++ )
	{
		double s = 0.0;
		for( j=0; j<nalphabets; j++ )
			s = MULADD( n_dis_consweight_multi[j][l], cpmx1[j][i1], s );
		scarr[l] = s;
	}
	for( g=0; g<mc_ngrp; g++ )
	{
		int gs = mc_gstart[g], gsize = mc_gstart[g+1] - gs, cnt = mc_gcnt[g];
		int *let = mc_glet + mc_goff[g], *perm = mc_perm + gs;
		double *v = mc_val + mc_vstart[g];
		if( cnt == 0 ) { for( n=0; n<gsize; n++ ) match[perm[n]] = 0.0; continue; }
		n = 0;
		if( cnt == 1 )
		{
			double s0 = scarr[let[0]];
#if defined(__AVX512F__) && !defined(__ARM_NEON)
			__m512d vs0 = _mm512_set1_pd( s0 ), zero = _mm512_setzero_pd();
			for( ; n+8<=gsize; n+=8 )
				_mm512_i32scatter_pd( match, _mm256_loadu_si256( (__m256i *)( perm + n ) ), VMULADD( vs0, _mm512_loadu_pd( v + n ), zero ), 8 );
#endif
			for( ; n<gsize; n++ ) match[perm[n]] = MULADD( s0, v[n], 0.0 );
		}
		else if( cnt == 2 )
		{
			double s0 = scarr[let[0]], s1 = scarr[let[1]], *v1 = v + gsize;
#if defined(__AVX512F__) && !defined(__ARM_NEON)
			__m512d vs0 = _mm512_set1_pd( s0 ), vs1 = _mm512_set1_pd( s1 ), zero = _mm512_setzero_pd();
			for( ; n+8<=gsize; n+=8 )
				_mm512_i32scatter_pd( match, _mm256_loadu_si256( (__m256i *)( perm + n ) ),
				                      VMULADD( vs1, _mm512_loadu_pd( v1 + n ), VMULADD( vs0, _mm512_loadu_pd( v + n ), zero ) ), 8 );
#endif
			for( ; n<gsize; n++ ) match[perm[n]] = MULADD( s1, v1[n], MULADD( s0, v[n], 0.0 ) );
		}
		else
		{
#if defined(__AVX512F__) && !defined(__ARM_NEON)
			for( ; n+8<=gsize; n+=8 )
			{
				__m512d acc = _mm512_setzero_pd();
				for( k=0; k<cnt; k++ ) acc = VMULADD( _mm512_set1_pd( scarr[let[k]] ), _mm512_loadu_pd( v + (size_t)k*gsize + n ), acc );
				_mm512_i32scatter_pd( match, _mm256_loadu_si256( (__m256i *)( perm + n ) ), acc, 8 );
			}
#endif
			for( ; n<gsize; n++ )
			{
				double acc = 0.0;
				for( k=0; k<cnt; k++ ) acc = MULADD( scarr[let[k]], v[(size_t)k*gsize+n], acc );
				match[perm[n]] = acc;
			}
		}
	}
}

static void match_calc( double *match, double **cpmx1, double **cpmx2, int i1, int lgth2, double **doublework, int **intwork, int initialize )
{
	if( !initialize && mc_ready && mc_len == lgth2 )
	{
		mc_match( match, cpmx1, i1, lgth2 );
		return;
	}
#if FASTMATCHCALC
	int j, l;
//	double scarr[26];
	double **cpmxpd = doublework;
	int **cpmxpdn = intwork;
	double *matchpt, *cpmxpdpt, **cpmxpdptpt;
	int *cpmxpdnpt, **cpmxpdnptpt;
	double *scarr;
	scarr = calloc( nalphabets, sizeof( double ) );
	if( initialize )
	{
		int count = 0;
		for( j=0; j<lgth2; j++ )
		{
			count = 0;
			for( l=0; l<nalphabets; l++ )
			{
				if( cpmx2[l][j] )
				{
					cpmxpd[j][count] = cpmx2[l][j];
					cpmxpdn[j][count] = l;
					count++;
				}
			}
			cpmxpdn[j][count] = -1;
		}
		mc_build( cpmx2, lgth2 );
	}

	{
		for( l=0; l<nalphabets; l++ )
		{
			scarr[l] = 0.0;
			for( j=0; j<nalphabets; j++ )
				scarr[l] += n_dis_consweight_multi[j][l] * cpmx1[j][i1];
//				scarr[l] += n_dis[j][l] * cpmx1[j][i1];
		}
		matchpt = match;
		cpmxpdnptpt = cpmxpdn;
		cpmxpdptpt = cpmxpd;
		while( lgth2-- )
		{
			*matchpt = 0.0;
			cpmxpdnpt = *cpmxpdnptpt++;
			cpmxpdpt = *cpmxpdptpt++;
			while( *cpmxpdnpt>-1 )
				*matchpt += scarr[*cpmxpdnpt++] * *cpmxpdpt++;
			matchpt++;
		} 
	}
	free( scarr );
#else
	int j, k, l;
//	double scarr[26];
	double **cpmxpd = doublework;
	int **cpmxpdn = intwork;
	double *scarr;
	scarr = calloc( nalphabets, sizeof( double ) );
	// simple
	if( initialize )
	{
		int count = 0;
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
	for( l=0; l<nalphabets; l++ )
	{
		scarr[l] = 0.0;
		for( k=0; k<nalphabets; k++ )
			scarr[l] += n_dis_consweight_multi[k][l] * cpmx1[k][i1];
//			scarr[l] += n_dis[k][l] * cpmx1[k][i1];
	}
	for( j=0; j<lgth2; j++ )
	{
		match[j] = 0.0;
		for( k=0; cpmxpdn[k][j]>-1; k++ )
			match[j] += scarr[cpmxpdn[k][j]] * cpmxpd[k][j];
	} 
#endif
}


static void fillzero( double *s, int l )
{
	while( l-- ) *s++ = 0.0;
}


static void match_calc_del( int **which, double ***matrices, double *match, int n1, char **seq1, double *eff1, int n2, char **seq2, double *eff2, int i1, int lgth2, int mid, int nmask, int *mask1, int *mask2 ) 
{
// osoi!
	int i, j, k, m;
	int c1, c2;
//	fprintf( stderr, "\nmatch_calc_dynamicmtx... %d", i1 );
//	fprintf( stderr, "\nseq1[0]=%s\n", seq1[0] );
//	fprintf( stderr, "\nseq2[0]=%s\n", seq2[0] );
//	for( i=0; i<n1; i++ ) for( j=0; j<n2; j++ )
//	{
//		if( flip ) reporterr( "in match_calc_slow, which[%d][%d] = %d\n", j, i, which[j][i] );
//		else       reporterr( "in match_calc_slow, which[%d][%d] = %d\n", i, j, which[i][j] );
//	}
	for( k=0; k<lgth2; k++ )
	{
		for( m=0; m<nmask; m++ )
		{
			i = mask1[m];
			j = mask2[m];
//			reporterr( "Deleting %d-%d (c=%d)\n", i, j, mid );
//			if( k==0 ) fprintf( stderr, "pairoffset[%d][%d] = %f\n", i, j, po );
			c1 = amino_n[(int)seq1[i][i1]];
			c2 = amino_n[(int)seq2[j][k]];
//			reporterr( "k=%d, c1=%d, c2=%d, seq1[i][i1]=%c, seq2[%d][%d]=%c\n", k, c1, c2, seq1[i][i1], j, k, seq2[j][k] );
			if( seq1[i][i1] == '-' || seq2[j][k] == '-' ) continue;
			if( c1 < 0 || c2 < 0 ) continue;
//			fprintf( stderr, "c1=%d, c2=%d\n", c1, c2 );
//			fprintf( stderr, "match[k] = %f -> ", match[k], mid );
			match[k] -= matrices[mid][c1][c2] * eff1[i] * eff2[j];
//			fprintf( stderr, "match[k] = %f (mid=%d)\n", match[k], mid );
		}
	}
//	fprintf( stderr, "done\n" );
	return;
}

static void match_calc_add( double **scoreingmtx, double *match, double **cpmx1, double **cpmx2, int i1, int lgth2, double **doublework, int **intwork, int initialize )
{
#if FASTMATCHCALC
//	fprintf( stderr, "\nmatch_calc... %d", i1 );
	int j, l;
//	double scarr[26];
	double **cpmxpd = doublework;
	int **cpmxpdn = intwork;
	double *matchpt, *cpmxpdpt, **cpmxpdptpt;
	int *cpmxpdnpt, **cpmxpdnptpt;
	double *scarr;
	scarr = calloc( nalphabets, sizeof( double ) );
	if( initialize )
	{
		int count = 0;
		for( j=0; j<lgth2; j++ )
		{
			count = 0;
			for( l=0; l<nalphabets; l++ )
			{
				if( cpmx2[l][j] )
				{
					cpmxpd[j][count] = cpmx2[l][j];
					cpmxpdn[j][count] = l;
					count++;
				}
			}
			cpmxpdn[j][count] = -1;
		}
	}

	{
#ifdef HAVE_SCARR_FILL
		scarr_fill( scarr, scoreingmtx, cpmx1, i1 );
#else
		for( l=0; l<nalphabets; l++ )
		{
			scarr[l] = 0.0;
			for( j=0; j<nalphabets; j++ )
//				scarr[l] += n_dis[j][l] * cpmx1[j][i1];
//				scarr[l] += n_dis_consweight_multi[j][l] * cpmx1[j][i1];
				scarr[l] += scoreingmtx[j][l] * cpmx1[j][i1];
		}
#endif
		matchpt = match;
		cpmxpdnptpt = cpmxpdn;
		cpmxpdptpt = cpmxpd;
		while( lgth2-- )
		{
//			*matchpt = 0.0;
			cpmxpdnpt = *cpmxpdnptpt++;
			cpmxpdpt = *cpmxpdptpt++;
			while( *cpmxpdnpt>-1 )
				*matchpt += scarr[*cpmxpdnpt++] * *cpmxpdpt++;
			matchpt++;
		} 
	}
	free( scarr );
//	fprintf( stderr, "done\n" );
#else
	int j, k, l;
//	double scarr[26];
	double **cpmxpd = doublework;
	int **cpmxpdn = intwork;
	double *scarr;
	scarr = calloc( nalphabets, sizeof( double ) );
// simple
	if( initialize )
	{
		int count = 0;
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
	scarr_fill( scarr, scoreingmtx, cpmx1, i1 );
#else
	for( l=0; l<nalphabets; l++ )
	{
		scarr[l] = 0.0;
		for( k=0; k<nalphabets; k++ )
//			scarr[l] += n_dis[k][l] * cpmx1[k][i1];
//			scarr[l] += n_dis_consweight_multi[k][l] * cpmx1[k][i1];
			scarr[l] += scoreingmtx[k][l] * cpmx1[k][i1];
	}
#endif
	for( j=0; j<lgth2; j++ )
	{
		match[j] = 0.0;
		for( k=0; cpmxpdn[k][j]>-1; k++ )
			match[j] += scarr[cpmxpdn[k][j]] * cpmxpd[k][j];
	} 
	free( scarr );
#endif
}

static void Atracking_localhom( double *impwmpt, double *lasthorizontalw, double *lastverticalw, 
						char **seq1, char **seq2, 
                        char **mseq1, char **mseq2, 
                        int **ijp, int icyc, int jcyc,
						int start1, int end1, int start2, int end2,
						int *gapmap1, int *gapmap2,
						int *warpis, int *warpjs, int warpbase )
{
	int i, j, l, iin, jin, ifi, jfi, lgth1, lgth2, k, limk;
	int *col1, *col2, cp;
//	char gap[] = "-";
	char *gap;
	double wm;
	gap = newgapstr;
	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );

#if 0
	for( i=0; i<lgth1; i++ ) 
	{
		fprintf( stderr, "lastverticalw[%d] = %f\n", i, lastverticalw[i] );
	}
#endif
 
	if( outgap == 1 )
		;
	else
	{
		wm = lastverticalw[0];
		for( i=0; i<lgth1; i++ )
		{
			if( lastverticalw[i] >= wm )
			{
				wm = lastverticalw[i];
				iin = i; jin = lgth2-1;
				ijp[lgth1][lgth2] = +( lgth1 - i );
			}
		}
		for( j=0; j<lgth2; j++ )
		{
			if( lasthorizontalw[j] >= wm )
			{
				wm = lasthorizontalw[j];
				iin = lgth1-1; jin = j;
				ijp[lgth1][lgth2] = -( lgth2 - j );
			}
		}
	}

    for( i=0; i<lgth1+1; i++ ) 
    {
        ijp[i][0] = i + 1;
    }
    for( j=0; j<lgth2+1; j++ ) 
    {
        ijp[0][j] = -( j + 1 );
    }

	/* Trace the path into column lists first (col1/col2: residue index or -1 for a gap, filled
	   from the end), then write every sequence row by row.  Same output as writing each column
	   across all sequences, without touching icyc+jcyc cache lines per column. */
	col1 = malloc( sizeof( int ) * ( lgth1+lgth2+2 ) );
	col2 = malloc( sizeof( int ) * ( lgth1+lgth2+2 ) );
	cp = lgth1+lgth2;
	iin = lgth1; jin = lgth2;
	*impwmpt = 0.0;
	limk = lgth1+lgth2+1;
	for( k=0; k<limk; k++ ) 
	{

		if( ijp[iin][jin] >= warpbase )
		{
			ifi = warpis[ijp[iin][jin]-warpbase];
			jfi = warpjs[ijp[iin][jin]-warpbase];
		}
		else if( ijp[iin][jin] < 0 ) 
		{
			ifi = iin-1; jfi = jin+ijp[iin][jin];
		}
		else if( ijp[iin][jin] > 0 )
		{
			ifi = iin-ijp[iin][jin]; jfi = jin-1;
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
				--cp; col1[cp] = l; col2[cp] = -1;
				k++;
			}
			l= jin;
			while( --l >= 0 )
			{
				--cp; col1[cp] = -1; col2[cp] = l;
				k++;
			}
			break;
		}
		else
		{
			l = iin - ifi;
			while( --l ) 
			{
				--cp; col1[cp] = ifi+l; col2[cp] = -1;
				k++;
			}
			l= jin - jfi;
			while( --l )
			{
				--cp; col1[cp] = -1; col2[cp] = jfi+l;
				k++;
			}
		}
		if( iin != lgth1 && jin != lgth2 ) // ??
		{
			*impwmpt += (double)part_imp_match_out_sc( gapmap1[iin]+start1, gapmap2[jin]+start2 );
//			fprintf( stderr, "impwm = %f (iin=%d, jin=%d) seq1=%c, seq2=%c\n", *impwmpt, iin, jin, seq1[0][iin], seq2[0][jin] );
		}
		if( iin <= 0 || jin <= 0 ) break;
		--cp; col1[cp] = ifi; col2[cp] = jfi;
		k++;
		iin = ifi; jin = jfi;
	}
	{
		int c, ncol = lgth1+lgth2-cp;
		for( i=0; i<icyc; i++ )
		{
			char *d, *s = seq1[i];
			mseq1[i] += cp;
			d = mseq1[i];
			for( c=0; c<ncol; c++ ) d[c] = ( col1[cp+c] >= 0 ) ? s[col1[cp+c]] : *gap;
			d[ncol] = 0;
		}
		for( j=0; j<jcyc; j++ )
		{
			char *d, *s = seq2[j];
			mseq2[j] += cp;
			d = mseq2[j];
			for( c=0; c<ncol; c++ ) d[c] = ( col2[cp+c] >= 0 ) ? s[col2[cp+c]] : *gap;
			d[ncol] = 0;
		}
	}
	free( col1 ); free( col2 );
}

static double Atracking( double *lasthorizontalw, double *lastverticalw, 
						char **seq1, char **seq2, 
                        char **mseq1, char **mseq2, 
                        int **ijp, int icyc, int jcyc,
						int *warpis, int *warpjs, int warpbase )
{
	int i, j, l, iin, jin, ifi, jfi, lgth1, lgth2, k, lastk, limk;
//	char gap[] = "-";
	char *gap;
	gap = newgapstr;
	double wm = 0.0;
	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );

#if 0
	for( i=0; i<lgth1; i++ ) 
	{
		fprintf( stderr, "lastverticalw[%d] = %f\n", i, lastverticalw[i] );
	}
#endif
 
	if( outgap == 1 )
		;
	else
	{
		wm = lastverticalw[0];
		for( i=0; i<lgth1; i++ )
		{
			if( lastverticalw[i] >= wm )
			{
				wm = lastverticalw[i];
				iin = i; jin = lgth2-1;
				ijp[lgth1][lgth2] = +( lgth1 - i );
			}
		}
		for( j=0; j<lgth2; j++ )
		{
			if( lasthorizontalw[j] >= wm )
			{
				wm = lasthorizontalw[j];
				iin = lgth1-1; jin = j;
				ijp[lgth1][lgth2] = -( lgth2 - j );
			}
		}
	}

    for( i=0; i<lgth1+1; i++ ) 
    {
        ijp[i][0] = i + 1;
    }
    for( j=0; j<lgth2+1; j++ ) 
    {
        ijp[0][j] = -( j + 1 );
    }

	for( i=0; i<icyc; i++ )
	{
		mseq1[i] += lgth1+lgth2;
		*mseq1[i] = 0;
	}
	for( j=0; j<jcyc; j++ )
	{
		mseq2[j] += lgth1+lgth2;
		*mseq2[j] = 0;
	}
	iin = lgth1; jin = lgth2;
	lastk = lgth1+lgth2;
	limk = lgth1+lgth2+1;
	for( k=0; k<limk; k++ ) 
	{
		if( ijp[iin][jin] >= warpbase )
		{
			ifi = warpis[ijp[iin][jin]-warpbase];
			jfi = warpjs[ijp[iin][jin]-warpbase];
		}
		else if( ijp[iin][jin] < 0 ) 
		{
			ifi = iin-1; jfi = jin+ijp[iin][jin];
		}
		else if( ijp[iin][jin] > 0 )
		{
			ifi = iin-ijp[iin][jin]; jfi = jin-1;
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
				for( i=0; i<icyc; i++ )
					*--mseq1[i] = seq1[i][l];
				for( j=0; j<jcyc; j++ ) 
					*--mseq2[j] = *gap;
				k++;
			}
			l= jin;
			while( --l >= 0 )
			{
				for( i=0; i<icyc; i++ ) 
					*--mseq1[i] = *gap;
				for( j=0; j<jcyc; j++ ) 
					*--mseq2[j] = seq2[j][l];
				k++;
			}
			break;
		}
		else
		{
			l = iin - ifi;
			while( --l ) 
			{
				for( i=0; i<icyc; i++ )
					*--mseq1[i] = seq1[i][ifi+l];
				for( j=0; j<jcyc; j++ ) 
					*--mseq2[j] = *gap;
				k++;
			}
			l= jin - jfi;
			while( --l )
			{
				for( i=0; i<icyc; i++ ) 
					*--mseq1[i] = *gap;
				for( j=0; j<jcyc; j++ ) 
					*--mseq2[j] = seq2[j][jfi+l];
				k++;
			}
		}

		if( iin <= 0 || jin <= 0 ) break;
		for( i=0; i<icyc; i++ ) 
			*--mseq1[i] = seq1[i][ifi];
		for( j=0; j<jcyc; j++ ) 
			*--mseq2[j] = seq2[j][jfi];
		k++;
		iin = ifi; jin = jfi;
	}
	return( 0.0 );
}


#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
/*
 * One row of the partA__align fill (trywarp == 0, USE_PENALTY_EX == 0), equivalent to the scalar
 * j-loop.  Every candidate for cell (i,j) comes from row i-1, and without the extension penalty
 * the horizontal state mi is just a running maximum (ties keep the earlier position), so it is
 * computed first as a scan and the cells are then independent.  Each multiply-add is the same
 * single-rounding fma the compiler emits for the original expressions, so values, ijp and the
 * m/mp state are bit-identical.
 */
static void partA_row( int i, int lgth2, double *prev, double *cur, double *m, int *mp, int *ijrow,
                       double *fgcp2, double *ogcp2, double *gapfreq2,
                       double fgcp1va, double ogcp1va, double gf1va, double gf1vapre, double mi0,
                       double *MI, int *MPI )
{
	int j, bi = 0;
	double best = mi0, g;

	MI[1] = mi0; MPI[1] = 0;
	j = 1;
#if defined(__AVX512F__) && defined(__AVX512VL__)
	/* The running best as a prefix scan over (value, position) pairs, 8 at a time: a later pair
	   replaces an earlier one only if its value is strictly greater (ties keep the earlier, as '>'), which is
	   associative, so three shift+compare+blend steps (and the carry from the previous block)
	   give each lane exactly the scalar loop's best and bi.  Blends, not max, so ties between
	   -0 and +0 resolve as in the scalar loop. */
	{
		__m512d cv = _mm512_set1_pd( best ), vpre = _mm512_set1_pd( gf1vapre );
		__m256i ck = _mm256_set1_epi32( bi ), vk = _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 ), veight = _mm256_set1_epi32( 8 );
		for( ; j+8<=lgth2; j+=8 )
		{
			__m512d x = VMULADD( _mm512_loadu_pd( ogcp2 + j ), vpre, _mm512_loadu_pd( prev + j - 1 ) ), e;
			__m256i t = vk, f;
			__mmask8 c;
			e = _mm512_castsi512_pd( _mm512_alignr_epi64( _mm512_castpd_si512( x ), _mm512_castpd_si512( cv ), 7 ) ); f = _mm256_alignr_epi32( t, ck, 7 );
			c = _mm512_cmp_pd_mask( x, e, _CMP_GT_OQ ); x = _mm512_mask_blend_pd( c, e, x ); t = _mm256_mask_blend_epi32( c, f, t );
			e = _mm512_castsi512_pd( _mm512_alignr_epi64( _mm512_castpd_si512( x ), _mm512_castpd_si512( cv ), 6 ) ); f = _mm256_alignr_epi32( t, ck, 6 );
			c = _mm512_cmp_pd_mask( x, e, _CMP_GT_OQ ); x = _mm512_mask_blend_pd( c, e, x ); t = _mm256_mask_blend_epi32( c, f, t );
			e = _mm512_castsi512_pd( _mm512_alignr_epi64( _mm512_castpd_si512( x ), _mm512_castpd_si512( cv ), 4 ) ); f = _mm256_alignr_epi32( t, ck, 4 );
			c = _mm512_cmp_pd_mask( x, e, _CMP_GT_OQ ); x = _mm512_mask_blend_pd( c, e, x ); t = _mm256_mask_blend_epi32( c, f, t );
			c = _mm512_cmp_pd_mask( x, cv, _CMP_GT_OQ ); x = _mm512_mask_blend_pd( c, cv, x ); t = _mm256_mask_blend_epi32( c, ck, t );
			_mm512_storeu_pd( MI + j + 1, x );
			_mm256_storeu_si256( (__m256i *)( MPI + j + 1 ), t );
			cv = _mm512_permutexvar_pd( _mm512_set1_epi64( 7 ), x );
			ck = _mm256_permutexvar_epi32( _mm256_set1_epi32( 7 ), t );
			vk = _mm256_add_epi32( vk, veight );
		}
		best = _mm512_cvtsd_f64( cv ); bi = _mm256_cvtsi256_si32( ck );
	}
#elif defined(MAFFT_AVX2_PATHS)
	/* The AVX-512 scan above, 4 at a time. */
	{
		__m256d cv = _mm256_set1_pd( best ), vpre = _mm256_set1_pd( gf1vapre );
		__m128i ck = _mm_set1_epi32( bi ), vk = _mm_setr_epi32( 0, 1, 2, 3 ), vfour = _mm_set1_epi32( 4 );
		for( ; j+4<=lgth2; j+=4 )
		{
			__m256d x = VMULADD4( _mm256_loadu_pd( ogcp2 + j ), vpre, _mm256_loadu_pd( prev + j - 1 ) ), e, c;
			__m128i t = vk, f, c32;
			e = _mm256_blend_pd( _mm256_permute4x64_pd( x, 0x90 ), cv, 0x1 ); f = _mm_alignr_epi8( t, ck, 12 );
			c = _mm256_cmp_pd( x, e, _CMP_GT_OQ ); c32 = PACKMASK4( c ); x = _mm256_blendv_pd( e, x, c ); t = _mm_blendv_epi8( f, t, c32 );
			e = _mm256_permute2f128_pd( cv, x, 0x21 ); f = _mm_alignr_epi8( t, ck, 8 );
			c = _mm256_cmp_pd( x, e, _CMP_GT_OQ ); c32 = PACKMASK4( c ); x = _mm256_blendv_pd( e, x, c ); t = _mm_blendv_epi8( f, t, c32 );
			c = _mm256_cmp_pd( x, cv, _CMP_GT_OQ ); c32 = PACKMASK4( c ); x = _mm256_blendv_pd( cv, x, c ); t = _mm_blendv_epi8( ck, t, c32 );
			_mm256_storeu_pd( MI + j + 1, x );
			_mm_storeu_si128( (__m128i *)( MPI + j + 1 ), t );
			cv = _mm256_permute4x64_pd( x, 0xff );
			ck = _mm_shuffle_epi32( t, 0xff );
			vk = _mm_add_epi32( vk, vfour );
		}
		best = _mm256_cvtsd_f64( cv ); bi = _mm_cvtsi128_si32( ck );
	}
#endif
	for( ; j<lgth2; j++ )
	{
		g = MULADD( ogcp2[j], gf1vapre, prev[j-1] );
		if( g > best ) { best = g; bi = j-1; }
		MI[j+1] = best; MPI[j+1] = bi;
	}

	j = 1;
#if defined(__ARM_NEON) && MAFFT_STOCK_FMA /* vfmaq: fused, like the stock arm64 build */
	{
		float64x2_t vgf1va = vdupq_n_f64( gf1va ), vfgcp1va = vdupq_n_f64( fgcp1va ), vogcp1va = vdupq_n_f64( ogcp1va );
		int32x2_t vi = vdup_n_s32( i ), vi1 = vdup_n_s32( i-1 ), vzero = vdup_n_s32( 0 ), vtwo = vdup_n_s32( 2 );
		int32x2_t vj = { 1, 2 };
		for( ; j+1<=lgth2; j+=2 )
		{
			float64x2_t p = vld1q_f64( prev + j - 1 );
			float64x2_t wm, g1, g3, g4, mv;
			uint64x2_t c;
			int32x2_t ij, mpv;

			g1 = vfmaq_f64( vld1q_f64( MI + j ), vld1q_f64( fgcp2 + j - 1 ), vgf1va );
			c = vcgtq_f64( g1, p );
			wm = vbslq_f64( c, g1, p );
			ij = vbsl_s32( vmovn_u64( c ), vsub_s32( vld1_s32( MPI + j ), vj ), vzero );

			mv = vld1q_f64( m + j );
			mpv = vld1_s32( mp + j );
			g3 = vfmaq_f64( mv, vfgcp1va, vld1q_f64( gapfreq2 + j ) );
			c = vcgtq_f64( g3, wm );
			wm = vbslq_f64( c, g3, wm );
			ij = vbsl_s32( vmovn_u64( c ), vsub_s32( vi, mpv ), ij );

			g4 = vfmaq_f64( p, vogcp1va, vld1q_f64( gapfreq2 + j - 1 ) );
			c = vcgtq_f64( g4, mv );
			vst1q_f64( m + j, vbslq_f64( c, g4, mv ) );
			vst1_s32( mp + j, vbsl_s32( vmovn_u64( c ), vi1, mpv ) );

			vst1q_f64( cur + j, vaddq_f64( vld1q_f64( cur + j ), wm ) );
			vst1_s32( ijrow + j, ij );
			vj = vadd_s32( vj, vtwo );
		}
	}
#elif defined(__AVX512F__) && defined(__AVX512VL__)
	/* The same, 8 cells at a time; VMULADD rounds like MULADD. */
	{
		__m512d vgf1va = _mm512_set1_pd( gf1va ), vfgcp1va = _mm512_set1_pd( fgcp1va ), vogcp1va = _mm512_set1_pd( ogcp1va );
		__m256i vi = _mm256_set1_epi32( i ), vi1 = _mm256_set1_epi32( i-1 ), veight = _mm256_set1_epi32( 8 );
		__m256i vj = _mm256_setr_epi32( 1, 2, 3, 4, 5, 6, 7, 8 );
		for( ; j+7<=lgth2; j+=8 )
		{
			__m512d p = _mm512_loadu_pd( prev + j - 1 );
			__m512d wm, g1, g3, g4, mv;
			__mmask8 c;
			__m256i ij, mpv;

			g1 = VMULADD( _mm512_loadu_pd( fgcp2 + j - 1 ), vgf1va, _mm512_loadu_pd( MI + j ) );
			c = _mm512_cmp_pd_mask( g1, p, _CMP_GT_OQ );
			wm = _mm512_mask_blend_pd( c, p, g1 );
			ij = _mm256_maskz_sub_epi32( c, _mm256_loadu_si256( (__m256i *)( MPI + j ) ), vj );

			mv = _mm512_loadu_pd( m + j );
			mpv = _mm256_loadu_si256( (__m256i *)( mp + j ) );
			g3 = VMULADD( vfgcp1va, _mm512_loadu_pd( gapfreq2 + j ), mv );
			c = _mm512_cmp_pd_mask( g3, wm, _CMP_GT_OQ );
			wm = _mm512_mask_blend_pd( c, wm, g3 );
			ij = _mm256_mask_blend_epi32( c, ij, _mm256_sub_epi32( vi, mpv ) );

			g4 = VMULADD( vogcp1va, _mm512_loadu_pd( gapfreq2 + j - 1 ), p );
			c = _mm512_cmp_pd_mask( g4, mv, _CMP_GT_OQ );
			_mm512_storeu_pd( m + j, _mm512_mask_blend_pd( c, mv, g4 ) );
			_mm256_storeu_si256( (__m256i *)( mp + j ), _mm256_mask_blend_epi32( c, mpv, vi1 ) );

			_mm512_storeu_pd( cur + j, _mm512_add_pd( _mm512_loadu_pd( cur + j ), wm ) );
			_mm256_storeu_si256( (__m256i *)( ijrow + j ), ij );
			vj = _mm256_add_epi32( vj, veight );
		}
	}
#elif defined(MAFFT_AVX2_PATHS)
	/* The AVX-512 block above, 4 cells at a time. */
	{
		__m256d vgf1va = _mm256_set1_pd( gf1va ), vfgcp1va = _mm256_set1_pd( fgcp1va ), vogcp1va = _mm256_set1_pd( ogcp1va );
		__m128i vi = _mm_set1_epi32( i ), vi1 = _mm_set1_epi32( i-1 ), vfour = _mm_set1_epi32( 4 );
		__m128i vj = _mm_setr_epi32( 1, 2, 3, 4 );
		for( ; j+3<=lgth2; j+=4 )
		{
			__m256d p = _mm256_loadu_pd( prev + j - 1 );
			__m256d wm, g1, g3, g4, mv, c;
			__m128i ij, mpv;

			g1 = VMULADD4( _mm256_loadu_pd( fgcp2 + j - 1 ), vgf1va, _mm256_loadu_pd( MI + j ) );
			c = _mm256_cmp_pd( g1, p, _CMP_GT_OQ );
			wm = _mm256_blendv_pd( p, g1, c );
			ij = _mm_and_si128( PACKMASK4( c ), _mm_sub_epi32( _mm_loadu_si128( (__m128i *)( MPI + j ) ), vj ) );

			mv = _mm256_loadu_pd( m + j );
			mpv = _mm_loadu_si128( (__m128i *)( mp + j ) );
			g3 = VMULADD4( vfgcp1va, _mm256_loadu_pd( gapfreq2 + j ), mv );
			c = _mm256_cmp_pd( g3, wm, _CMP_GT_OQ );
			wm = _mm256_blendv_pd( wm, g3, c );
			ij = _mm_blendv_epi8( ij, _mm_sub_epi32( vi, mpv ), PACKMASK4( c ) );

			g4 = VMULADD4( vogcp1va, _mm256_loadu_pd( gapfreq2 + j - 1 ), p );
			c = _mm256_cmp_pd( g4, mv, _CMP_GT_OQ );
			_mm256_storeu_pd( m + j, _mm256_blendv_pd( mv, g4, c ) );
			_mm_storeu_si128( (__m128i *)( mp + j ), _mm_blendv_epi8( mpv, vi1, PACKMASK4( c ) ) );

			_mm256_storeu_pd( cur + j, _mm256_add_pd( _mm256_loadu_pd( cur + j ), wm ) );
			_mm_storeu_si128( (__m128i *)( ijrow + j ), ij );
			vj = _mm_add_epi32( vj, vfour );
		}
	}
#endif
	for( ; j<=lgth2; j++ )
	{
		double p = prev[j-1], wm = p;
		int ij = 0;
		g = MULADD( fgcp2[j-1], gf1va, MI[j] );
		if( g > wm ) { wm = g; ij = -( j - MPI[j] ); }
		g = MULADD( fgcp1va, gapfreq2[j], m[j] );
		if( g > wm ) { wm = g; ij = +( i - mp[j] ); }
		g = MULADD( ogcp1va, gapfreq2[j-1], p );
		if( g > m[j] ) { m[j] = g; mp[j] = i-1; }
		cur[j] += wm;
		ijrow[j] = ij;
	}
}

double partA__align( char **seq1, char **seq2, double *eff1, double *eff2, int icyc, int jcyc, int alloclen, int constraint, double *impmatch, int start1, int end1, int start2, int end2, int *gapmap1, int *gapmap2, char *sgap1, char *sgap2, char *egap1, char *egap2, int *chudanpt, int chudanref, int *chudanres )
/* score no keisan no sai motokaraaru gap no atukai ni mondai ga aru */
{
//	int k;
	register int i, j;
	int lasti, lastj; /* outgap == 0 -> lgth1, outgap == 1 -> lgth1+1 */
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
	static TLS double *ogcp1;
	static TLS double *ogcp2;
	static TLS double *fgcp1;
	static TLS double *fgcp2;
	static TLS double **cpmx1;
	static TLS double **cpmx2;
	static TLS double *gapfreq1;
	static TLS double *gapfreq2;
	static TLS int **intwork;
	static TLS double **doublework;
	static TLS int orlgth1 = 0, orlgth2 = 0;
	double fpenalty = (double)penalty;
	double fpenalty_shift = (double)penalty_shift;
#if USE_PENALTY_EX
	double fpenalty_ex = (double)penalty_ex;
#endif
	double *fgcp2pt;
	double *ogcp2pt;
	double fgcp1va;
	double ogcp1va;
	double *gf2pt;
	double *gf2ptpre;
	double gf1va;
	double gf1vapre;
	double headgapfreq1;
	double headgapfreq2;

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
	double curm = 0.0;
	double *wmrecordspt, *wmrecords1pt, *prevwmrecordspt;
	int *warpipt, *warpjpt;
	static TLS double *rowMI = NULL;
	static TLS int *rowMPI = NULL;
	static TLS int rowalloc = 0;


	if( seq1 == NULL )
	{
		if( orlgth1 )
		{
//			fprintf( stderr, "## Freeing local arrays in A__align\n" );
			orlgth1 = 0;
			orlgth2 = 0;

			part_imp_match_init_strict( NULL, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL );

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

			FreeFloatVec( ogcp1 );
			FreeFloatVec( ogcp2 );
			FreeFloatVec( fgcp1 );
			FreeFloatVec( fgcp2 );


			FreeFloatMtx( cpmx1 );
			FreeFloatMtx( cpmx2 );

			FreeFloatVec( gapfreq1 );
			FreeFloatVec( gapfreq2 );

			FreeFloatMtx( doublework );
			FreeIntMtx( intwork );

		}
		else
		{
//			fprintf( stderr, "## Not allocated\n" );
		}
		return( 0.0 );
	}
//	fprintf( stderr, "IN partA__align\n" );


	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );
#if 1
//	if( lgth1 == 0 ) fprintf( stderr, "WARNING: lgth1=0 in partA__align\n" );
//	if( lgth2 == 0 ) fprintf( stderr, "WARNING: lgth2=0 in partA__align\n" );

	if( lgth1 == 0 && lgth2 == 0 )
		return( 0.0 );

	if( lgth1 == 0 )
	{
		for( i=0; i<icyc; i++ )
		{
			j = lgth2;
			seq1[i][j] = 0;
			while( j ) seq1[i][--j] = *newgapstr;
//			fprintf( stderr, "seq1[i] = %s\n", seq1[i] );
		}
		return( 0.0 );
	}

	if( lgth2 == 0 )
	{
		for( i=0; i<jcyc; i++ )
		{
			j = lgth1;
			seq2[i][j] = 0;
			while( j ) seq2[i][--j] = *newgapstr;
//			fprintf( stderr, "seq2[i] = %s\n", seq2[i] );
		}
		return( 0.0 );
	}
#endif

	warpbase = lgth1 + lgth2;
	warpis = NULL;
	warpjs = NULL;
	warpn = 0;




	if( trywarp )
	{
		if( outgap == 0 )
		{
			fprintf( stderr, "At present, outgap must be 1.\n" );
			exit( 1 );
		}
		wmrecords = AllocateFloatVec( lgth2+1 );
		warpi = AllocateIntVec( lgth2+1 );
		warpj = AllocateIntVec( lgth2+1 );
		prevwmrecords = AllocateFloatVec( lgth2+1 );
		prevwarpi = AllocateIntVec( lgth2+1 );
		prevwarpj = AllocateIntVec( lgth2+1 );
		for( i=0; i<lgth2+1; i++ ) wmrecords[i] = 0.0;
		for( i=0; i<lgth2+1; i++ ) prevwmrecords[i] = 0.0;
		for( i=0; i<lgth2+1; i++ ) prevwarpi[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) prevwarpj[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) warpi[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) warpj[i] = -warpbase;
	}

#if 0
	fprintf( stderr, "eff in SA+++align\n" );
	for( i=0; i<icyc; i++ ) fprintf( stderr, "eff1[%d] = %f\n", i, eff1[i] );
#endif
	if( orlgth1 == 0 )
	{
		mseq1 = AllocateCharMtx( njob, 0 );
		mseq2 = AllocateCharMtx( njob, 0 );
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

			FreeFloatVec( ogcp1 );
			FreeFloatVec( ogcp2 );
			FreeFloatVec( fgcp1 );
			FreeFloatVec( fgcp2 );


			FreeFloatMtx( cpmx1 );
			FreeFloatMtx( cpmx2 );

			FreeFloatVec( gapfreq1 );
			FreeFloatVec( gapfreq2 );

			FreeFloatMtx( doublework );
			FreeIntMtx( intwork );
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

		ogcp1 = AllocateFloatVec( ll1+2 );
		ogcp2 = AllocateFloatVec( ll2+2 );
		fgcp1 = AllocateFloatVec( ll1+2 );
		fgcp2 = AllocateFloatVec( ll2+2 );

		cpmx1 = AllocateFloatMtx( nalphabets, ll1+2 );
		cpmx2 = AllocateFloatMtx( nalphabets, ll2+2 );

		gapfreq1 = AllocateFloatVec( ll1+2 );
		gapfreq2 = AllocateFloatVec( ll2+2 );

#if FASTMATCHCALC
		doublework = AllocateFloatMtx( MAX( ll1, ll2 )+2, nalphabets ); 
		intwork = AllocateIntMtx( MAX( ll1, ll2 )+2, nalphabets ); 
#else
		doublework = AllocateFloatMtx( nalphabets, MAX( ll1, ll2 )+2 ); 
		intwork = AllocateIntMtx( nalphabets, MAX( ll1, ll2 )+2 ); 
#endif

#if DEBUG
		fprintf( stderr, "succeeded\n" );
#endif

		orlgth1 = ll1 - 100;
		orlgth2 = ll2 - 100;
	}


	for( i=0; i<icyc; i++ ) mseq1[i] = mseq[i];
	for( j=0; j<jcyc; j++ ) mseq2[j] = mseq[icyc+j];


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

	cpmx_calc_new( seq1, cpmx1, eff1, lgth1, icyc );
	cpmx_calc_new( seq2, cpmx2, eff2, lgth2, jcyc );

	if( sgap1 )
	{
		new_OpeningGapCount( ogcp1, icyc, seq1, eff1, lgth1, sgap1 );
		new_OpeningGapCount( ogcp2, jcyc, seq2, eff2, lgth2, sgap2 );
		new_FinalGapCount( fgcp1, icyc, seq1, eff1, lgth1, egap1 );
		new_FinalGapCount( fgcp2, jcyc, seq2, eff2, lgth2, egap2 );
		outgapcount( &headgapfreq1, icyc, sgap1, eff1 );
		outgapcount( &headgapfreq2, jcyc, sgap2, eff2 );
		outgapcount( gapfreq1+lgth1, icyc, egap1, eff1 );
		outgapcount( gapfreq2+lgth2, jcyc, egap2, eff2 );
	}
	else
	{
		st_OpeningGapCount( ogcp1, icyc, seq1, eff1, lgth1 );
		st_OpeningGapCount( ogcp2, jcyc, seq2, eff2, lgth2 );
		st_FinalGapCount( fgcp1, icyc, seq1, eff1, lgth1 );
		st_FinalGapCount( fgcp2, jcyc, seq2, eff2, lgth2 );
		headgapfreq1 = 0.0;
		headgapfreq2 = 0.0;
		gapfreq1[lgth1] = 0.0;
		gapfreq2[lgth2] = 0.0;
	}

	if( legacygapcost == 0 )
	{
		gapcountf( gapfreq1, seq1, icyc, eff1, lgth1 );
		gapcountf( gapfreq2, seq2, jcyc, eff2, lgth2 );
		for( i=0; i<lgth1+1; i++ ) gapfreq1[i] = 1.0 - gapfreq1[i];
		for( i=0; i<lgth2+1; i++ ) gapfreq2[i] = 1.0 - gapfreq2[i];
		headgapfreq1 = 1.0 - headgapfreq1;
		headgapfreq2 = 1.0 - headgapfreq2;
	}
	else
	{
		for( i=0; i<lgth1+1; i++ ) gapfreq1[i] = 1.0;
		for( i=0; i<lgth2+1; i++ ) gapfreq2[i] = 1.0;
		headgapfreq1 = 1.0;
		headgapfreq2 = 1.0;
	}

	for( i=0; i<lgth1; i++ ) 
	{
		ogcp1[i] = 0.5 * ( 1.0 - ogcp1[i] ) * fpenalty * ( gapfreq1[i] );
		fgcp1[i] = 0.5 * ( 1.0 - fgcp1[i] ) * fpenalty * ( gapfreq1[i] );
	}
	for( i=0; i<lgth2; i++ ) 
	{
		ogcp2[i] = 0.5 * ( 1.0 - ogcp2[i] ) * fpenalty * ( gapfreq2[i] );
		fgcp2[i] = 0.5 * ( 1.0 - fgcp2[i] ) * fpenalty * ( gapfreq2[i] );
	}
#if 0
	for( i=0; i<lgth1; i++ ) 
		fprintf( stderr, "ogcp1[%d]=%f\n", i, ogcp1[i] );
#endif

	currentw = w1;
	previousw = w2;


	match_calc( initverticalw, cpmx2, cpmx1, 0, lgth1, doublework, intwork, 1 );
	if( constraint )
		part_imp_match_out_vead_tate_gapmap( initverticalw, gapmap2[0]+start2, lgth1, start1, gapmap1 );


	match_calc( currentw, cpmx1, cpmx2, 0, lgth2, doublework, intwork, 1 );
	if( constraint )
		part_imp_match_out_vead_gapmap( currentw, gapmap1[0]+start1, lgth2, start2, gapmap2 );
#if 0 // -> tbfast.c
	if( localhom )
		imp_match_calc( currentw, icyc, jcyc, lgth1, lgth2, seq1, seq2, eff1, eff2, localhom, 1, 0 );

#endif

	if( outgap == 1 )
	{
		for( i=1; i<lgth1+1; i++ )
		{
//			initverticalw[i] += ( ogcp1[0] + fgcp1[i-1] ) ;
			initverticalw[i] += ( ogcp1[0] * headgapfreq2 + fgcp1[i-1] * gapfreq2[0] ) ;
#if USE_PENALTY_EX
			tsuika hitsuyou
#endif
		}
		for( j=1; j<lgth2+1; j++ )
		{
//			currentw[j] += ( ogcp2[0] + fgcp2[j-1] ) ;
			currentw[j] += ( ogcp2[0] * headgapfreq1 + fgcp2[j-1] * gapfreq1[0] ) ;
#if USE_PENALTY_EX
			tsuika hitsuyou
#endif
		}
	}
#if OUTGAP0TRY
	else
	{
		for( j=1; j<lgth2+1; j++ )
			currentw[j] -= offset * j / 2.0;
		for( i=1; i<lgth1+1; i++ )
			initverticalw[i] -= offset * i / 2.0;
	}
#endif

	for( j=1; j<lgth2+1; ++j ) 
	{
//		m[j] = currentw[j-1] + ogcp1[1]; mp[j] = 0;
		m[j] = currentw[j-1] + ogcp1[1] * gapfreq2[j-1]; mp[j] = 0;;
	}

	lastverticalw[0] = currentw[lgth2-1];

	if( outgap ) lasti = lgth1+1; else lasti = lgth1;
	if( rowalloc < lgth2+2 )
	{
		free( rowMI ); free( rowMPI );
		rowalloc = lgth2+2;
		rowMI = malloc( sizeof( double ) * rowalloc );
		rowMPI = malloc( sizeof( int ) * rowalloc );
	}
	lastj = lgth2+1;

#if XXXXXXX
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
fprintf( stderr, "fcgp\n" );
for( i=0; i<lgth1; i++ ) 
	fprintf( stderr, "fgcp1[%d]=%f\n", i, ogcp1[i] );
for( i=0; i<lgth2; i++ ) 
	fprintf( stderr, "fgcp2[%d]=%f\n", i, ogcp2[i] );
#endif

	for( i=1; i<lasti; i++ )
	{

#ifdef enablemultithread
//		fprintf( stderr, "chudan = %d, %d\n", *chudanpt, chudanref );
		if( chudanpt && *chudanpt != chudanref ) 
		{
//			fprintf( stderr, "\n\n## CHUUDAN!!! i\n" );
			*chudanres = 1;
			return( -1.0 );
		}
#endif

		wtmp = previousw; 
		previousw = currentw;
		currentw = wtmp;

		previousw[0] = initverticalw[i-1];

		match_calc( currentw, cpmx1, cpmx2, i, lgth2, doublework, intwork, 0 );
#if 0
fprintf( stderr, "\n" );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw before imp = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
		if( constraint )
		{
//			fprintf( stderr, "Calling imp_match_calc (o) lgth = %d, i = %d\n", lgth1, i );
//			imp_match_out_vead( currentw, i, lgth2 );
			part_imp_match_out_vead_gapmap( currentw, gapmap1[i]+start1, lgth2, start2, gapmap2 );
		}
#if 0
fprintf( stderr, "specificity = 0\n" );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
		currentw[0] = initverticalw[i];


//		mi = previousw[0] + ogcp2[1]; mpi = 0;
		mi = previousw[0] + ogcp2[1] * gapfreq1[i-1]; mpi=0;

		ijppt = ijp[i] + 1;
		mjpt = m + 1;
		prept = previousw;
		curpt = currentw + 1;
		mpjpt = mp + 1;
		fgcp2pt = fgcp2;
		ogcp2pt = ogcp2+1;
		fgcp1va = fgcp1[i-1];
		ogcp1va = ogcp1[i];
		gf1va = gapfreq1[i];
		gf1vapre = gapfreq1[i-1];
		gf2pt = gapfreq2+1;
		gf2ptpre = gapfreq2;

		if( trywarp )
		{
			prevwmrecordspt = prevwmrecords;
			wmrecordspt = wmrecords+1;
			wmrecords1pt = wmrecords;
			warpipt = warpi + 1;
			warpjpt = warpj + 1;
		}

#if !USE_PENALTY_EX
		if( !trywarp )
		{
			partA_row( i, lgth2, previousw, currentw, m, mp, ijp[i], fgcp2, ogcp2, gapfreq2,
			           fgcp1va, ogcp1va, gf1va, gf1vapre, mi, rowMI, rowMPI );
			goto partA_rowdone;
		}
#endif
		for( j=1; j<lastj; j++ )
		{
#ifdef xxxenablemultithread
//			fprintf( stderr, "chudan = %d, %d\n", *chudanpt, chudanref );
			if( chudanpt && *chudanpt != chudanref ) 
			{
//				fprintf( stderr, "\n\n## CHUUDAN!!! j\n" );
				*chudanres = 1;
				return( -1.0 );
			}
#endif
			wm = *prept;
			*ijppt = 0;

#if 0
			fprintf( stderr, "%5.0f->", wm );
#endif
//			g = mi + *fgcp2pt * gapfreq1[i];
			if( (g = mi + *fgcp2pt * gf1va) > wm )
			{
				wm = g;
				*ijppt = -( j - mpi );
			}
//			g = *prept + *ogcp2pt * gapfreq1[i-1];
//			if( (g = *prept + *ogcp2pt * gf1vapre) >= mi )
			if( (g = *prept + *ogcp2pt * gf1vapre) > mi ) // 2018/Apr
			{
				mi = g;
				mpi = j-1;
			}
#if USE_PENALTY_EX
			mi += fpenalty_ex;
#endif

//			g = *mjpt + fgcp1va * gapfreq2[j];
			if( (g = *mjpt + fgcp1va * *gf2pt) > wm )
			{
				wm = g;
				*ijppt = +( i - *mpjpt );
			}
//			g = *prept + ogcp1va * gapfreq2[j-1];
//			if( (g = *prept + ogcp1va * *gf2ptpre) >= *mjpt ) 
			if( (g = *prept + ogcp1va * *gf2ptpre) > *mjpt ) // 2018/Apr
			{
				*mjpt = g;
				*mpjpt = i-1;
			}
#if USE_PENALTY_EX
			m[j] += fpenalty_ex;
#endif
			if( trywarp )
			{
#if USE_PENALTY_EX
				if( ( g=*prevwmrecordspt++ + fpenalty_shift + fpenalty_ex * ( i - prevwarpi[j-1] + j - prevwarpj[j-1] ) ) > wm ) // naka ha osokute kamawanai
#else
				if( ( g=*prevwmrecordspt++ + fpenalty_shift ) > wm ) // naka ha osokute kamawanai
#endif
				{
//					fprintf( stderr, "WARP in partA__align\n" );
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
				curm = *curpt + wm;
				if( *wmrecords1pt > *wmrecordspt )
				{
					*wmrecordspt = *wmrecords1pt;
					*warpipt  = *(warpipt-1);
					*warpjpt  = *(warpjpt-1);
				}
				if( curm > *wmrecordspt )
				{
					*wmrecordspt = curm;
					*warpipt = i;
					*warpjpt = j;
				}
				wmrecordspt++;
				wmrecords1pt++;
				warpipt++;
				warpjpt++;
			}

#if 0
			fprintf( stderr, "%5.0f ", wm );
#endif
			*curpt += wm;
			ijppt++;
			mjpt++;
			prept++;
			mpjpt++;
			curpt++;
			fgcp2pt++;
			ogcp2pt++;
			gf2ptpre++;
			gf2pt++;

		}
#if !USE_PENALTY_EX
partA_rowdone:
#endif
		lastverticalw[i] = currentw[lgth2-1];

		if( trywarp )
		{
			fltncpy( prevwmrecords, wmrecords, lastj );
			intncpy( prevwarpi, warpi, lastj );
			intncpy( prevwarpj, warpj, lastj );
		}
	}
	if( trywarp )
	{
//		fprintf( stderr, "wm = %f\n", wm );
//		fprintf( stderr, "warpn = %d\n", warpn );
		free( wmrecords );
		free( prevwmrecords );
		free( warpi );
		free( warpj );
		free( prevwarpi );
		free( prevwarpj );
	}

#if OUTGAP0TRY
	if( !outgap )
	{
		for( j=1; j<lgth2+1; j++ )
			currentw[j] -= offset * ( lgth2 - j ) / 2.0;
		for( i=1; i<lgth1+1; i++ )
			lastverticalw[i] -= offset * ( lgth1 - i  / 2.0);
	}
#endif
		
	/*
	fprintf( stderr, "\n" );
	for( i=0; i<icyc; i++ ) fprintf( stderr,"%s\n", seq1[i] );
	fprintf( stderr, "#####\n" );
	for( j=0; j<jcyc; j++ ) fprintf( stderr,"%s\n", seq2[j] );
	fprintf( stderr, "====>" );
	for( i=0; i<icyc; i++ ) strcpy( mseq1[i], seq1[i] );
	for( j=0; j<jcyc; j++ ) strcpy( mseq2[j], seq2[j] );
	*/
	if( constraint )
	{
		Atracking_localhom( impmatch, currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, icyc, jcyc, start1, end1, start2, end2, gapmap1, gapmap2, warpis, warpjs, warpbase );
	}
	else
		Atracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, icyc, jcyc, warpis, warpjs, warpbase );

	if( warpis ) free( warpis );
	if( warpjs ) free( warpjs );

//	fprintf( stderr, "### impmatch = %f\n", *impmatch );

	resultlen = strlen( mseq1[0] );
	if( alloclen < resultlen || resultlen > N )
	{
		fprintf( stderr, "alloclen=%d, resultlen=%d, N=%d\n", alloclen, resultlen, N );
		ErrorExit( "LENGTH OVER!\n" );
	}


	for( i=0; i<icyc; i++ ) strcpy( seq1[i], mseq1[i] );
	for( j=0; j<jcyc; j++ ) strcpy( seq2[j], mseq2[j] );
	/*
	fprintf( stderr, "\n" );
	for( i=0; i<icyc; i++ ) fprintf( stderr, "%s\n", mseq1[i] );
	fprintf( stderr, "#####\n" );
	for( j=0; j<jcyc; j++ ) fprintf( stderr, "%s\n", mseq2[j] );
	*/


	return( wm );
}

double partA__align_variousdist( int **which, double ***matrices, double **n_dynamicmtx, char **seq1, char **seq2, double *eff1, double *eff2, double **eff1s, double **eff2s, int icyc, int jcyc, int alloclen, int constraint, double *impmatch, int start1, int end1, int start2, int end2, int *gapmap1, int *gapmap2, char *sgap1, char *sgap2, char *egap1, char *egap2, int *chudanpt, int chudanref, int *chudanres )
/* score no keisan no sai motokaraaru gap no atukai ni mondai ga aru */
{
//	int k;
	register int i, j, c;
	int lasti, lastj; /* outgap == 0 -> lgth1, outgap == 1 -> lgth1+1 */
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
	static TLS double *ogcp1;
	static TLS double *ogcp2;
	static TLS double *fgcp1;
	static TLS double *fgcp2;
	static TLS double ***cpmx1s;
	static TLS double ***cpmx2s;
	static TLS double *gapfreq1;
	static TLS double *gapfreq2;
	static TLS int ***intwork;
	static TLS double ***doublework;
	static TLS int orlgth1 = 0, orlgth2 = 0;
	double fpenalty = (double)penalty;
	double fpenalty_shift = (double)penalty_shift;
#if USE_PENALTY_EX
	double fpenalty_ex = (double)penalty_ex;
#endif
	double *fgcp2pt;
	double *ogcp2pt;
	double fgcp1va;
	double ogcp1va;
	double *gf2pt;
	double *gf2ptpre;
	double gf1va;
	double gf1vapre;
	double headgapfreq1;
	double headgapfreq2;

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
	double curm = 0.0;
	double *wmrecordspt, *wmrecords1pt, *prevwmrecordspt;
	int *warpipt, *warpjpt;
	int *nmask, **masklist1, **masklist2;


	if( seq1 == NULL )
	{
		if( orlgth1 )
		{
//			fprintf( stderr, "## Freeing local arrays in A__align\n" );
			orlgth1 = 0;
			orlgth2 = 0;

			part_imp_match_init_strict( NULL, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL );

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

			FreeFloatVec( ogcp1 );
			FreeFloatVec( ogcp2 );
			FreeFloatVec( fgcp1 );
			FreeFloatVec( fgcp2 );


			FreeFloatCub( cpmx1s );
			FreeFloatCub( cpmx2s );

			FreeFloatVec( gapfreq1 );
			FreeFloatVec( gapfreq2 );

			FreeFloatCub( doublework );
			FreeIntCub( intwork );

		}
		else
		{
//			fprintf( stderr, "## Not allocated\n" );
		}
		return( 0.0 );
	}

	masklist1 = AllocateIntMtx( maxdistclass, 0 );
	masklist2 = AllocateIntMtx( maxdistclass, 0 );
	nmask = calloc( maxdistclass, sizeof( int ) );

	for( c=0; c<maxdistclass; c++ )
	{
		for( i=0; i<icyc; i++ ) for( j=0; j<jcyc; j++ )
		{
			if( eff1s[c][i] * eff2s[c][j] != 0.0 )
			{

				if( c != which[i][j] )
				{
					masklist1[c] = realloc( masklist1[c], sizeof( int ) * (nmask[c]+1) );
					masklist2[c] = realloc( masklist2[c], sizeof( int ) * (nmask[c]+1) );
					masklist1[c][nmask[c]] = i;
					masklist2[c][nmask[c]] = j;
					nmask[c]++;
				}
			}
		}
	}
	for( c=0; c<maxdistclass; c++ ) if( nmask[c] ) break;
	if( c<maxdistclass ) reporterr( "Found a complex grouping. This step may be a bit slow.\n" );

	lgth1 = strlen( seq1[0] );
	lgth2 = strlen( seq2[0] );
#if 1
//	if( lgth1 == 0 ) fprintf( stderr, "WARNING: lgth1=0 in partA__align\n" );
//	if( lgth2 == 0 ) fprintf( stderr, "WARNING: lgth2=0 in partA__align\n" );

	if( lgth1 == 0 && lgth2 == 0 )
		return( 0.0 );

	if( lgth1 == 0 )
	{
		for( i=0; i<icyc; i++ )
		{
			j = lgth2;
			seq1[i][j] = 0;
			while( j ) seq1[i][--j] = *newgapstr;
//			fprintf( stderr, "seq1[i] = %s\n", seq1[i] );
		}
		return( 0.0 );
	}

	if( lgth2 == 0 )
	{
		for( i=0; i<jcyc; i++ )
		{
			j = lgth1;
			seq2[i][j] = 0;
			while( j ) seq2[i][--j] = *newgapstr;
//			fprintf( stderr, "seq2[i] = %s\n", seq2[i] );
		}
		return( 0.0 );
	}
#endif

	warpbase = lgth1 + lgth2;
	warpis = NULL;
	warpjs = NULL;
	warpn = 0;


	if( trywarp )
	{
//		fprintf( stderr, "IN partA__align_variousdist\n" );
		if( outgap == 0 )
		{
			fprintf( stderr, "At present, outgap must be 1 to allow shift.\n" );
			exit( 1 );
		}
		wmrecords = AllocateFloatVec( lgth2+1 );
		warpi = AllocateIntVec( lgth2+1 );
		warpj = AllocateIntVec( lgth2+1 );
		prevwmrecords = AllocateFloatVec( lgth2+1 );
		prevwarpi = AllocateIntVec( lgth2+1 );
		prevwarpj = AllocateIntVec( lgth2+1 );
		for( i=0; i<lgth2+1; i++ ) wmrecords[i] = 0.0;
		for( i=0; i<lgth2+1; i++ ) prevwmrecords[i] = 0.0;
		for( i=0; i<lgth2+1; i++ ) prevwarpi[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) prevwarpj[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) warpi[i] = -warpbase;
		for( i=0; i<lgth2+1; i++ ) warpj[i] = -warpbase;
	}

#if 0
	fprintf( stderr, "eff in SA+++align\n" );
	for( i=0; i<icyc; i++ ) fprintf( stderr, "eff1[%d] = %f\n", i, eff1[i] );
#endif
	if( orlgth1 == 0 )
	{
		mseq1 = AllocateCharMtx( njob, 0 );
		mseq2 = AllocateCharMtx( njob, 0 );
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

			FreeFloatVec( ogcp1 );
			FreeFloatVec( ogcp2 );
			FreeFloatVec( fgcp1 );
			FreeFloatVec( fgcp2 );


			FreeFloatCub( cpmx1s );
			FreeFloatCub( cpmx2s );

			FreeFloatVec( gapfreq1 );
			FreeFloatVec( gapfreq2 );

			FreeFloatCub( doublework );
			FreeIntCub( intwork );
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

		ogcp1 = AllocateFloatVec( ll1+2 );
		ogcp2 = AllocateFloatVec( ll2+2 );
		fgcp1 = AllocateFloatVec( ll1+2 );
		fgcp2 = AllocateFloatVec( ll2+2 );

		cpmx1s = AllocateFloatCub( maxdistclass, nalphabets, ll1+2 );
		cpmx2s = AllocateFloatCub( maxdistclass, nalphabets, ll2+2 );

		gapfreq1 = AllocateFloatVec( ll1+2 );
		gapfreq2 = AllocateFloatVec( ll2+2 );

		doublework = AllocateFloatCub( maxdistclass, MAX( ll1, ll2 )+2, nalphabets ); 
		intwork = AllocateIntCub( maxdistclass, MAX( ll1, ll2 )+2, nalphabets ); 

#if DEBUG
		fprintf( stderr, "succeeded\n" );
#endif

		orlgth1 = ll1 - 100;
		orlgth2 = ll2 - 100;
	}


	for( i=0; i<icyc; i++ ) mseq1[i] = mseq[i];
	for( j=0; j<jcyc; j++ ) mseq2[j] = mseq[icyc+j];


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

//	cpmx_calc_new( seq1, cpmx1, eff1, lgth1, icyc );
//	cpmx_calc_new( seq2, cpmx2, eff2, lgth2, jcyc );
	for( c=0; c<maxdistclass; c++ )
	{
		cpmx_calc_new( seq1, cpmx1s[c], eff1s[c], lgth1, icyc );
		cpmx_calc_new( seq2, cpmx2s[c], eff2s[c], lgth2, jcyc );
	}

	if( sgap1 )
	{
		new_OpeningGapCount( ogcp1, icyc, seq1, eff1, lgth1, sgap1 );
		new_OpeningGapCount( ogcp2, jcyc, seq2, eff2, lgth2, sgap2 );
		new_FinalGapCount( fgcp1, icyc, seq1, eff1, lgth1, egap1 );
		new_FinalGapCount( fgcp2, jcyc, seq2, eff2, lgth2, egap2 );
		outgapcount( &headgapfreq1, icyc, sgap1, eff1 );
		outgapcount( &headgapfreq2, jcyc, sgap2, eff2 );
		outgapcount( gapfreq1+lgth1, icyc, egap1, eff1 );
		outgapcount( gapfreq2+lgth2, jcyc, egap2, eff2 );
	}
	else
	{
		st_OpeningGapCount( ogcp1, icyc, seq1, eff1, lgth1 );
		st_OpeningGapCount( ogcp2, jcyc, seq2, eff2, lgth2 );
		st_FinalGapCount( fgcp1, icyc, seq1, eff1, lgth1 );
		st_FinalGapCount( fgcp2, jcyc, seq2, eff2, lgth2 );
		headgapfreq1 = 0.0;
		headgapfreq2 = 0.0;
		gapfreq1[lgth1] = 0.0;
		gapfreq2[lgth2] = 0.0;
	}

	if( legacygapcost == 0 )
	{
		gapcountf( gapfreq1, seq1, icyc, eff1, lgth1 );
		gapcountf( gapfreq2, seq2, jcyc, eff2, lgth2 );
		for( i=0; i<lgth1+1; i++ ) gapfreq1[i] = 1.0 - gapfreq1[i];
		for( i=0; i<lgth2+1; i++ ) gapfreq2[i] = 1.0 - gapfreq2[i];
		headgapfreq1 = 1.0 - headgapfreq1;
		headgapfreq2 = 1.0 - headgapfreq2;
	}
	else
	{
		for( i=0; i<lgth1+1; i++ ) gapfreq1[i] = 1.0;
		for( i=0; i<lgth2+1; i++ ) gapfreq2[i] = 1.0;
		headgapfreq1 = 1.0;
		headgapfreq2 = 1.0;
	}

	for( i=0; i<lgth1; i++ ) 
	{
		ogcp1[i] = 0.5 * ( 1.0 - ogcp1[i] ) * fpenalty * ( gapfreq1[i] );
		fgcp1[i] = 0.5 * ( 1.0 - fgcp1[i] ) * fpenalty * ( gapfreq1[i] );
	}
	for( i=0; i<lgth2; i++ ) 
	{
		ogcp2[i] = 0.5 * ( 1.0 - ogcp2[i] ) * fpenalty * ( gapfreq2[i] );
		fgcp2[i] = 0.5 * ( 1.0 - fgcp2[i] ) * fpenalty * ( gapfreq2[i] );
	}
#if 0
	for( i=0; i<lgth1; i++ ) 
		fprintf( stderr, "ogcp1[%d]=%f\n", i, ogcp1[i] );
#endif

	currentw = w1;
	previousw = w2;


//	match_calc( initverticalw, cpmx2, cpmx1, 0, lgth1, doublework, intwork, 1 );
	fillzero( initverticalw, lgth1 );
	for( c=0; c<maxdistclass; c++ )
	{
		match_calc_add( matrices[c], initverticalw, cpmx2s[c], cpmx1s[c], 0, lgth1, doublework[c], intwork[c], 1 );
		if( nmask[c] ) match_calc_del( which, matrices, initverticalw, jcyc, seq2, eff2, icyc, seq1, eff1, 0, lgth1, c, nmask[c], masklist2[c], masklist1[c] );
	}

	if( constraint )
		part_imp_match_out_vead_tate_gapmap( initverticalw, gapmap2[0]+start2, lgth1, start1, gapmap1 );


//	match_calc( currentw, cpmx1, cpmx2, 0, lgth2, doublework, intwork, 1 );
	fillzero( currentw, lgth2 );
	for( c=0; c<maxdistclass; c++ )
	{
		match_calc_add( matrices[c], currentw, cpmx1s[c], cpmx2s[c], 0, lgth2, doublework[c], intwork[c], 1 );
		if( nmask[c] ) match_calc_del( which, matrices, currentw, icyc, seq1, eff1, jcyc, seq2, eff2, 0, lgth2, c, nmask[c], masklist1[c], masklist2[c] );
	}
	if( constraint )
		part_imp_match_out_vead_gapmap( currentw, gapmap1[0]+start1, lgth2, start2, gapmap2 );
#if 0 // -> tbfast.c, localhom ga hitsuyou
	if( localhom )
		imp_match_calc( currentw, icyc, jcyc, lgth1, lgth2, seq1, seq2, eff1, eff2, localhom, 1, 0 );

#endif

	if( outgap == 1 )
	{
		for( i=1; i<lgth1+1; i++ )
		{
//			initverticalw[i] += ( ogcp1[0] + fgcp1[i-1] ) ;
			initverticalw[i] += ( ogcp1[0] * headgapfreq2 + fgcp1[i-1] * gapfreq2[0] ) ;
#if USE_PENALTY_EX
			tsuika hitsuyou 
#endif
		}
		for( j=1; j<lgth2+1; j++ )
		{
//			currentw[j] += ( ogcp2[0] + fgcp2[j-1] ) ;
			currentw[j] += ( ogcp2[0] * headgapfreq1 + fgcp2[j-1] * gapfreq1[0] ) ;
#if USE_PENALTY_EX
			tsuika hitsuyou 
#endif
		}
	}
#if OUTGAP0TRY
	else
	{
		for( j=1; j<lgth2+1; j++ )
			currentw[j] -= offset * j / 2.0;
		for( i=1; i<lgth1+1; i++ )
			initverticalw[i] -= offset * i / 2.0;
	}
#endif

	for( j=1; j<lgth2+1; ++j ) 
	{
//		m[j] = currentw[j-1] + ogcp1[1]; mp[j] = 0;
		m[j] = currentw[j-1] + ogcp1[1] * gapfreq2[j-1]; mp[j] = 0;;
	}

	lastverticalw[0] = currentw[lgth2-1];

	if( outgap ) lasti = lgth1+1; else lasti = lgth1;
	lastj = lgth2+1;

#if XXXXXXX
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
fprintf( stderr, "fcgp\n" );
for( i=0; i<lgth1; i++ ) 
	fprintf( stderr, "fgcp1[%d]=%f\n", i, ogcp1[i] );
for( i=0; i<lgth2; i++ ) 
	fprintf( stderr, "fgcp2[%d]=%f\n", i, ogcp2[i] );
#endif

	for( i=1; i<lasti; i++ )
	{

#ifdef enablemultithread
//		fprintf( stderr, "chudan = %d, %d\n", *chudanpt, chudanref );
		if( chudanpt && *chudanpt != chudanref ) 
		{
//			fprintf( stderr, "\n\n## CHUUDAN!!! i\n" );
			*chudanres = 1;
			if( masklist1 ) freeintmtx( masklist1, maxdistclass ); masklist1 = NULL;
			if( masklist2 ) freeintmtx( masklist2, maxdistclass ); masklist2 = NULL;
			if( nmask ) free( nmask ); nmask = NULL;
			return( -1.0 );
		}
#endif

		wtmp = previousw; 
		previousw = currentw;
		currentw = wtmp;

		previousw[0] = initverticalw[i-1];

//		match_calc( currentw, cpmx1, cpmx2, i, lgth2, doublework, intwork, 0 );
		fillzero( currentw, lgth2 );
		for( c=0; c<maxdistclass; c++ )
		{
			match_calc_add( matrices[c], currentw, cpmx1s[c], cpmx2s[c], i, lgth2, doublework[c], intwork[c], 0 );
			if( nmask[c] ) match_calc_del( which, matrices, currentw, icyc, seq1, eff1, jcyc, seq2, eff2, i, lgth2, c, nmask[c], masklist1[c], masklist2[c] );
		}
#if 0
fprintf( stderr, "\n" );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw before imp = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
		if( constraint )
		{
//			fprintf( stderr, "Calling imp_match_calc (o) lgth = %d, i = %d\n", lgth1, i );
//			imp_match_out_vead( currentw, i, lgth2 );
			part_imp_match_out_vead_gapmap( currentw, gapmap1[i]+start1, lgth2, start2, gapmap2 );
		}
#if 0
fprintf( stderr, "specificity = %f\n", specificityconsideration );
fprintf( stderr, "i=%d\n", i );
fprintf( stderr, "currentw = \n" );
for( j=0; j<lgth2; j++ )
{
	fprintf( stderr, "%5.2f ", currentw[j] );
}
fprintf( stderr, "\n" );
#endif
		currentw[0] = initverticalw[i];


//		mi = previousw[0] + ogcp2[1]; mpi = 0;
		mi = previousw[0] + ogcp2[1] * gapfreq1[i-1]; mpi=0;

		ijppt = ijp[i] + 1;
		mjpt = m + 1;
		prept = previousw;
		curpt = currentw + 1;
		mpjpt = mp + 1;
		fgcp2pt = fgcp2;
		ogcp2pt = ogcp2+1;
		fgcp1va = fgcp1[i-1];
		ogcp1va = ogcp1[i];
		gf1va = gapfreq1[i];
		gf1vapre = gapfreq1[i-1];
		gf2pt = gapfreq2+1;
		gf2ptpre = gapfreq2;

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
#ifdef xxxenablemultithread
//			fprintf( stderr, "chudan = %d, %d\n", *chudanpt, chudanref );
			if( chudanpt && *chudanpt != chudanref ) 
			{
//				fprintf( stderr, "\n\n## CHUUDAN!!! j\n" );
				*chudanres = 1;
				if( masklist1 ) freeintmtx( masklist1, maxdistclass ); masklist1 = NULL;
				if( masklist2 ) freeintmtx( masklist2, maxdistclass ); masklist2 = NULL;
				if( nmask ) free( nmask ); nmask = NULL;
				return( -1.0 );
			}
#endif
			wm = *prept;
			*ijppt = 0;

#if 0
			fprintf( stderr, "%5.0f->", wm );
#endif
//			g = mi + *fgcp2pt * gapfreq1[i];
			if( (g = mi + *fgcp2pt * gf1va) > wm )
			{
				wm = g;
				*ijppt = -( j - mpi );
			}
//			g = *prept + *ogcp2pt * gapfreq1[i-1];
//			if( (g = *prept + *ogcp2pt * gf1vapre) >= mi )
			if( (g = *prept + *ogcp2pt * gf1vapre) > mi ) // 2018/Apr
			{
				mi = g;
				mpi = j-1;
			}
#if USE_PENALTY_EX
			mi += fpenalty_ex;
#endif

//			g = *mjpt + fgcp1va * gapfreq2[j];
			if( (g = *mjpt + fgcp1va * *gf2pt) > wm )
			{
				wm = g;
				*ijppt = +( i - *mpjpt );
			}
//			g = *prept + ogcp1va * gapfreq2[j-1];
//			if( (g = *prept + ogcp1va * *gf2ptpre) >= *mjpt )
			if( (g = *prept + ogcp1va * *gf2ptpre) > *mjpt ) // 2018/Apr
			{
				*mjpt = g;
				*mpjpt = i-1;
			}
#if USE_PENALTY_EX
			m[j] += fpenalty_ex;
#endif
			if( trywarp )
			{
#if USE_PENALTY_EX
				if( ( g=*prevwmrecordspt++ + fpenalty_shift + fpenalty_ex * ( i - prevwarpi[j-1] + j - prevwarpj[j-1] ) ) > wm ) // naka ha osokute kamawanai
#else
				if( ( g=*prevwmrecordspt++ + fpenalty_shift ) > wm ) // naka ha osokute kamawanai
#endif
				{
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

				curm = *curpt + wm;
				if( *wmrecords1pt > *wmrecordspt )
				{
					*wmrecordspt = *wmrecords1pt;
					*warpipt  = *(warpipt-1);
					*warpjpt  = *(warpjpt-1);
				}
				if( curm > *wmrecordspt )
				{
					*wmrecordspt = curm;
					*warpipt = i;
					*warpjpt = j;
				}
				wmrecordspt++;
				wmrecords1pt++;
				warpipt++;
				warpjpt++;
			}

#if 0
			fprintf( stderr, "%5.0f ", wm );
#endif
			*curpt += wm;
			ijppt++;
			mjpt++;
			prept++;
			mpjpt++;
			curpt++;
			fgcp2pt++;
			ogcp2pt++;
			gf2ptpre++;
			gf2pt++;

		}
		lastverticalw[i] = currentw[lgth2-1];

		if( trywarp )
		{
			fltncpy( prevwmrecords, wmrecords, lastj );
			intncpy( prevwarpi, warpi, lastj );
			intncpy( prevwarpj, warpj, lastj );
		}
	}
	if( trywarp )
	{
//		fprintf( stderr, "wm = %f\n", wm );
//		fprintf( stderr, "warpn = %d\n", warpn );
		free( wmrecords );
		free( prevwmrecords );
		free( warpi );
		free( warpj );
		free( prevwarpi );
		free( prevwarpj );
	}

#if OUTGAP0TRY
	if( !outgap )
	{
		for( j=1; j<lgth2+1; j++ )
			currentw[j] -= offset * ( lgth2 - j ) / 2.0;
		for( i=1; i<lgth1+1; i++ )
			lastverticalw[i] -= offset * ( lgth1 - i  / 2.0);
	}
#endif
		
	/*
	fprintf( stderr, "\n" );
	for( i=0; i<icyc; i++ ) fprintf( stderr,"%s\n", seq1[i] );
	fprintf( stderr, "#####\n" );
	for( j=0; j<jcyc; j++ ) fprintf( stderr,"%s\n", seq2[j] );
	fprintf( stderr, "====>" );
	for( i=0; i<icyc; i++ ) strcpy( mseq1[i], seq1[i] );
	for( j=0; j<jcyc; j++ ) strcpy( mseq2[j], seq2[j] );
	*/
	if( constraint )
	{
		Atracking_localhom( impmatch, currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, icyc, jcyc, start1, end1, start2, end2, gapmap1, gapmap2, warpis, warpjs, warpbase );
	}
	else
		Atracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, icyc, jcyc, warpis, warpjs, warpbase );

	if( warpis ) free( warpis );
	if( warpjs ) free( warpjs );

//	fprintf( stderr, "### impmatch = %f\n", *impmatch );

	resultlen = strlen( mseq1[0] );
	if( alloclen < resultlen || resultlen > N )
	{
		fprintf( stderr, "alloclen=%d, resultlen=%d, N=%d\n", alloclen, resultlen, N );
		ErrorExit( "LENGTH OVER!\n" );
	}


	for( i=0; i<icyc; i++ ) strcpy( seq1[i], mseq1[i] );
	for( j=0; j<jcyc; j++ ) strcpy( seq2[j], mseq2[j] );
	/*
	fprintf( stderr, "\n" );
	for( i=0; i<icyc; i++ ) fprintf( stderr, "%s\n", mseq1[i] );
	fprintf( stderr, "#####\n" );
	for( j=0; j<jcyc; j++ ) fprintf( stderr, "%s\n", mseq2[j] );
	*/

	if( masklist1 ) freeintmtx( masklist1, maxdistclass ); masklist1 = NULL;
	if( masklist2 ) freeintmtx( masklist2, maxdistclass ); masklist2 = NULL;
	if( nmask ) free( nmask ); nmask = NULL;

	return( wm );
}
