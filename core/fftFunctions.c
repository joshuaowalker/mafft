#include "mltaln.h"
#if defined(MAFFT_A64)
/* One bit (bit 4t) per byte t of x that differs from every byte of the masks OR-ed into eq. */
#define NEON_NIBBLES( eq ) ( ~vget_lane_u64( vreinterpret_u64_u8( vshrn_n_u16( vreinterpretq_u16_u8( eq ), 4 ) ), 0 ) & 0x1111111111111111ULL )
/*
 * alignableReagion's column profiles when the gap has a bin (DNA): built bin-major in a scratch
 * buffer, where one row's 16 consecutive gap columns are 16 consecutive doubles of the gap bin
 * (one vector add each pair), then transposed into cp[i*nu+bin].  Every bin receives the same
 * additions in the same order (rows in order, one add per row), so cp is bit-identical.
 */
static int areg_binmajor( char **seq1, char **seq2, int clus1, int clus2, double *eff1, double *eff2, int len, int nu, int *cidx, double *cp1, double *cp2 )
{
	static TLS double *t = NULL;
	static TLS size_t tcap = 0;
	int side, j, i, a, gb = cidx['-'];
	const uint8x16_t dash = vdupq_n_u8( '-' );
	if( tcap < (size_t)len * nu )
	{
		free( t );
		tcap = (size_t)len * nu;
		t = malloc( sizeof( double ) * tcap );
		if( !t ) { tcap = 0; return( 0 ); }
	}
	for( side=0; side<2; side++ )
	{
		int nrow = side ? clus2 : clus1;
		char **seq = side ? seq2 : seq1;
		double *eff = side ? eff2 : eff1, *cp = side ? cp2 : cp1, *tg = t + (size_t)gb * len;
		memset( t, 0, sizeof( double ) * (size_t)len * nu );
		for( j=0; j<nrow; j++ )
		{
			unsigned char *s = (unsigned char *)seq[j];
			double e = eff[j];
			float64x2_t ve = vdupq_n_f64( e );
			for( i=0; i+16<=len; i+=16 )
			{
				uint8x16_t g = vceqq_u8( vld1q_u8( s + i ), dash );
				if( vminvq_u8( g ) )
				{
					int q;
					for( q=0; q<16; q+=2 ) vst1q_f64( tg + i + q, vaddq_f64( vld1q_f64( tg + i + q ), ve ) );
				}
				else
				{
					int q;
					for( q=i; q<i+16; q++ ) { int ci = cidx[s[q]]; if( ci >= 0 ) t[(size_t)ci*len+q] += e; }
				}
			}
			for( ; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) t[(size_t)ci*len+i] += e; }
		}
		for( i=0; i<len; i++ ) for( a=0; a<nu; a++ ) cp[(size_t)i*nu+a] = t[(size_t)a*len+i];
	}
	return( 1 );
}
#endif

#define SEGMENTSIZE 150
#define TMPTMPTMP 0

#define DEBUG 0

void keika( char *str, int current, int all )
{
	if( current == 0 )
		fprintf( stderr, "%s :         ", str );

		fprintf( stderr, "\b\b\b\b\b\b\b\b" );
		fprintf( stderr, "%3d /%3d", current+1, all+1 );

	if( current+1 == all )
		fprintf( stderr, "\b\b\b\b\b\b\b\bdone.     \n" );
}

double maxItch( double *soukan, int size )
{
	int i;
	double value = 0.0;
	double cand;
	for( i=0; i<size; i++ ) 
		if( ( cand = soukan[i] ) > value ) value = cand;
	return( value );
}

void calcNaiseki( Fukusosuu *value, Fukusosuu *x, Fukusosuu *y )
{
	value->R =  x->R * y->R + x->I * y->I;
	value->I = -x->R * y->I + x->I * y->R;
}

Fukusosuu *AllocateFukusosuuVec( int l1 )
{
	Fukusosuu *value;
	value = (Fukusosuu *)calloc( l1, sizeof( Fukusosuu ) );
	if( !value )
	{
		fprintf( stderr, "Cannot allocate %d FukusosuuVec\n", l1 );
		return( NULL );
	}
	return( value );
}
	
Fukusosuu **AllocateFukusosuuMtx( int l1, int l2 )
{
	Fukusosuu **value;
	int j;
//	fprintf( stderr, "allocating %d x %d FukusosuuMtx\n", l1, l2 );
	value = (Fukusosuu **)calloc( l1+1, sizeof( Fukusosuu * ) );
	if( !value ) 
	{
		fprintf( stderr, "Cannot allocate %d x %d FukusosuuVecMtx\n", l1, l2 );
		exit( 1 );
	}
	for( j=0; j<l1; j++ ) 
	{
		value[j] = AllocateFukusosuuVec( l2 );
		if( !value[j] )
		{
			fprintf( stderr, "Cannot allocate %d x %d FukusosuuVecMtx\n", l1, l2 );
			exit( 1 );
		}
	}
	value[l1] = NULL;
	return( value );
}

Fukusosuu ***AllocateFukusosuuCub( int l1, int l2, int l3 )
{
	Fukusosuu ***value;
	int i;
	value = calloc( l1+1, sizeof( Fukusosuu ** ) );
	if( !value ) ErrorExit( "Cannot allocate Fukusosuu" );
	for( i=0; i<l1; i++ ) value[i] = AllocateFukusosuuMtx( l2, l3 );
	value[l1] = NULL;
	return( value );
}

void FreeFukusosuuVec( Fukusosuu *vec )
{
	free( (void *)vec );
}

void FreeFukusosuuMtx( Fukusosuu **mtx )
{
	int i;

	for( i=0; mtx[i]; i++ ) 
		free( (void *)mtx[i] );
	free( (void *)mtx );
}

int getKouho( int *kouho, int nkouho, double *soukan, int nlen2 )
{
	int i, j;
	int nlen4 = nlen2 / 2;
	double max;
	double tmp;
	int ikouho = 0; // by D.Mathog, iinoka?
	for( j=0; j<nkouho; j++ ) 
	{
		max = -9999.9;
		for( i=0; i<nlen2; i++ ) 
		{
			if( ( tmp = soukan[i] ) > max )
			{
				ikouho = i;
				max = tmp;
			}
		}
#if 0
		if( max < 0.15 )
		{
			break;
		}
#endif
#if 0
		fprintf( stderr, "Kouho No.%d, pos=%d, score=%f, lag=%d\n", j, ikouho, soukan[ikouho], ikouho-nlen4 );
#endif
		soukan[ikouho] = -9999.9;
		kouho[j] = ( ikouho - nlen4 );
	}
	return( j );
}

void zurasu2( int lag, int    clus1, int    clus2, 
                       char  **seq1, char  **seq2, 
		 			   char **aseq1, char **aseq2 )
{
	int i;
#if 0
	fprintf( stderr, "### lag = %d\n", lag );
#endif
	if( lag > 0 )
	{
		for( i=0; i<clus1; i++ ) aseq1[i] = seq1[i];
		for( i=0; i<clus2; i++ ) aseq2[i] = seq2[i]+lag;
	}
	else
	{
		for( i=0; i<clus1; i++ ) aseq1[i] = seq1[i]-lag;
		for( i=0; i<clus2; i++ ) aseq2[i] = seq2[i];
	}
}

void zurasu( int lag, int    clus1, int    clus2, 
                      char  **seq1, char  **seq2, 
					  char **aseq1, char **aseq2 )
{
	int i;
#if DEBUG
	fprintf( stderr, "lag = %d\n", lag );
#endif
	if( lag > 0 )
	{
		for( i=0; i<clus1; i++ ) strcpy( aseq1[i], seq1[i] );
		for( i=0; i<clus2; i++ ) strcpy( aseq2[i], seq2[i]+lag );
	}
	else
	{
		for( i=0; i<clus1; i++ ) strcpy( aseq1[i], seq1[i]-lag );
		for( i=0; i<clus2; i++ ) strcpy( aseq2[i], seq2[i] );
	}
}


int alignableReagion( int    clus1, int    clus2, 
					   char  **seq1, char  **seq2,
					   double *eff1, double *eff2,
					   Segment *seg )
{
	int i, j, k;
	int status, starttmp = 0; // by D.Mathog, a gess
	double score;
	int value = 0;
	int len, maxlen;
	int length = 0; // by D.Mathog, a gess
	static TLS double *stra = NULL;
	static TLS int alloclen = 0;
	double totaleff;
	double cumscore;
	static TLS double threshold;
	static TLS double *prf1 = NULL;
	static TLS double *prf2 = NULL;
	static TLS int *hat1 = NULL;
	static TLS int *hat2 = NULL;
	int pre1, pre2;
#if 0
	char **seq1pt;
	char **seq2pt;
	double *eff1pt;
	double *eff2pt;
#endif

#if 0
	fprintf( stderr, "### In alignableRegion, clus1=%d, clus2=%d \n", clus1, clus2 );
	fprintf( stderr, "seq1[0] = %s\n", seq1[0] );
	fprintf( stderr, "seq2[0] = %s\n", seq2[0] );
	fprintf( stderr, "eff1[0] = %f\n", eff1[0] );
	fprintf( stderr, "eff2[0] = %f\n", eff2[0] );
#endif

	if( clus1 == 0 )
	{
		if( stra ) FreeDoubleVec( stra ); stra = NULL;
		if( prf1 ) FreeDoubleVec( prf1 ); prf1 = NULL;
		if( prf2 ) FreeDoubleVec( prf2 ); prf2 = NULL;
		if( hat1 ) FreeIntVec( hat1 ); hat1 = NULL;
		if( hat2 ) FreeIntVec( hat2 ); hat2 = NULL;
		alloclen = 0;
		return( 0 );
	}

	if( prf1 == NULL )
	{
		prf1 = AllocateDoubleVec( nalphabets );
		prf2 = AllocateDoubleVec( nalphabets );
		hat1 = AllocateIntVec( nalphabets+1 );
		hat2 = AllocateIntVec( nalphabets+1 );
	}

	len = MIN( strlen( seq1[0] ), strlen( seq2[0] ) );
	maxlen = MAX( strlen( seq1[0] ), strlen( seq2[0] ) ) + fftWinSize;
	if( alloclen < maxlen )
	{
		if( alloclen )
		{
			FreeDoubleVec( stra );
		}
		else
		{
			threshold = (int)fftThreshold / 100.0 * 600.0 * fftWinSize;
		}
		stra = AllocateDoubleVec( maxlen );
		alloclen = maxlen;
	}


	totaleff = 0.0;
	for( i=0; i<clus1; i++ ) for( j=0; j<clus2; j++ ) totaleff += eff1[i] * eff2[j];
	/* Build the column profiles row-wise (contiguous in each sequence) instead of column-major,
	   keeping only the bins that occur: the site score below reads bins 0..25 only, and only the
	   nonzero ones, in descending order.  Each bin still receives eff[j] in ascending j and the
	   products are summed in the same order, so stra[] is bit-identical. */
	{
		static TLS double *cprf = NULL;
		static TLS size_t ccap = 0;
		unsigned char seenc[0x100];
		int used[26], ub[26], cidx[0x100], nu = 0, ok = 1, c;
		memset( seenc, 0, sizeof( seenc ) );
#if defined(MAFFT_AVX512)
		/* Mark a few characters whose bins are valid as present up front and visit only the other
		   bytes.  An extra valid character only adds a bin whose profile is all zero, which the
		   site score below skips, so stra[] and the ok test come out the same. */
		{
			static const unsigned char common[] = "-acgtACGT";
			__m512i cv[9];
			int nc = 0, q;
			for( q=0; common[q]; q++ )
				if( amino_n[common[q]] >= 0 && amino_n[common[q]] < nalphabets ) { seenc[common[q]] = 1; cv[nc++] = _mm512_set1_epi8( (char)common[q] ); }
			for( j=0; j<clus1+clus2; j++ )
			{
				unsigned char *s = (unsigned char *)( j < clus1 ? seq1[j] : seq2[j-clus1] );
				for( i=0; i+64<=len; i+=64 )
				{
					__m512i x = _mm512_loadu_si512( s + i );
					unsigned long long m = 0;
					for( q=0; q<nc; q++ ) m |= _mm512_cmpeq_epi8_mask( x, cv[q] );
					m = ~m;
					while( m ) { seenc[s[i+__builtin_ctzll( m )]] = 1; m &= m - 1; }
				}
				for( ; i<len; i++ ) seenc[s[i]] = 1;
			}
		}
#elif defined(MAFFT_A64)
		/* The AVX-512BW block above, 16 bytes at a time (Linux arm64). */
		{
			static const unsigned char common[] = "-acgtACGT";
			uint8x16_t cv[9];
			int nc = 0, q;
			for( q=0; common[q]; q++ )
				if( amino_n[common[q]] >= 0 && amino_n[common[q]] < nalphabets ) { seenc[common[q]] = 1; cv[nc++] = vdupq_n_u8( common[q] ); }
			for( j=0; j<clus1+clus2; j++ )
			{
				unsigned char *s = (unsigned char *)( j < clus1 ? seq1[j] : seq2[j-clus1] );
				for( i=0; i+16<=len; i+=16 )
				{
					uint8x16_t x = vld1q_u8( s + i ), eq = vdupq_n_u8( 0 );
					unsigned long long m;
					for( q=0; q<nc; q++ ) eq = vorrq_u8( eq, vceqq_u8( x, cv[q] ) );
					m = NEON_NIBBLES( eq );
					while( m ) { seenc[s[i+( __builtin_ctzll( m ) >> 2 )]] = 1; m &= m - 1; }
				}
				for( ; i<len; i++ ) seenc[s[i]] = 1;
			}
		}
#elif defined(MAFFT_AVX2_NEWPATHS)
		/* The AVX-512BW pass above, 32 bytes at a time. */
		{
			static const unsigned char common[] = "-acgtACGT";
			__m256i cv[9];
			int nc = 0, q;
			for( q=0; common[q]; q++ )
				if( amino_n[common[q]] >= 0 && amino_n[common[q]] < nalphabets ) { seenc[common[q]] = 1; cv[nc++] = _mm256_set1_epi8( (char)common[q] ); }
			for( j=0; j<clus1+clus2; j++ )
			{
				unsigned char *s = (unsigned char *)( j < clus1 ? seq1[j] : seq2[j-clus1] );
				for( i=0; i+32<=len; i+=32 )
				{
					__m256i x = _mm256_loadu_si256( (__m256i *)( s + i ) ), e = _mm256_setzero_si256();
					unsigned int m;
					for( q=0; q<nc; q++ ) e = _mm256_or_si256( e, _mm256_cmpeq_epi8( x, cv[q] ) );
					m = ~(unsigned int)_mm256_movemask_epi8( e );
					while( m ) { seenc[s[i+__builtin_ctz( m )]] = 1; m &= m - 1; }
				}
				for( ; i<len; i++ ) seenc[s[i]] = 1;
			}
		}
#else
		for( j=0; j<clus1; j++ ) { unsigned char *s = (unsigned char *)seq1[j]; for( i=0; i<len; i++ ) seenc[s[i]] = 1; }
		for( j=0; j<clus2; j++ ) { unsigned char *s = (unsigned char *)seq2[j]; for( i=0; i<len; i++ ) seenc[s[i]] = 1; }
#endif
		for( k=0; k<26; k++ ) used[k] = 0;
		for( c=0; c<0x100; c++ ) if( seenc[c] )
		{
			int bin = amino_n[c];
			if( bin < 0 || bin >= nalphabets ) { ok = 0; break; }
			if( bin < 26 ) used[bin] = 1;
		}
		if( ok )
		{
			double *cp1, *cp2;
			int bin[26];
			for( k=25; k>=0; k-- ) if( used[k] ) { bin[k] = nu; ub[nu++] = k; } else bin[k] = -1;
			for( c=0; c<0x100; c++ ) cidx[c] = ( seenc[c] && amino_n[c] < 26 ) ? bin[(int)amino_n[c]] : -1;
			if( ccap < (size_t)len * nu * 2 + 1 )
			{
				free( cprf );
				ccap = (size_t)len * nu * 2 + 1;
				cprf = malloc( sizeof( double ) * ccap );
			}
			cp1 = cprf; cp2 = cprf + (size_t)len * nu;
			memset( cprf, 0, sizeof( double ) * (size_t)len * nu * 2 );
#if defined(MAFFT_AVX512X)
			/* Bin-major profiles (cp[ci*len+i]): for each character of a 64-column block, the columns
			   holding it get e added in its bin, 8 at a time under a mask.  Each bin of each column
			   still receives its additions in row order, and the site score reads the same values. */
			for( j=0; j<clus1+clus2; j++ )
			{
				unsigned char *s = (unsigned char *)( j < clus1 ? seq1[j] : seq2[j-clus1] );
				double *cp = ( j < clus1 ) ? cp1 : cp2;
				__m512d ve = _mm512_set1_pd( ( j < clus1 ) ? eff1[j] : eff2[j-clus1] );
				for( i=0; i<len; i+=64 )
				{
					unsigned long long lm = ( len - i >= 64 ) ? ~0ULL : ( ( 1ULL << ( len - i ) ) - 1 ), rest = lm;
					__m512i x = _mm512_maskz_loadu_epi8( lm, (void *)( s + i ) );
					while( rest )
					{
						unsigned char c = s[i+__builtin_ctzll( rest )];
						unsigned long long m = _mm512_cmpeq_epi8_mask( x, _mm512_set1_epi8( (char)c ) ) & rest;
						int ci = cidx[c], q;
						rest &= ~m;
						if( ci < 0 ) continue;
						for( q=0; q<8; q++ )
						{
							__mmask8 m8 = (__mmask8)( m >> ( 8 * q ) );
							double *pt = cp + (size_t)ci * len + i + 8 * q;
							_mm512_mask_storeu_pd( pt, m8, _mm512_add_pd( _mm512_maskz_loadu_pd( m8, pt ), ve ) );
						}
					}
				}
			}
			for( i=0; i<len; i++ )
			{
				int a, b;
				stra[i] = 0.0;
				for( a=0; a<nu; a++ )
				{
					double p1 = cp1[(size_t)a*len+i];
					if( !p1 ) continue;
					for( b=0; b<nu; b++ )
					{
						double p2 = cp2[(size_t)b*len+i];
						if( !p2 ) continue;
						stra[i] += n_disFFT[ub[a]][ub[b]] * p1 * p2;
					}
				}
				stra[i] /= totaleff;
			}
			goto profiles_done;
#else
#if defined(MAFFT_AVX512)
			/* the same adds in the same order, visiting only the non-gap bytes (when '-' has no bin) */
			for( j=0; j<clus1+clus2 && cidx['-'] < 0; j++ )
			{
				unsigned char *s = (unsigned char *)( j < clus1 ? seq1[j] : seq2[j-clus1] );
				double e = ( j < clus1 ) ? eff1[j] : eff2[j-clus1], *cp = ( j < clus1 ) ? cp1 : cp2;
				__m512i dash = _mm512_set1_epi8( '-' );
				for( i=0; i+64<=len; i+=64 )
				{
					unsigned long long m = ~_mm512_cmpeq_epi8_mask( _mm512_loadu_si512( s + i ), dash );
					while( m )
					{
						int ii = i + __builtin_ctzll( m ), ci = cidx[s[ii]];
						if( ci >= 0 ) cp[ii*nu+ci] += e;
						m &= m - 1;
					}
				}
				for( ; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) cp[i*nu+ci] += e; }
			}
			if( cidx['-'] >= 0 )
#elif defined(MAFFT_A64)
			/* the same adds in the same order, visiting only the non-gap bytes (when '-' has no bin) */
			for( j=0; j<clus1+clus2 && cidx['-'] < 0; j++ )
			{
				unsigned char *s = (unsigned char *)( j < clus1 ? seq1[j] : seq2[j-clus1] );
				double e = ( j < clus1 ) ? eff1[j] : eff2[j-clus1], *cp = ( j < clus1 ) ? cp1 : cp2;
				uint8x16_t dash = vdupq_n_u8( '-' );
				for( i=0; i+16<=len; i+=16 )
				{
					unsigned long long m = NEON_NIBBLES( vceqq_u8( vld1q_u8( s + i ), dash ) );
					while( m )
					{
						int ii = i + ( __builtin_ctzll( m ) >> 2 ), ci = cidx[s[ii]];
						if( ci >= 0 ) cp[ii*nu+ci] += e;
						m &= m - 1;
					}
				}
				for( ; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) cp[i*nu+ci] += e; }
			}
			if( cidx['-'] >= 0 && !areg_binmajor( seq1, seq2, clus1, clus2, eff1, eff2, len, nu, cidx, cp1, cp2 ) )
#elif defined(MAFFT_AVX2_NEWPATHS)
			/* The AVX-512BW pass above, 32 bytes at a time: the same adds in the same order. */
			for( j=0; j<clus1+clus2 && cidx['-'] < 0; j++ )
			{
				unsigned char *s = (unsigned char *)( j < clus1 ? seq1[j] : seq2[j-clus1] );
				double e = ( j < clus1 ) ? eff1[j] : eff2[j-clus1], *cp = ( j < clus1 ) ? cp1 : cp2;
				__m256i dash = _mm256_set1_epi8( '-' );
				for( i=0; i+32<=len; i+=32 )
				{
					unsigned int m = ~(unsigned int)_mm256_movemask_epi8( _mm256_cmpeq_epi8( _mm256_loadu_si256( (__m256i *)( s + i ) ), dash ) );
					while( m )
					{
						int ii = i + __builtin_ctz( m ), ci = cidx[s[ii]];
						if( ci >= 0 ) cp[ii*nu+ci] += e;
						m &= m - 1;
					}
				}
				for( ; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) cp[i*nu+ci] += e; }
			}
			if( cidx['-'] >= 0 )
#endif
			{
			for( j=0; j<clus1; j++ ) { unsigned char *s = (unsigned char *)seq1[j]; double e = eff1[j]; for( i=0; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) cp1[i*nu+ci] += e; } }
			for( j=0; j<clus2; j++ ) { unsigned char *s = (unsigned char *)seq2[j]; double e = eff2[j]; for( i=0; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) cp2[i*nu+ci] += e; } }
			}
			for( i=0; i<len; i++ )
			{
				double *q1 = cp1 + (size_t)i * nu, *q2 = cp2 + (size_t)i * nu;
				int a, b;
				stra[i] = 0.0;
				for( a=0; a<nu; a++ )
				{
					double p1 = q1[a];
					if( !p1 ) continue;
					for( b=0; b<nu; b++ )
					{
						double p2 = q2[b];
						if( !p2 ) continue;
						stra[i] += n_disFFT[ub[a]][ub[b]] * p1 * p2;
					}
				}
				stra[i] /= totaleff;
			}
			goto profiles_done;
#endif
		}
	}
	for( i=0; i<len; i++ )
	{
		/* make prfs */
		for( j=0; j<nalphabets; j++ )
		{
			prf1[j] = 0.0;
			prf2[j] = 0.0;
		}
#if 0
		seq1pt = seq1;
		eff1pt = eff1;
		j = clus1;
		while( j-- ) prf1[amino_n[(*seq1pt++)[i]]] += *eff1pt++;
#else
		for( j=0; j<clus1; j++ ) prf1[amino_n[(unsigned char)seq1[j][i]]] += eff1[j];
#endif
		for( j=0; j<clus2; j++ ) prf2[amino_n[(unsigned char)seq2[j][i]]] += eff2[j];

		/* make hats */
		pre1 = pre2 = nalphabets;
		for( j=25; j>=0; j-- )
		{
			if( prf1[j] )
			{
				hat1[pre1] = j;
				pre1 = j;
			}
			if( prf2[j] )
			{
				hat2[pre2] = j;
				pre2 = j;
			}
		}
		hat1[pre1] = -1;
		hat2[pre2] = -1;

		/* make site score */
		stra[i] = 0.0;
		for( k=hat1[nalphabets]; k!=-1; k=hat1[k] ) 
			for( j=hat2[nalphabets]; j!=-1; j=hat2[j] ) 
//				stra[i] += n_dis[k][j] * prf1[k] * prf2[j];
				stra[i] += n_disFFT[k][j] * prf1[k] * prf2[j];
		stra[i] /= totaleff;
	}

profiles_done:
	(seg+0)->skipForeward = 0;
	(seg+1)->skipBackward = 0;
	status = 0;
	cumscore = 0.0;
	score = 0.0;
	for( j=0; j<fftWinSize; j++ ) score += stra[j];

	for( i=1; i<len-fftWinSize; i++ )
	{
		score = score - stra[i-1] + stra[i+fftWinSize-1];
#if TMPTMPTMP
		fprintf( stderr, "%d %10.0f   ? %10.0f\n", i, score, threshold );
#endif

		if( score > threshold )
		{
#if 0
			seg->start = i;
			seg->end = i;
			seg->center = ( seg->start + seg->end + fftWinSize ) / 2 ;
			seg->score = score;
			status = 0;
			value++;
#else
			if( !status )
			{
				status = 1;
				starttmp = i;
				length = 0;
				cumscore = 0.0;
			}
			length++;
			cumscore += score;
#endif
		}
		if( score <= threshold || length > SEGMENTSIZE )
		{
			if( status )
			{
				if( length > fftWinSize )
				{
					seg->start = starttmp;
					seg->end = i;
					seg->center = ( seg->start + seg->end + fftWinSize ) / 2 ;
					seg->score = cumscore;
#if 0
					fprintf( stderr, "%d-%d length = %d, score = %f, value = %d\n", seg->start, seg->end, length, cumscore, value );
#endif
					if( length > SEGMENTSIZE )
					{
						(seg+0)->skipForeward = 1;
						(seg+1)->skipBackward = 1;
					}
					else
					{
						(seg+0)->skipForeward = 0;
						(seg+1)->skipBackward = 0;
					}
					value++;
					seg++;
				}
				length = 0;
				cumscore = 0.0;
				status = 0;
				starttmp = i;
				if( value > MAXSEG - 3 ) ErrorExit( "TOO MANY SEGMENTS!");
			}
		}
	}
	if( status && length > fftWinSize )
	{
		seg->end = i;
		seg->start = starttmp;
		seg->center = ( starttmp + i + fftWinSize ) / 2 ;
		seg->score = cumscore;
#if 0
fprintf( stderr, "%d-%d length = %d\n", seg->start, seg->end, length );
#endif
		value++;
	}
#if TMPTMPTMP
	exit( 0 );
#endif
//	fprintf( stderr, "returning %d\n", value );
	return( value );
}


static int permit( Segment *seg1, Segment *seg2 )
{
	return( 0 );
	if( seg1->end >= seg2->start ) return( 0 );
	if( seg1->pair->end >= seg2->pair->start ) return( 0 );
	else return( 1 );
}

void blockAlign2( int *cut1, int *cut2, Segment **seg1, Segment **seg2, double **ocrossscore, int *ncut )
{
	int i, j, k, shift, cur1, cur2, count, klim;
	static TLS int crossscoresize = 0;
	static TLS int *result1 = NULL;
	static TLS int *result2 = NULL;
	static TLS int *ocut1 = NULL;
	static TLS int *ocut2 = NULL;
	double maximum;
	static TLS double **crossscore = NULL;
	static TLS int **track = NULL;
	static TLS double maxj, maxi;
	static TLS int pointj, pointi;

	if( cut1 == NULL) 
	{
		if( result1 )
		{
			if( result1 ) free( result1 ); result1 = NULL;
			if( result2 ) free( result2 ); result2 = NULL;
			if( ocut1 ) free( ocut1 ); ocut1 = NULL;
			if( ocut2 ) free( ocut2 ); ocut2 = NULL;
			if( track ) FreeIntMtx( track ); track = NULL;
	        	if( crossscore ) FreeDoubleMtx( crossscore ); crossscore = NULL;
		}
		crossscoresize = 0;
		return;
	}

	if( result1 == NULL )
	{
		result1 = AllocateIntVec( MAXSEG );
		result2 = AllocateIntVec( MAXSEG );
		ocut1 = AllocateIntVec( MAXSEG );
		ocut2 = AllocateIntVec( MAXSEG );
	}

    if( crossscoresize < *ncut+2 )
    {
        crossscoresize = *ncut+2;
		if( fftkeika ) fprintf( stderr, "allocating crossscore and track, size = %d\n", crossscoresize );
		if( track ) FreeIntMtx( track );
        if( crossscore ) FreeDoubleMtx( crossscore );
		track = AllocateIntMtx( crossscoresize, crossscoresize );
        crossscore = AllocateDoubleMtx( crossscoresize, crossscoresize );
    }

#if 0
	for( i=0; i<*ncut-2; i++ )
		fprintf( stderr, "%d.start = %d, score = %f\n", i, seg1[i]->start, seg1[i]->score );

	for( i=0; i<*ncut; i++ )
		fprintf( stderr, "i=%d, cut1 = %d, cut2 = %d\n", i, cut1[i], cut2[i] );
	for( i=0; i<*ncut; i++ ) 
	{
		for( j=0; j<*ncut; j++ )
			fprintf( stderr, "%#4.0f ", ocrossscore[i][j] );
		fprintf( stderr, "\n" );
	}
#endif

	for( i=0; i<*ncut; i++ ) for( j=0; j<*ncut; j++ )  /* mudadanaa */
		crossscore[i][j] = ocrossscore[i][j];
	for( i=0; i<*ncut; i++ ) 
	{
		ocut1[i] = cut1[i];
		ocut2[i] = cut2[i];
	}

	for( i=1; i<*ncut; i++ )
	{
#if 0
		fprintf( stderr, "### i=%d/%d\n", i,*ncut );
#endif
		for( j=1; j<*ncut; j++ )
		{
			pointi = 0; maxi = 0.0;
			klim = j-2;
			for( k=0; k<klim; k++ )
			{
/*
				fprintf( stderr, "k=%d, i=%d\n", k, i );
*/
				if( k && k<*ncut-1 && j<*ncut-1 && !permit( seg1[k-1], seg1[j-1] ) ) continue;
				if( crossscore[i-1][k] > maxj )
				{
					pointi = k;
					maxi = crossscore[i-1][k];
				}
			}

			pointj = 0; maxj = 0.0;
			klim = i-2;
			for( k=0; k<klim; k++ )
			{
				if( k && k<*ncut-1 && i<*ncut-1 && !permit( seg2[k-1], seg2[i-1] ) ) continue;
				if( crossscore[k][j-1] > maxj )
				{
					pointj = k;
					maxj = crossscore[k][j-1];
				}
			}	

			maxi += penalty;
			maxj += penalty;

			maximum = crossscore[i-1][j-1];
			track[i][j] = 0;

			if( maximum < maxi )
			{
				maximum = maxi ;
				track[i][j] = j - pointi;
			}

			if( maximum < maxj )
			{
				maximum = maxj ;
				track[i][j] = pointj - i;
			}

			crossscore[i][j] += maximum;
		}
	}
#if 0
	for( i=0; i<*ncut; i++ ) 
	{
		for( j=0; j<*ncut; j++ )
			fprintf( stderr, "%3d ", track[i][j] );
		fprintf( stderr, "\n" );
	}
#endif


	result1[MAXSEG-1] = *ncut-1;
	result2[MAXSEG-1] = *ncut-1;

	for( i=MAXSEG-1; i>=1; i-- )
	{
		cur1 = result1[i];
		cur2 = result2[i];
		if( cur1 == 0 || cur2 == 0 ) break;
		shift = track[cur1][cur2];
		if( shift == 0 )
		{
			result1[i-1] = cur1 - 1;
			result2[i-1] = cur2 - 1;
			continue;
		}
		else if( shift > 0 )
		{
			result1[i-1] = cur1 - 1;
			result2[i-1] = cur2 - shift;
		}
		else if( shift < 0 )
		{
			result1[i-1] = cur1 + shift;
			result2[i-1] = cur2 - 1;
		}
	}

	count = 0;
	for( j=i; j<MAXSEG; j++ )
	{
		if( ocrossscore[result1[j]][result2[j]] == 0.0 ) continue;

		if( result1[j] == result1[j-1] || result2[j] == result2[j-1] )
			if( ocrossscore[result1[j]][result2[j]] > ocrossscore[result1[j-1]][result2[j-1]] )
				count--;
				
		cut1[count] = ocut1[result1[j]];
		cut2[count] = ocut2[result2[j]];

		count++;
	}

	*ncut = count;
#if 0
	for( i=0; i<*ncut; i++ )
		fprintf( stderr, "i=%d, cut1 = %d, cut2 = %d\n", i, cut1[i], cut2[i] );
#endif
}

void blockAlign3( int *cut1, int *cut2, Segment **seg1, Segment **seg2, double **ocrossscore, int *ncut )
// memory complexity = O(n^3), time complexity = O(n^2)
{
	int i, j, shift, cur1, cur2, count;
	static TLS int crossscoresize = 0;
	static TLS int jumpposi, *jumppos;
	static TLS double jumpscorei, *jumpscore;
	static TLS int *result1 = NULL;
	static TLS int *result2 = NULL;
	static TLS int *ocut1 = NULL;
	static TLS int *ocut2 = NULL;
	double maximum;
	static TLS double **crossscore = NULL;
	static TLS int **track = NULL;

	if( result1 == NULL )
	{
		result1 = AllocateIntVec( MAXSEG );
		result2 = AllocateIntVec( MAXSEG );
		ocut1 = AllocateIntVec( MAXSEG );
		ocut2 = AllocateIntVec( MAXSEG );
	}
    if( crossscoresize < *ncut+2 )
    {
        crossscoresize = *ncut+2;
		if( fftkeika ) fprintf( stderr, "allocating crossscore and track, size = %d\n", crossscoresize );
		if( track ) FreeIntMtx( track );
        if( crossscore ) FreeDoubleMtx( crossscore );
        if( jumppos ) FreeIntVec( jumppos );
        if( jumpscore ) FreeDoubleVec( jumpscore );
		track = AllocateIntMtx( crossscoresize, crossscoresize );
        crossscore = AllocateDoubleMtx( crossscoresize, crossscoresize );
        jumppos = AllocateIntVec( crossscoresize );
        jumpscore = AllocateDoubleVec( crossscoresize );
    }

#if 0
	for( i=0; i<*ncut-2; i++ )
		fprintf( stderr, "%d.start = %d, score = %f\n", i, seg1[i]->start, seg1[i]->score );

	for( i=0; i<*ncut; i++ )
		fprintf( stderr, "i=%d, cut1 = %d, cut2 = %d\n", i, cut1[i], cut2[i] );
	for( i=0; i<*ncut; i++ ) 
	{
		for( j=0; j<*ncut; j++ )
			fprintf( stderr, "%#4.0f ", ocrossscore[i][j] );
		fprintf( stderr, "\n" );
	}
#endif

	for( i=0; i<*ncut; i++ ) for( j=0; j<*ncut; j++ )  /* mudadanaa */
		crossscore[i][j] = ocrossscore[i][j];
	for( i=0; i<*ncut; i++ ) 
	{
		ocut1[i] = cut1[i];
		ocut2[i] = cut2[i];
	}
	for( j=0; j<*ncut; j++ )
	{
		jumpscore[j] = -999.999;
		jumppos[j] = -1;
	}

	for( i=1; i<*ncut; i++ )
	{

		jumpscorei = -999.999;
		jumpposi = -1;

		for( j=1; j<*ncut; j++ )
		{
#if 1
			fprintf( stderr, "in blockalign3, ### i=%d, j=%d\n", i, j );
#endif


#if 0
			for( k=0; k<j-2; k++ )
			{
/*
				fprintf( stderr, "k=%d, i=%d\n", k, i );
*/
				if( k && k<*ncut-1 && j<*ncut-1 && !permit( seg1[k-1], seg1[j-1] ) ) continue;
				if( crossscore[i-1][k] > maxj )
				{
					pointi = k;
					maxi = crossscore[i-1][k];
				}
			}

			pointj = 0; maxj = 0.0;
			for( k=0; k<i-2; k++ )
			{
				if( k && k<*ncut-1 && i<*ncut-1 && !permit( seg2[k-1], seg2[i-1] ) ) continue;
				if( crossscore[k][j-1] > maxj )
				{
					pointj = k;
					maxj = crossscore[k][j-1];
				}
			}	


			maxi += penalty;
			maxj += penalty;
#endif
			maximum = crossscore[i-1][j-1];
			track[i][j] = 0;

			if( maximum < jumpscorei && permit( seg1[jumpposi], seg1[i] ) )
			{
				maximum = jumpscorei;
				track[i][j] = j - jumpposi;
			}

			if( maximum < jumpscore[j] && permit( seg2[jumppos[j]], seg2[j] ) )
			{
				maximum = jumpscore[j];
				track[i][j] = jumpscore[j] - i;
			}

			crossscore[i][j] += maximum;

			if( jumpscorei < crossscore[i-1][j] )
			{
				jumpscorei = crossscore[i-1][j];
				jumpposi = j;
			}

			if( jumpscore[j] < crossscore[i][j-1] )
			{
				jumpscore[j] = crossscore[i][j-1];
				jumppos[j] = i;
			}
		}
	}
#if 0
	for( i=0; i<*ncut; i++ ) 
	{
		for( j=0; j<*ncut; j++ )
			fprintf( stderr, "%3d ", track[i][j] );
		fprintf( stderr, "\n" );
	}
#endif


	result1[MAXSEG-1] = *ncut-1;
	result2[MAXSEG-1] = *ncut-1;

	for( i=MAXSEG-1; i>=1; i-- )
	{
		cur1 = result1[i];
		cur2 = result2[i];
		if( cur1 == 0 || cur2 == 0 ) break;
		shift = track[cur1][cur2];
		if( shift == 0 )
		{
			result1[i-1] = cur1 - 1;
			result2[i-1] = cur2 - 1;
			continue;
		}
		else if( shift > 0 )
		{
			result1[i-1] = cur1 - 1;
			result2[i-1] = cur2 - shift;
		}
		else if( shift < 0 )
		{
			result1[i-1] = cur1 + shift;
			result2[i-1] = cur2 - 1;
		}
	}

	count = 0;
	for( j=i; j<MAXSEG; j++ )
	{
		if( ocrossscore[result1[j]][result2[j]] == 0.0 ) continue;

		if( result1[j] == result1[j-1] || result2[j] == result2[j-1] )
			if( ocrossscore[result1[j]][result2[j]] > ocrossscore[result1[j-1]][result2[j-1]] )
				count--;
				
		cut1[count] = ocut1[result1[j]];
		cut2[count] = ocut2[result2[j]];

		count++;
	}

	*ncut = count;
#if 0
	for( i=0; i<*ncut; i++ )
		fprintf( stderr, "i=%d, cut1 = %d, cut2 = %d\n", i, cut1[i], cut2[i] );
#endif
}

