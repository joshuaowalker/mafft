/*
 * mafft_simd.h: which vector code a build compiles, decided once for all of core/.
 *
 * The exact-speedup code paths (see README.md) test only the MAFFT_* macros below, never the
 * compiler's target macros directly:
 *
 *   MAFFT_A64           AArch64 (Advanced SIMD/NEON is always present)
 *   MAFFT_SVE           AArch64 with SVE
 *   MAFFT_SSE41         x86-64 with SSE4.1
 *   MAFFT_AVX2          x86-64 with AVX2
 *   MAFFT_AVX512        x86-64 with AVX-512 F, BW and VL (-march=x86-64-v4 and later)
 *   MAFFT_AVX512_VBMI2  MAFFT_AVX512 plus VBMI2 (Ice Lake, Zen 4 and later)
 *   MAFFT_STOCK_FMA     1 when the stock build of this target contracts a*b+c into one fused
 *                       multiply-add (clang on arm64), else 0; override with -DMAFFT_STOCK_FMA=0/1
 *
 * and the multiply-adds that round like the stock build: MULADD (scalar), NMULADD (2 doubles,
 * NEON), VMULADD4 (4 doubles, AVX2), VMULADD (8 doubles, AVX-512).
 */
#ifndef MAFFT_SIMD_H
#define MAFFT_SIMD_H

#include <math.h>
#include <stdint.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define MAFFT_A64 1
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define MAFFT_SVE 1
#endif
#endif

#if defined(__x86_64__) && ( defined(__SSE4_1__) || defined(__AVX2__) || defined(__AVX512F__) )
#include <immintrin.h>
#if defined(__SSE4_1__)
#define MAFFT_SSE41 1
#endif
#if defined(__AVX2__)
#define MAFFT_AVX2 1
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define MAFFT_AVX512 1
#if defined(__AVX512VBMI2__)
#define MAFFT_AVX512_VBMI2 1
#endif
#endif
#endif

/* a*b+c rounded the way the stock build rounds it, for rewritten code that must stay
   bit-identical.  clang on arm64 contracts a*b+c into one fused (singly rounded) multiply-add;
   gcc with -std=c99 (the Makefile default) never contracts, and the x86-64 baseline has no FMA
   at all.  Override with -DMAFFT_STOCK_FMA=0/1 for a reference built differently. */
#ifndef MAFFT_STOCK_FMA
#if defined(__aarch64__) && defined(__clang__)
#define MAFFT_STOCK_FMA 1
#else
#define MAFFT_STOCK_FMA 0
#endif
#endif
#if MAFFT_STOCK_FMA
#define MULADD(a,b,c) fma( (a), (b), (c) )
#else
#define MULADD(a,b,c) ( (a)*(b) + (c) )
#endif

#if defined(MAFFT_A64)
#if MAFFT_STOCK_FMA
#define NMULADD(a,b,c) vfmaq_f64( (c), (a), (b) )
#else
#define NMULADD(a,b,c) vaddq_f64( vmulq_f64( (a), (b) ), (c) )
#endif
#endif

/* The AVX2 (4 x double) paths: in a build that does not contract a*b+c, or, with
   MAFFT_STOCK_FMA, where FMA instructions can round like the contracting stock build. */
#if defined(MAFFT_AVX2) && ( !MAFFT_STOCK_FMA || defined(__FMA__) )
#define MAFFT_AVX2_PATHS 1
#if MAFFT_STOCK_FMA
#define VMULADD4(a,b,c) _mm256_fmadd_pd( (a), (b), (c) )
#else
#define VMULADD4(a,b,c) _mm256_add_pd( _mm256_mul_pd( (a), (b) ), (c) )
#endif
/* 4 x 64-bit compare mask -> 4 x 32-bit mask */
#define PACKMASK4(c) _mm256_castsi256_si128( _mm256_permutevar8x32_epi32( _mm256_castpd_si256( c ), _mm256_setr_epi32( 0, 2, 4, 6, 0, 2, 4, 6 ) ) )
#endif
/* AVX2 code that came after dikarya1 (the gapruns and alignableReagion passes).  In the
   non-contracting (gcc) class it would change the objects of the dikarya1 build, which was
   verified on its own hardware, so there it is opt-in: -DMAFFT_AVX2_EXTRA=1. */
#if defined(MAFFT_AVX2) && ( MAFFT_STOCK_FMA || defined(MAFFT_AVX2_EXTRA) )
#define MAFFT_AVX2_NEWPATHS 1
#endif

#if defined(MAFFT_AVX512)
#if MAFFT_STOCK_FMA
#define VMULADD(a,b,c) _mm512_fmadd_pd( (a), (b), (c) )
#else
#define VMULADD(a,b,c) _mm512_add_pd( _mm512_mul_pd( (a), (b) ), (c) )
#endif
#endif


/*
 * Byte masks, for the passes that look for gaps or particular characters in a row of bytes
 * (gapruns, alignableReagion, cpmx_calc_new, makeresmap).  When MAFFT_BM_BYTES is defined:
 *
 *   mafft_bm_eq( p, c )        a mask with one bit per byte of p[0 .. MAFFT_BM_BYTES-1]: bit
 *                              k * MAFFT_BM_BITS is set when p[k] == c, no other bit is set
 *   mafft_bm_eqany( p, c, n )  the same for p[k] equal to any of c[0 .. n-1]
 *   MAFFT_BM_ALL               the mask with every byte's bit set
 *   MAFFT_BM_INDEX( m )        the byte index of the lowest set bit of m (m != 0); clear it with
 *                              m &= m - 1
 *   mafft_bm_addmask( r, m, v )  r[k] += v for every byte k set in m, in any order: each element
 *                              receives one add of v, the same as the scalar loop gives it
 *
 * AVX-512BW: 64 bytes, one bit each; AVX2: 32 bytes, one bit each; NEON: 16 bytes, one bit per
 * 4 (the low bit of each nibble, which vshrn produces without a movemask).
 *
 * MAFFT_BM_PERCHAR (AVX-512): a block mixing characters is worth splitting into one masked add
 * per character; elsewhere such a block is cheaper element by element (a block of gaps is one
 * vector add everywhere).
 */
#if defined(MAFFT_AVX512)
#define MAFFT_BM_BYTES 64
#define MAFFT_BM_BITS 1
#define MAFFT_BM_ALL 0xffffffffffffffffULL
#define MAFFT_BM_PERCHAR 1
static inline unsigned long long mafft_bm_eq( const unsigned char *p, unsigned char c )
{
	return( _mm512_cmpeq_epi8_mask( _mm512_loadu_si512( (const void *)p ), _mm512_set1_epi8( (char)c ) ) );
}
static inline unsigned long long mafft_bm_eqany( const unsigned char *p, const unsigned char *c, int n )
{
	__m512i x = _mm512_loadu_si512( (const void *)p );
	unsigned long long m = 0;
	int k;
	for( k=0; k<n; k++ ) m |= _mm512_cmpeq_epi8_mask( x, _mm512_set1_epi8( (char)c[k] ) );
	return( m );
}
static inline void mafft_bm_addmask( double *r, unsigned long long m, double v )
{
	const __m512d vv = _mm512_set1_pd( v );
	int q;
	for( q=0; q<8; q++ )
	{
		__mmask8 m8 = (__mmask8)( m >> ( 8 * q ) );
		if( m8 ) _mm512_mask_storeu_pd( r + 8 * q, m8, _mm512_add_pd( _mm512_maskz_loadu_pd( m8, r + 8 * q ), vv ) );
	}
}
#elif defined(MAFFT_AVX2)
#define MAFFT_BM_BYTES 32
#define MAFFT_BM_BITS 1
#define MAFFT_BM_ALL 0xffffffffULL
static inline unsigned long long mafft_bm_eq( const unsigned char *p, unsigned char c )
{
	return( (unsigned int)_mm256_movemask_epi8( _mm256_cmpeq_epi8( _mm256_loadu_si256( (const __m256i *)p ), _mm256_set1_epi8( (char)c ) ) ) );
}
static inline unsigned long long mafft_bm_eqany( const unsigned char *p, const unsigned char *c, int n )
{
	__m256i x = _mm256_loadu_si256( (const __m256i *)p ), e = _mm256_setzero_si256();
	int k;
	for( k=0; k<n; k++ ) e = _mm256_or_si256( e, _mm256_cmpeq_epi8( x, _mm256_set1_epi8( (char)c[k] ) ) );
	return( (unsigned int)_mm256_movemask_epi8( e ) );
}
#elif defined(MAFFT_A64)
#define MAFFT_BM_BYTES 16
#define MAFFT_BM_BITS 4
#define MAFFT_BM_ALL 0x1111111111111111ULL
static inline unsigned long long mafft_bm_eq( const unsigned char *p, unsigned char c )
{
	uint8x16_t e = vceqq_u8( vld1q_u8( p ), vdupq_n_u8( c ) );
	return( vget_lane_u64( vreinterpret_u64_u8( vshrn_n_u16( vreinterpretq_u16_u8( e ), 4 ) ), 0 ) & MAFFT_BM_ALL );
}
static inline unsigned long long mafft_bm_eqany( const unsigned char *p, const unsigned char *c, int n )
{
	uint8x16_t x = vld1q_u8( p ), e = vdupq_n_u8( 0 );
	int k;
	for( k=0; k<n; k++ ) e = vorrq_u8( e, vceqq_u8( x, vdupq_n_u8( c[k] ) ) );
	return( vget_lane_u64( vreinterpret_u64_u8( vshrn_n_u16( vreinterpretq_u16_u8( e ), 4 ) ), 0 ) & MAFFT_BM_ALL );
}
#endif
#if defined(MAFFT_BM_BYTES)
#define MAFFT_BM_INDEX( m ) ( __builtin_ctzll( m ) / MAFFT_BM_BITS )
#if !defined(MAFFT_AVX512)
static inline void mafft_bm_addmask( double *r, unsigned long long m, double v )
{
	if( m == MAFFT_BM_ALL ) { int k; for( k=0; k<MAFFT_BM_BYTES; k++ ) r[k] += v; }
	else for( ; m; m &= m - 1 ) r[MAFFT_BM_INDEX( m )] += v;
}
#endif
#endif

/* a hardware population count (x86-64-v2 and later, every AArch64) */
#if defined(__POPCNT__) || defined(MAFFT_A64)
#define MAFFT_POPCNT 1
#endif

/*
 * mafft_eq64( p, c ): bit k set when p[k] == c, for k < 64, one bit per byte, in every build
 * (p must have 64 readable bytes; mafft_eq64n handles a shorter tail of n bytes).
 */
static inline unsigned long long mafft_eq64n( const unsigned char *p, int n, unsigned char c )
{
	unsigned long long m = 0;
	int k;
	for( k=0; k<n; k++ ) if( p[k] == c ) m |= 1ULL << k;
	return( m );
}
#if defined(MAFFT_A64)
/* the nibble mask of 16 bytes (bit 4k) packed into 16 bits (bit k) */
static inline unsigned long long mafft_pack_nibbles( unsigned long long x )
{
	x = ( x | ( x >> 3 ) ) & 0x0303030303030303ULL;
	x = ( x | ( x >> 6 ) ) & 0x000F000F000F000FULL;
	x = ( x | ( x >> 12 ) ) & 0x000000FF000000FFULL;
	x = ( x | ( x >> 24 ) ) & 0xFFFFULL;
	return( x );
}
#endif
static inline unsigned long long mafft_eq64( const unsigned char *p, unsigned char c )
{
#if defined(MAFFT_AVX512)
	return( mafft_bm_eq( p, c ) );
#elif defined(MAFFT_AVX2)
	return( mafft_bm_eq( p, c ) | ( mafft_bm_eq( p + 32, c ) << 32 ) );
#elif defined(MAFFT_A64)
	return( mafft_pack_nibbles( mafft_bm_eq( p, c ) ) | ( mafft_pack_nibbles( mafft_bm_eq( p + 16, c ) ) << 16 )
	      | ( mafft_pack_nibbles( mafft_bm_eq( p + 32, c ) ) << 32 ) | ( mafft_pack_nibbles( mafft_bm_eq( p + 48, c ) ) << 48 ) );
#else
	return( mafft_eq64n( p, 64, c ) );
#endif
}

#endif
