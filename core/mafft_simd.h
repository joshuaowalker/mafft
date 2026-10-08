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
 *   MAFFT_AVX512X       MAFFT_AVX512 plus VPOPCNTDQ and VBMI2
 *   MAFFT_STOCK_FMA     1 when the stock build of this target contracts a*b+c into one fused
 *                       multiply-add (clang on arm64), else 0; override with -DMAFFT_STOCK_FMA=0/1
 *
 * and the multiply-adds that round like the stock build: MULADD (scalar), NMULADD (2 doubles,
 * NEON), VMULADD4 (4 doubles, AVX2), VMULADD (8 doubles, AVX-512).
 */
#ifndef MAFFT_SIMD_H
#define MAFFT_SIMD_H

#include <math.h>

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
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VBMI2__)
#define MAFFT_AVX512X 1
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

#endif
