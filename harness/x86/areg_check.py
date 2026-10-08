#!/usr/bin/env python3
"""Turn fftFunctions.c into a per-call check of the bin-major profiles in alignableReagion
(MAFFT_AVX512X).

After the masked accumulation and the site scores, every call rebuilds the column-major
profiles with the original scalar loops (cp[i*nu+ci] += e, rows in order), recomputes each site
score from them with the original loop, and compares stra[0..len) bit for bit, else abort.
Each process reports "AREG_CHECK: N calls, M sites, all identical" at exit.
Usage: areg_check.py CORE_DIR_or_fftFunctions.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'fftFunctions.c')
s = open(p, encoding='latin-1').read()
i0 = s.index('/* Bin-major profiles (cp[ci*len+i])')
anchor = '\t\t\tgoto profiles_done;\n#else\n'
k = s.index(anchor, i0)
check = r'''			{
				static long ac_calls = 0, ac_sites = 0;
				double *r1 = calloc( (size_t)len * nu * 2 + 1, sizeof( double ) ), *r2 = r1 + (size_t)len * nu, sc;
				int a, b;
				for( j=0; j<clus1; j++ ) { unsigned char *s = (unsigned char *)seq1[j]; double e = eff1[j]; for( i=0; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) r1[i*nu+ci] += e; } }
				for( j=0; j<clus2; j++ ) { unsigned char *s = (unsigned char *)seq2[j]; double e = eff2[j]; for( i=0; i<len; i++ ) { int ci = cidx[s[i]]; if( ci >= 0 ) r2[i*nu+ci] += e; } }
				for( i=0; i<len; i++ )
				{
					double *q1 = r1 + (size_t)i * nu, *q2 = r2 + (size_t)i * nu;
					sc = 0.0;
					for( a=0; a<nu; a++ )
					{
						double p1 = q1[a];
						if( !p1 ) continue;
						for( b=0; b<nu; b++ )
						{
							double p2 = q2[b];
							if( !p2 ) continue;
							sc += n_disFFT[ub[a]][ub[b]] * p1 * p2;
						}
					}
					sc /= totaleff;
					if( memcmp( &sc, stra + i, sizeof( double ) ) ) { fprintf( stderr, "AREG_CHECK: site %d %a vs %a\n", i, stra[i], sc ); abort(); }
				}
				for( i=0; i<len; i++ ) for( a=0; a<nu; a++ )
					if( memcmp( r1 + (size_t)i*nu + a, cp1 + (size_t)a*len + i, sizeof( double ) ) || memcmp( r2 + (size_t)i*nu + a, cp2 + (size_t)a*len + i, sizeof( double ) ) )
					{ fprintf( stderr, "AREG_CHECK: profile cell %d,%d differs\n", i, a ); abort(); }
				free( r1 );
				ac_calls++; ac_sites += len;
				if( ( ac_calls & ( ac_calls - 1 ) ) == 0 ) fprintf( stderr, "\nAREG_CHECK: %ld calls, %ld sites, all identical\n", ac_calls, ac_sites );
			}
'''
s = s[:k] + check + s[k:]
open(p, 'w', encoding='latin-1').write(s)
print('patched alignableReagion')
