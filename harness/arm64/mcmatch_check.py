#!/usr/bin/env python3
"""Per-call check of mc_match (partSalignmm.c; AVX-512 or Linux arm64 NEON): vector loops off vs on, match[0..lgth2) bitwise.
Usage: mcmatch_check.py COREDIR (rewrites partSalignmm.c)"""
import os, sys
p = os.path.join(sys.argv[1], 'partSalignmm.c')
s = open(p, encoding='latin-1').read()
sig = 'static void mc_match( double *match, double **cpmx1, int i1, int lgth2 )'
assert s.count(sig) == 1
s = s.replace(sig, 'static int mc_vec = 1;\nstatic void mc_match_impl( double *match, double **cpmx1, int i1, int lgth2 )')
n = 0
for pat in ('for( ; l+8<=nalphabets; l+=8 )', 'for( ; n+8<=gsize; n+=8 )', 'for( ; l+2<=nalphabets; l+=2 )', 'for( ; l<nalphabets; l++ ) /* an odd letter left over */', 'for( ; n+2<=gsize; n+=2 )'):
    n += s.count(pat)
    s = s.replace(pat, pat.replace('for( ; ', 'for( ; mc_vec && '))
assert n >= 6, n
wrapper = r'''
static void mc_match( double *match, double **cpmx1, int i1, int lgth2 )
{
	static long calls = 0, cells = 0;
	double *m0 = malloc( sizeof( double ) * ( lgth2 + 1 ) );
	int j;
	mc_vec = 0; mc_match_impl( m0, cpmx1, i1, lgth2 );
	mc_vec = 1; mc_match_impl( match, cpmx1, i1, lgth2 );
	for( j=0; j<lgth2; j++ )
		if( memcmp( m0+j, match+j, sizeof( double ) ) )
		{ fprintf( stderr, "MC_CHECK: col %d of %d: %.17g vs %.17g\n", j, lgth2, m0[j], match[j] ); abort(); }
	free( m0 );
	calls++; cells += lgth2;
	if( ( calls & ( calls - 1 ) ) == 0 ) fprintf( stderr, "MC_CHECK: %ld calls, %ld cells, all identical\n", calls, cells );
}
'''
anchor = 'static void match_calc( double *match, double **cpmx1, double **cpmx2, int i1, int lgth2, double **doublework, int **intwork, int initialize )'
assert s.count(anchor) == 1
s = s.replace(anchor, wrapper + '\n' + anchor)
open(p, 'w', encoding='latin-1').write(s)
print('patched mc_match: %d vector loops' % n)
