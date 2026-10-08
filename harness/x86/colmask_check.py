#!/usr/bin/env python3
"""Turn mltaln9.c into a per-call check of cpmx_colmask (MAFFT_AVX512X).

The match_calc set-ups in Salignmm.c and partSalignmm.c (and mc_build) list each column's
nonzero letters from cpmx_colmask() instead of testing "if( cpmx2[l][j] )" for every l; they visit
the set bits in ascending l, the order of the original loop.  This check recomputes every mask
with the original scalar test after each call and aborts on any difference.
Each process reports "COLMASK_CHECK: N calls, M columns, all identical" at exit.
Usage: colmask_check.py CORE_DIR_or_mltaln9.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'mltaln9.c')
s = open(p, encoding='latin-1').read()
head = 'void cpmx_colmask( double **cpmx, int nalph, int lgth, unsigned int *mask )\n{'
assert s.count(head) == 1
s = s.replace(head, '''static long cm_calls = 0, cm_cols = 0;
static void cm_report( void ) { fprintf( stderr, "\\nCOLMASK_CHECK: %ld calls, %ld columns, all identical\\n", cm_calls, cm_cols ); }
static void cpmx_colmask_impl( double **cpmx, int nalph, int lgth, unsigned int *mask );
void cpmx_colmask( double **cpmx, int nalph, int lgth, unsigned int *mask )
{
	int j, l;
	if( cm_calls++ == 0 ) atexit( cm_report );
	cpmx_colmask_impl( cpmx, nalph, lgth, mask );
	for( j=0; j<lgth; j++ )
	{
		unsigned int m = 0;
		for( l=0; l<nalph; l++ ) if( cpmx[l][j] ) m |= 1u << l;
		if( m != mask[j] ) { fprintf( stderr, "COLMASK_CHECK: column %d: %x vs %x\\n", j, m, mask[j] ); abort(); }
	}
	cm_cols += lgth;
}
static void cpmx_colmask_impl( double **cpmx, int nalph, int lgth, unsigned int *mask )
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched cpmx_colmask')
