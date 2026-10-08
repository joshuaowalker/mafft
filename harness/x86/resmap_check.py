#!/usr/bin/env python3
"""Turn mltaln9.c into a per-call check of makeresmap_avx512 (fillimp_track, MAFFT_AVX512X).

Every call runs the original makeresmap() into a scratch map and the AVX-512 version into the
real one; the residue counts and every map entry must match, else abort.
Each process reports "RESMAP_CHECK: N calls, M residues, all identical" at exit.
Usage: resmap_check.py CORE_DIR_or_mltaln9.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'mltaln9.c')
s = open(p, encoding='latin-1').read()
head = 'static int makeresmap_avx512( char *seq, int *map )\n{'
assert s.count(head) == 1
s = s.replace(head, '''static long rm_calls = 0, rm_res = 0;
static void rm_report( void ) { fprintf( stderr, "\\nRESMAP_CHECK: %ld calls, %ld residues, all identical\\n", rm_calls, rm_res ); }
static int makeresmap_avx512_impl( char *seq, int *map );
static int makeresmap_avx512( char *seq, int *map )
{
	int len = strlen( seq ), *ref = malloc( sizeof( int ) * ( len + 1 ) ), n0, n1, k;
	if( rm_calls++ == 0 ) atexit( rm_report );
	n0 = makeresmap( seq, ref );
	n1 = makeresmap_avx512_impl( seq, map );
	if( n0 != n1 ) { fprintf( stderr, "RESMAP_CHECK: count %d vs %d\\n", n0, n1 ); abort(); }
	for( k=0; k<n0; k++ ) if( ref[k] != map[k] ) { fprintf( stderr, "RESMAP_CHECK: map[%d] %d vs %d\\n", k, ref[k], map[k] ); abort(); }
	rm_res += n0;
	free( ref );
	return( n1 );
}
static int makeresmap_avx512_impl( char *seq, int *map )
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched makeresmap_avx512')
