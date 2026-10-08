#!/usr/bin/env python3
"""Turn partSalignmm.c into a per-call check of at_expand (Atracking_localhom output rows,
MAFFT_AVX512X).

After each at_expand(), every output row is rebuilt with the original loop
(d[c] = col[c] >= 0 ? s[col[c]] : gap, then the NUL) and compared byte for byte, else abort.
Each process reports "ATEXP_CHECK: N calls, R rows, all identical" at exit.
Usage: atexp_check.py CORE_DIR_or_partSalignmm.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'partSalignmm.c')
s = open(p, encoding='latin-1').read()
head = 'static void at_expand( char **seq, char **mseq, int n, int *col, int ncol, int cp, int first, char gapc )\n{'
assert s.count(head) == 1
s = s.replace(head, '''static long ae_calls = 0, ae_rows = 0;
static void ae_report( void ) { fprintf( stderr, "\\nATEXP_CHECK: %ld calls, %ld rows, all identical\\n", ae_calls, ae_rows ); }
static void at_expand_impl( char **seq, char **mseq, int n, int *col, int ncol, int cp, int first, char gapc );
static void at_expand( char **seq, char **mseq, int n, int *col, int ncol, int cp, int first, char gapc )
{
	int r, c;
	char *ref = malloc( ncol + 1 );
	if( ae_calls++ == 0 ) atexit( ae_report );
	at_expand_impl( seq, mseq, n, col, ncol, cp, first, gapc );
	for( r=0; r<n; r++ )
	{
		for( c=0; c<ncol; c++ ) ref[c] = ( col[c] >= 0 ) ? seq[r][col[c]] : gapc;
		ref[ncol] = 0;
		if( memcmp( ref, mseq[r], ncol + 1 ) ) { fprintf( stderr, "ATEXP_CHECK: row %d differs (ncol %d)\\n", r, ncol ); abort(); }
	}
	ae_rows += n;
	free( ref );
}
static void at_expand_impl( char **seq, char **mseq, int n, int *col, int ncol, int cp, int first, char gapc )
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched at_expand')
