#!/usr/bin/env python3
"""Turn mltaln9.c into a per-call check of igs_pairscores_x (MAFFT_AVX512X).

Every call of igs_pairscores() runs the bit-mask version into a scratch array and then the
original code (nxt arrays, gathered or scalar column sums) into out[]; when both succeed every
pair score must match bit for bit, and the bit-mask version must not succeed where the original
declines.  Each process reports "IGSX_CHECK: N calls (X by the bit-mask path), P pairs, all
identical" at exit.
Usage: igsx_check.py CORE_DIR_or_mltaln9.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'mltaln9.c')
s = open(p, encoding='latin-1').read()
call = '\tif( igs_pairscores_x( seq1, seq2, clus1, clus2, len, out, dzi ) ) return( 1 );\n'
assert s.count(call) == 1
s = s.replace(call, '\tigchk_x = igs_pairscores_x( seq1, seq2, clus1, clus2, len, igchk_buf, dzi );\n')
head = 'static int igs_pairscores( char **seq1, char **seq2, int clus1, int clus2, int len, double *out )\n{'
assert s.count(head) == 1
s = s.replace(head, '''static int igchk_x = 0; static double *igchk_buf = NULL;
static long igchk_calls = 0, igchk_xcalls = 0, igchk_pairs = 0;
static void igchk_report( void ) { fprintf( stderr, "\\nIGSX_CHECK: %ld calls (%ld by the bit-mask path), %ld pairs, all identical\\n", igchk_calls, igchk_xcalls, igchk_pairs ); }
static int igs_pairscores_orig( char **seq1, char **seq2, int clus1, int clus2, int len, double *out );
static int igs_pairscores( char **seq1, char **seq2, int clus1, int clus2, int len, double *out )
{
	int r, k;
	if( igchk_calls++ == 0 ) atexit( igchk_report );
	igchk_buf = realloc( igchk_buf, sizeof( double ) * ( (size_t)clus1 * clus2 + 1 ) );
	igchk_x = 0;
	r = igs_pairscores_orig( seq1, seq2, clus1, clus2, len, out );
	if( igchk_x && !r ) { fprintf( stderr, "IGSX_CHECK: bit-mask path succeeded where the original declined\\n" ); abort(); }
	if( igchk_x && r )
	{
		igchk_xcalls++;
		for( k=0; k<clus1*clus2; k++ ) if( memcmp( out + k, igchk_buf + k, sizeof( double ) ) )
		{ fprintf( stderr, "IGSX_CHECK: pair %d (%d x %d, len %d): %.17g vs %.17g\\n", k, clus1, clus2, len, out[k], igchk_buf[k] ); abort(); }
		igchk_pairs += (long)clus1 * clus2;
	}
	return( r );
}
static int igs_pairscores_orig( char **seq1, char **seq2, int clus1, int clus2, int len, double *out )
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched igs_pairscores')
