#!/usr/bin/env python3
"""Per-call check of igs_pairscores (mltaln9.c) on Linux arm64: the NEON igs_prep and table-lookup
column sums (igs_vec) off vs on; return value and every out[i*clus2+j] bitwise."""
import os, sys
p = os.path.join(sys.argv[1], 'mltaln9.c')
s = open(p, encoding='latin-1').read()
sig = 'static int igs_pairscores( char **seq1, char **seq2, int clus1, int clus2, int len, double *out )\n{'
assert s.count(sig) == 1
s = s.replace(sig, 'static int igs_pairscores_impl( char **seq1, char **seq2, int clus1, int clus2, int len, double *out )\n{')
assert s.count('static int igs_vec = 1;') == 1
wrapper = r'''
static int igs_pairscores( char **seq1, char **seq2, int clus1, int clus2, int len, double *out )
{
	static long calls = 0, cells = 0;
	double *o0 = malloc( sizeof( double ) * clus1 * clus2 );
	int r0, r1, k;
	igs_vec = 0; r0 = igs_pairscores_impl( seq1, seq2, clus1, clus2, len, o0 );
	igs_vec = 1; r1 = igs_pairscores_impl( seq1, seq2, clus1, clus2, len, out );
	if( r0 != r1 ) { fprintf( stderr, "IGS_CHECK: return %d vs %d\n", r0, r1 ); abort(); }
	if( r1 ) for( k=0; k<clus1*clus2; k++ ) if( memcmp( o0+k, out+k, sizeof( double ) ) ) { fprintf( stderr, "IGS_CHECK: pair %d\n", k ); abort(); }
	free( o0 );
	cells += (long)clus1 * clus2;
	calls++;
	if( ( calls & ( calls - 1 ) ) == 0 ) fprintf( stderr, "IGS_CHECK: %ld calls, %ld pairs, all identical\n", calls, cells );
	return( r1 );
}
'''
# insert after the impl's body: find the next "\n}\n" after impl start at top level
start = s.index('static int igs_pairscores_impl(')
k = s.index('\n{', start) + 1; depth = 0
while True:
    if s[k] == '{': depth += 1
    elif s[k] == '}':
        depth -= 1
        if depth == 0: break
    k += 1
s = s[:k+1] + '\n' + wrapper + s[k+1:]
open(p, 'w', encoding='latin-1').write(s)
print('patched igs_pairscores')
