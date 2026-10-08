#!/usr/bin/env python3
"""Per-call check of alignableReagion (fftFunctions.c; AVX-512BW or Linux arm64 NEON): vector passes and the pre-marked
characters off vs on; return value and every segment's start/end/center/score (bitwise) must
match.  Usage: areg_check.py COREDIR (rewrites fftFunctions.c)"""
import os, sys
p = os.path.join(sys.argv[1], 'fftFunctions.c')
s = open(p, encoding='latin-1').read()
sig = 'int alignableReagion( int    clus1, int    clus2, '
assert s.count(sig) == 1
s = s.replace(sig, 'static int alignableReagion_impl( int    clus1, int    clus2, ')
s = s.replace('#include "mltaln.h"\n', '#include "mltaln.h"\nstatic int ar_vec = 1;\n', 1)
n = 0
for pat in ('for( i=0; i+64<=len; i+=64 )', 'for( i=0; i+16<=len; i+=16 )', 'for( q=0; common[q]; q++ )'):
    n += s.count(pat)
for pat, rep in (('for( i=0; i+64<=len; i+=64 )', 'for( i=0; ar_vec && i+64<=len; i+=64 )'),
                 ('for( i=0; i+16<=len; i+=16 )', 'for( i=0; ar_vec && i+16<=len; i+=16 )'),
                 ('for( q=0; common[q]; q++ )', 'for( q=0; ar_vec && common[q]; q++ )')):
    s = s.replace(pat, rep)
assert n == 7, n
wrapper = r'''
int alignableReagion( int clus1, int clus2, char **seq1, char **seq2, double *eff1, double *eff2, Segment *seg )
{
	static long calls = 0;
	static Segment *s0 = NULL;
	static int s0n = 0;
	int v0, v1, k, cap = 100000;
	if( clus1 == 0 ) return( alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, seg ) );
	if( s0n < cap ) { s0 = realloc( s0, sizeof( Segment ) * cap ); s0n = cap; }
	ar_vec = 0; v0 = alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, s0 );
	ar_vec = 1; v1 = alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, seg );
	if( v0 != v1 ) { fprintf( stderr, "AREG_CHECK: value %d vs %d\n", v0, v1 ); abort(); }
	for( k=0; k<v1; k++ )
		if( s0[k].start != seg[k].start || s0[k].end != seg[k].end || s0[k].center != seg[k].center || memcmp( &s0[k].score, &seg[k].score, sizeof( double ) ) )
		{ fprintf( stderr, "AREG_CHECK: segment %d differs\n", k ); abort(); }
	calls++;
	if( ( calls & ( calls - 1 ) ) == 0 ) fprintf( stderr, "AREG_CHECK: %ld calls, all identical\n", calls );
	return( v1 );
}
'''
s = s + wrapper
open(p, 'w', encoding='latin-1').write(s)
print('patched alignableReagion: %d gates' % n)
