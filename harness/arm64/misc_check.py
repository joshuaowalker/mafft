#!/usr/bin/env python3
"""Per-call checks of two small Linux arm64 NEON rewrites:
- makeresmap (mltaln9.c, the residue->column map of fillimp_track): the NEON version against
  the original one-column loop, return value and every map entry;
- part_imp_match_out_vead_gapmap (partSalignmm.c, adds an impmtx row through gapmap2): the
  NEON version against the original scalar loop on a copy of imp, every element bitwise.
Abort on any difference; counts are printed at power-of-two calls.
Usage: misc_check.py COREDIR (rewrites mltaln9.c and partSalignmm.c)"""
import os, sys
core = sys.argv[1]

p = os.path.join(core, 'mltaln9.c')
s = open(p, encoding='latin-1').read()
sig = 'static int makeresmap( char *seq, int *map )\n{'
assert s.count(sig) == 1
s = s.replace(sig, 'static int makeresmap_impl( char *seq, int *map )\n{')
i = s.index('static int makeresmap_impl(')
j = s.index('\n}\n', i) + 3
s = s[:j] + r'''static int makeresmap( char *seq, int *map )
{
	static long calls = 0;
	int n0 = 0, n1, col, *m0 = malloc( sizeof( int ) * ( strlen( seq ) + 1 ) );
	for( col=0; seq[col]; col++ ) if( seq[col] != '-' ) m0[n0++] = col;
	n1 = makeresmap_impl( seq, map );
	if( n0 != n1 || memcmp( m0, map, sizeof( int ) * n0 ) ) { fprintf( stderr, "RESMAP_CHECK: differs\n" ); abort(); }
	free( m0 );
	calls++;
	if( ( calls & ( calls - 1 ) ) == 0 ) fprintf( stderr, "RESMAP_CHECK: %ld calls, all identical\n", calls );
	return( n1 );
}
''' + s[j:]
open(p, 'w', encoding='latin-1').write(s)

p = os.path.join(core, 'partSalignmm.c')
s = open(p, encoding='latin-1').read()
sig = 'static void part_imp_match_out_vead_gapmap( double *imp, int i1, int lgth2, int start2, int *gapmap2 )\n{'
assert s.count(sig) == 1
s = s.replace(sig, 'static void vead_impl( double *imp, int i1, int lgth2, int start2, int *gapmap2 )\n{')
i = s.index('static void vead_impl(')
j = s.index('\n}\n', i) + 3
s = s[:j] + r'''static void part_imp_match_out_vead_gapmap( double *imp, int i1, int lgth2, int start2, int *gapmap2 )
{
	static long calls = 0, cells = 0;
	double *c0 = malloc( sizeof( double ) * ( lgth2 + 1 ) );
	int j;
	for( j=0; j<lgth2; j++ ) c0[j] = imp[j] + impmtx[i1][start2+gapmap2[j]];
	vead_impl( imp, i1, lgth2, start2, gapmap2 );
	if( memcmp( c0, imp, sizeof( double ) * lgth2 ) ) { fprintf( stderr, "VEAD_CHECK: differs\n" ); abort(); }
	free( c0 );
	cells += lgth2;
	calls++;
	if( ( calls & ( calls - 1 ) ) == 0 ) fprintf( stderr, "VEAD_CHECK: %ld calls, %ld cells, all identical\n", calls, cells );
}
''' + s[j:]
open(p, 'w', encoding='latin-1').write(s)
print('patched makeresmap and part_imp_match_out_vead_gapmap')
