#!/usr/bin/env python3
"""Turn Lalign11.c into a per-call check of the marker-based Lfill_int (NEON, SVE, AVX2 or AVX-512).

Every call runs the fill twice: first with its vector loops off, so the scalar tail fills every
cell of every row, into a scratch ijp; then with them on, into the real ijp.  The return value,
maxwm, the end point, every ijp cell (rows 1..lgth1, cols 1..lgth2: markers, 0 and localstop)
and every stored DP row that Ltracking's replay reads (lr rows 0..lgth1) must match bit for bit,
else abort.  The first-index search for the row maximum is gated the same way.
Usage: marks_check.py CORE_DIR_or_Lalign11.c  (rewrites in place; needs an arm64 Linux, AVX2 or AVX-512 build)
"""
import os, sys

p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'Lalign11.c')
s = open(p, encoding='latin-1').read()

head = 'static int Lfill_int( double **amino_dynamicmtx,'
i0 = s.index('#ifdef LFILL_MARKS\n/*\n * Lfill_int for AVX2')
i1 = s.index('#else\n' + head, i0)
body = s[i0:i1]
assert body.count(head) == 1
body = body.replace('if( lgth2 >= 7 )', 'if( Lfill_vec && lgth2 >= 7 )')
assert body.count('needwm = ( maxwm <= ithr );') == 1
body = body.replace('needwm = ( maxwm <= ithr );', 'needwm = !Lfill_vec || ( maxwm <= ithr );')
body = body.replace(head, 'static int Lfill_vec = 1;\nstatic int Lfill_int_impl( double **amino_dynamicmtx,')
n = 0
for pat in ('for( ; j+3<=lgth2; j+=4 )', 'for( ; j+7<=lgth2; j+=8 )', 'for( ; j+15<=lgth2; j+=16 )', 'for( ; j+vl-1<=lgth2; j+=vl )', 'for( ; j+2*vl-1<=lgth2; j+=2*vl )'):
    n += body.count(pat)
    body = body.replace(pat, pat.replace('for( ; ', 'for( ; Lfill_vec && '))
assert n >= 2, n

wrapper = r'''
static long Lfill_calls = 0, Lfill_cells = 0;
static void Lfill_report( void ) { fprintf( stderr, "MARKS_CHECK: %ld calls, %ld cells, all identical\n", Lfill_calls, Lfill_cells ); }
static int Lfill_int( double **amino_dynamicmtx, double **n_dynamicmtx, double scoreoffset,
                      char *s1, char *s2, int lgth1, int lgth2, int **ijp, int lstop,
                      double *maxwmpt, int *endalipt, int *endaljpt )
{
	int i, j, r0, r1, ei0, ej0, ei1, ej1;
	double m0, m1;
	int **sc = malloc( sizeof( int * ) * ( lgth1 + 1 ) ), *rows = NULL;
	if( Lfill_calls++ == 0 ) atexit( Lfill_report );
	for( i=0; i<=lgth1; i++ ) sc[i] = calloc( lgth2 + 1, sizeof( int ) );
	Lfill_vec = 0;
	r0 = Lfill_int_impl( amino_dynamicmtx, n_dynamicmtx, scoreoffset, s1, s2, lgth1, lgth2, sc, lstop, &m0, &ei0, &ej0 );
	if( r0 )
	{
		rows = malloc( sizeof( int ) * ( lgth1 + 1 ) * ( lgth2 + 1 ) );
		for( i=0; i<=lgth1; i++ ) memcpy( rows + (size_t)i * ( lgth2 + 1 ), LROW( i ), sizeof( int ) * ( lgth2 + 1 ) );
	}
	Lfill_vec = 1;
	r1 = Lfill_int_impl( amino_dynamicmtx, n_dynamicmtx, scoreoffset, s1, s2, lgth1, lgth2, ijp, lstop, &m1, &ei1, &ej1 );
	if( r0 != r1 ) { fprintf( stderr, "MARKS_CHECK: return %d vs %d\n", r0, r1 ); abort(); }
	if( r1 )
	{
		if( memcmp( &m0, &m1, sizeof( double ) ) || ei0 != ei1 || ej0 != ej1 )
		{ fprintf( stderr, "MARKS_CHECK: max %f/%f end %d,%d vs %d,%d\n", m0, m1, ei0, ej0, ei1, ej1 ); abort(); }
		for( i=1; i<=lgth1; i++ ) for( j=1; j<=lgth2; j++ )
			if( sc[i][j] != ijp[i][j] )
			{ fprintf( stderr, "MARKS_CHECK: ijp[%d][%d] %d vs %d (lgth %d,%d)\n", i, j, sc[i][j], ijp[i][j], lgth1, lgth2 ); abort(); }
		/* row lgth1's cell 0 is never written (no row follows it) */
		for( i=0; i<=lgth1; i++ ) for( j=( i == lgth1 ); j<=lgth2; j++ )
			if( rows[(size_t)i * ( lgth2 + 1 ) + j] != LROW( i )[j] )
			{ fprintf( stderr, "MARKS_CHECK: row %d cell %d %d vs %d (lgth %d,%d)\n", i, j, rows[(size_t)i * ( lgth2 + 1 ) + j], LROW( i )[j], lgth1, lgth2 ); abort(); }
		*maxwmpt = m1; *endalipt = ei1; *endaljpt = ej1;
		Lfill_cells += (long)lgth1 * lgth2;
	}
	free( rows );
	for( i=0; i<=lgth1; i++ ) free( sc[i] );
	free( sc );
	return( r1 );
}
'''
s = s[:i0] + body + wrapper + s[i1:]
open(p, 'w', encoding='latin-1').write(s)
print('patched %d vector loops' % n)
