#!/usr/bin/env python3
"""Turn Lalign11.c into a per-call check of Lfill_kb and its traceback (opt8; MAFFT_AVX512 builds).

Every call of Lfill_kb first runs a reference fill: the opt5 integer Lfill_int that stores the
original offsets in ijp (the version for builds without the marker fill), with all its vector
loops switched off so that only its scalar code runs, a transcription of the original double
fill that the opt5 harness checked against it.  Then the byte fill runs, and the return value,
maxwm, the end point and, for every cell (i, j) in 1..lgth1 x 1..lgth2, the offset given by the
kind byte (markers resolved by the lres_hk()/lres_vmp() rules over the stored striped rows) must equal the
reference ijp cell, else abort.  After
Ltracking (reading the kind bytes), Ltracking is also run over the reference ijp, and the two aligned
strings and offsets must match.
Each process reports "KB_CHECK: N calls, M cells, T tracks, all identical" at exit.
Usage: kb_check.py CORE_DIR_or_Lalign11.c  (rewrites in place; build with AVX-512, e.g.
-march=x86-64-v4)
"""
import os, sys

p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'Lalign11.c')
s = open(p, encoding='latin-1').read()

# 1. the reference: the opt5 offset fill (the #else branch of LFILL_MARKS), scalar only
head = 'static int Lfill_int( double **amino_dynamicmtx,'
i0 = s.index('#else\n' + head) + len('#else\n')
i1 = s.index('\n}\n#endif\n', i0) + 3
ref = s[i0:i1].replace(head, 'static int Lfill_ref( double **amino_dynamicmtx,', 1)
n = 0
for pat in ('for( ; j+15<=lgth2; j+=16 )', 'for( ; j+7<=lgth2; j+=8 )', 'for( ; j+3<=lgth2; j+=4 )', 'for( ; j+3<lgth2; j+=4 )'):
    n += ref.count(pat)
    ref = ref.replace(pat, pat.replace('for( ; ', 'for( ; 0 && '))
assert n >= 1, n

# 2. rename the bit fill and add a checking wrapper after Ltracking_kb
s = s.replace('static int Lfill_kb( double **amino_dynamicmtx,', 'static int Lfill_ref( double **amino_dynamicmtx, double **n_dynamicmtx, double scoreoffset, char *s1, char *s2, int lgth1, int lgth2, int **ijp, int lstop, double *maxwmpt, int *endalipt, int *endaljpt );\nstatic int Lfill_kb_impl( double **amino_dynamicmtx,', 1)
wrapper = r'''
static long kbchk_calls = 0, kbchk_cells = 0, kbchk_tracks = 0;
static int **kbchk_ijp = NULL, kbchk_rows = 0;
static void kbchk_report( void ) { fprintf( stderr, "\nKB_CHECK: %ld calls, %ld cells, %ld tracks, all identical\n", kbchk_calls, kbchk_cells, kbchk_tracks ); }
static int Lfill_kb( double **amino_dynamicmtx, double **n_dynamicmtx, double scoreoffset,
                       char *s1, char *s2, int lgth1, int lgth2,
                       double *maxwmpt, int *endalipt, int *endaljpt )
{
	int i, j, r0, r1, ei0, ej0, ei1, ej1, lstop = lgth1 + lgth2 + 1;
	double m0, m1;
	if( kbchk_calls++ == 0 ) atexit( kbchk_report );
	for( i=0; i<kbchk_rows; i++ ) free( kbchk_ijp[i] );
	free( kbchk_ijp );
	kbchk_rows = lgth1 + 1;
	kbchk_ijp = malloc( sizeof( int * ) * kbchk_rows );
	for( i=0; i<kbchk_rows; i++ ) kbchk_ijp[i] = calloc( lgth2 + 1, sizeof( int ) );
	r0 = Lfill_ref( amino_dynamicmtx, n_dynamicmtx, scoreoffset, s1, s2, lgth1, lgth2, kbchk_ijp, lstop, &m0, &ei0, &ej0 );
	r1 = Lfill_kb_impl( amino_dynamicmtx, n_dynamicmtx, scoreoffset, s1, s2, lgth1, lgth2, &m1, &ei1, &ej1 );
	if( r0 != r1 ) { fprintf( stderr, "KB_CHECK: return %d vs %d (lgth %d,%d)\n", r0, r1, lgth1, lgth2 ); abort(); }
	if( r1 )
	{
		if( memcmp( &m0, &m1, sizeof( double ) ) || ei0 != ei1 || ej0 != ej1 )
		{ fprintf( stderr, "KB_CHECK: max %f/%f end %d,%d vs %d,%d\n", m0, m1, ei0, ej0, ei1, ej1 ); abort(); }
		/* every cell: the kind byte, with the markers resolved by the rules of lres_hk() and
		   lres_vmp() applied incrementally over the stored rows (O(1) per cell) */
		{
			int *cvm = malloc( sizeof( int ) * ( lgth2 + 1 ) ), *cvmp = malloc( sizeof( int ) * ( lgth2 + 1 ) );
			for( j=1; j<=lgth2; j++ ) { cvm[j] = ( j == 1 ) ? kb_v0 : KB_RV( 0, j-1 ); cvmp[j] = 0; }
			for( i=1; i<=lgth1; i++ )
			{
				int tb = KB_RV( i-1, 0 ), tk = 0;
				for( j=1; j<=lgth2; j++ )
				{
					int v, code = KB_KIND( i, j-1 );
					if( j >= 2 ) { int q = KB_RV( i-1, j-2 ) - ( j-2 ) * kb_ext; if( q > tb ) { tb = q; tk = j-2; } }
					v = ( code == 0 ) ? 0 : ( code == KB_H ) ? -( j - tk ) : ( code == KB_V ) ? i - cvmp[j] : ( code == KB_STOP ) ? lstop : 0x7fffffff;
					if( v != kbchk_ijp[i][j] )
					{ fprintf( stderr, "KB_CHECK: ijp[%d][%d] ref %d kind %d -> %d (lgth %d,%d)\n", i, j, kbchk_ijp[i][j], code, v, lgth1, lgth2 ); abort(); }
				}
				for( j=1; j<=lgth2; j++ ) { int pp = KB_RV( i-1, j-1 ); if( pp > cvm[j] ) { cvm[j] = pp; cvmp[j] = i-1; } cvm[j] += kb_ext; }
			}
			free( cvm ); free( cvmp );
		}
		*maxwmpt = m1; *endalipt = ei1; *endaljpt = ej1;
		kbchk_cells += (long)lgth1 * lgth2;
	}
	return( r1 );
}
'''
k = s.index('static int kb_isstop( int i, int j ) {')
k = s.index('\n#endif\n', k)
s = s[:k] + '\n' + wrapper + s[k:]
# the reference function goes right before L__align11
k = s.index('double L__align11( double **n_dynamicmtx, double scoreoffset,')
s = s[:k] + ref + '\n\n' + s[k:]

# 3. after the kb traceback, replay Ltracking over the reference ijp and compare
call = '\tLtracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, off1pt, off2pt, endali, endalj, warpis, warpjs, warpbase, kbfill );\n'
assert s.count(call) == 1
check = r'''	if( !kbfill ) Ltracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, off1pt, off2pt, endali, endalj, warpis, warpjs, warpbase, 0 );
	else
	{
		char *a1, *a2; int o1, o2;
		Ltracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, ijp, off1pt, off2pt, endali, endalj, warpis, warpjs, warpbase, 1 );
		a1 = strcpy( malloc( strlen( mseq1[0] ) + 1 ), mseq1[0] ); a2 = strcpy( malloc( strlen( mseq2[0] ) + 1 ), mseq2[0] );
		mseq1[0] = mseq[0]; mseq2[0] = mseq[1];
		Ltracking( currentw, lastverticalw, seq1, seq2, mseq1, mseq2, kbchk_ijp, &o1, &o2, endali, endalj, warpis, warpjs, warpbase, 0 );
		if( strcmp( a1, mseq1[0] ) || strcmp( a2, mseq2[0] ) || o1 != *off1pt || o2 != *off2pt )
		{ fprintf( stderr, "KB_CHECK: tracking differs (offsets %d,%d vs %d,%d)\n", *off1pt, *off2pt, o1, o2 ); abort(); }
		free( a1 ); free( a2 );
		kbchk_tracks++;
	}
'''
s = s.replace(call, check)
open(p, 'w', encoding='latin-1').write(s)
print('patched: reference with %d vector loops off, wrapper, tracking check' % n)
