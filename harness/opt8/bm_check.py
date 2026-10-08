#!/usr/bin/env python3
"""Turn a copy of core/ into a per-call check of the byte-mask passes (opt8).

gapruns, makeresmap (mltaln9.c), cpmx_calc_new (tddis.c) and alignableReagion (fftFunctions.c)
each run their byte-mask loops (mafft_bm_eq, MAFFT_BM_BYTES at a time) and leave the rest to a
scalar tail.  Every call is run twice on the same inputs: with the byte-mask loops off, so the
scalar tail does all of it, and then as built.  The outputs must match bit for bit (the runs,
the residue map, every cpmx cell, alignableReagion's return value and segments), else abort.
Each process reports "BM_CHECK: <function> N calls, all identical" on stderr at exit.
Usage: bm_check.py CORE_DIR  (rewrites the copy in place; needs a build with MAFFT_BM_BYTES)
"""
import os, re, sys

core = sys.argv[1]
def load(f): return open(os.path.join(core, f), encoding='latin-1').read()
def save(f, s): open(os.path.join(core, f), 'w', encoding='latin-1').write(s)

GATE = r'(for\( (?:[a-z]+=0)?; )([a-z]+\+MAFFT_BM_BYTES<=)'
def gate(s, n):
    s, k = re.subn(GATE, r'\1mafft_bm_on && \2', s)
    assert k == n, k
    return s

REPORT = r'''
static void bmchk_report( const char *f, long *calls ) { if( *calls ) fprintf( stderr, "BM_CHECK: %s %ld calls, all identical\n", f, *calls ); }
'''

# mltaln9.c: gapruns and makeresmap
s = load('mltaln9.c')
s = s.replace('#include "mltaln.h"\n', '#include "mltaln.h"\nint mafft_bm_on = 1;\n' + REPORT, 1)
s = gate(s, 2)
a = 'static int gapruns( char *s, int len, int *st, int *en )\n{'
assert s.count(a) == 1
s = s.replace(a, '''static int gapruns_impl( char *s, int len, int *st, int *en );
static long gr_calls = 0;
static void gr_report( void ) { bmchk_report( "gapruns", &gr_calls ); }
static int gapruns( char *s, int len, int *st, int *en )
{
	int *st0 = malloc( sizeof( int ) * ( len + 2 ) ), *en0 = malloc( sizeof( int ) * ( len + 2 ) ), n0, n1, r;
	if( gr_calls++ == 0 ) atexit( gr_report );
	mafft_bm_on = 0; n0 = gapruns_impl( s, len, st0, en0 );
	mafft_bm_on = 1; n1 = gapruns_impl( s, len, st, en );
	if( n0 != n1 ) { fprintf( stderr, "BM_CHECK: gapruns n %d vs %d\\n", n0, n1 ); abort(); }
	for( r=0; r<n1; r++ ) if( st0[r] != st[r] || en0[r] != en[r] ) { fprintf( stderr, "BM_CHECK: gapruns run %d\\n", r ); abort(); }
	free( st0 ); free( en0 );
	return( n1 );
}
static int gapruns_impl( char *s, int len, int *st, int *en )
{''')
a = 'static int makeresmap( char *seq, int *map )\n{'
assert s.count(a) == 1
s = s.replace(a, '''static int makeresmap_impl( char *seq, int *map );
static long rm_calls = 0;
static void rm_report( void ) { bmchk_report( "makeresmap", &rm_calls ); }
static int makeresmap( char *seq, int *map )
{
	int len = (int)strlen( seq ), *m0 = malloc( sizeof( int ) * ( len + 1 ) ), n0, n1;
	if( rm_calls++ == 0 ) atexit( rm_report );
	mafft_bm_on = 0; n0 = makeresmap_impl( seq, m0 );
	mafft_bm_on = 1; n1 = makeresmap_impl( seq, map );
	if( n0 != n1 || memcmp( m0, map, sizeof( int ) * n1 ) ) { fprintf( stderr, "BM_CHECK: makeresmap differs\\n" ); abort(); }
	free( m0 );
	return( n1 );
}
static int makeresmap_impl( char *seq, int *map )
{''')
save('mltaln9.c', s)

# tddis.c: cpmx_calc_new
s = load('tddis.c')
s = s.replace('#include "mltaln.h"\n', '#include "mltaln.h"\nextern int mafft_bm_on;\n' + REPORT, 1)
s = gate(s, 1)
a = 'void cpmx_calc_new( char **seq, double **cpmx, double *eff, int lgth, int clus ) // summ eff must be 1.0\n{'
assert s.count(a) == 1
s = s.replace(a, '''static void cpmx_calc_new_impl( char **seq, double **cpmx, double *eff, int lgth, int clus );
static long cp_calls = 0;
static void cp_report( void ) { bmchk_report( "cpmx_calc_new", &cp_calls ); }
void cpmx_calc_new( char **seq, double **cpmx, double *eff, int lgth, int clus )
{
	double **c0 = AllocateDoubleMtx( nalphabets, lgth + 1 );
	int l;
	if( cp_calls++ == 0 ) atexit( cp_report );
	mafft_bm_on = 0; cpmx_calc_new_impl( seq, c0, eff, lgth, clus );
	mafft_bm_on = 1; cpmx_calc_new_impl( seq, cpmx, eff, lgth, clus );
	for( l=0; l<nalphabets; l++ ) if( memcmp( c0[l], cpmx[l], sizeof( double ) * lgth ) ) { fprintf( stderr, "BM_CHECK: cpmx_calc_new row %d\\n", l ); abort(); }
	FreeDoubleMtx( c0 );
}
static void cpmx_calc_new_impl( char **seq, double **cpmx, double *eff, int lgth, int clus ) // summ eff must be 1.0
{''')
save('tddis.c', s)

# fftFunctions.c: alignableReagion
s = load('fftFunctions.c')
s = s.replace('#include "mltaln.h"\n', '#include "mltaln.h"\nextern int mafft_bm_on;\n' + REPORT, 1)
s = gate(s, 2)
m = re.search(r'int alignableReagion\(\s*int\s+clus1,\s*int\s+clus2,[^{]*Segment \*seg \)\n\{', s)
assert m
a = m.group(0)
assert s.count(a) == 1
s = s.replace(a, '''static int alignableReagion_impl( int clus1, int clus2, char **seq1, char **seq2, double *eff1, double *eff2, Segment *seg );
static long ar_calls = 0;
static void ar_report( void ) { bmchk_report( "alignableReagion", &ar_calls ); }
int alignableReagion( int clus1, int clus2, char **seq1, char **seq2, double *eff1, double *eff2, Segment *seg )
{
	static Segment *s0 = NULL;
	int v0, v1, k;
	if( clus1 == 0 ) return( alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, seg ) );
	if( !s0 ) s0 = malloc( sizeof( Segment ) * MAXSEG );
	if( ar_calls++ == 0 ) atexit( ar_report );
	mafft_bm_on = 0; v0 = alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, s0 );
	mafft_bm_on = 1; v1 = alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, seg );
	if( v0 != v1 ) { fprintf( stderr, "BM_CHECK: alignableReagion %d vs %d segments\\n", v0, v1 ); abort(); }
	for( k=0; k<v1; k++ )
		if( s0[k].start != seg[k].start || s0[k].end != seg[k].end || s0[k].center != seg[k].center || memcmp( &s0[k].score, &seg[k].score, sizeof( double ) ) )
		{ fprintf( stderr, "BM_CHECK: alignableReagion segment %d\\n", k ); abort(); }
	return( v1 );
}
static int alignableReagion_impl( int    clus1, int    clus2,
					   char  **seq1, char  **seq2,
					   double *eff1, double *eff2,
					   Segment *seg )
{''')
save('fftFunctions.c', s)
print('patched gapruns, makeresmap, cpmx_calc_new, alignableReagion')
