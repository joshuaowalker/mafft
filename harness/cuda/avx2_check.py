#!/usr/bin/env python3
"""Per-call check of the AVX2 paths added for x86-64-v3 builds (the box class: AVX2 + FMA, no
AVX-512).  Rewrites a scratch copy of core/ in place:

  partA_row (partSalignmm.c), A_row (Salignmm.c): every call runs the row with its 4-wide vector
    loops off (the scalar loop) on copies, then on, and compares cur, m, mp, ijrow, MI, MPI bit
    for bit (opt5's row_check with the AVX2 loops).  New in the contracting class
    (MAFFT_STOCK_FMA=1); gcc builds have had them since dikarya1.
  scarr_fill (mltaln9.c): every call also computes the original
    scarr[l] = 0.0; for j: scarr[l] += mtx[j][l] * cpmx1[j][i1];
    (compiled in the same file with the same flags, so contracted like the stock statement)
    and compares bit for bit.
  mc_match (partSalignmm.c): the scarr[] it builds is compared with the original loop over every
    letter, s = MULADD( n_dis_consweight_multi[j][l], cpmx1[j][i1], s ).
  gapruns (mltaln9.c): the 32-byte loop off vs on; n and every st/en must match.
  alignableReagion (fftFunctions.c): the 32-byte passes and the pre-marked characters off vs on;
    return value and every segment (start, end, center, score bitwise) must match.

Any difference aborts after writing it to $AVXCHK_LOG (if set) and stderr; at exit every process
appends "AVXCHK <function>: <calls> calls[, <cells> cells], all identical" lines there.
Usage: avx2_check.py COREDIR
"""
import os, sys

core = sys.argv[1]
LOGFN = r'''
static void avxchk_log( const char *msg )
{
	char *fn = getenv( "AVXCHK_LOG" );
	FILE *fp = fn ? fopen( fn, "a" ) : NULL;
	if( fp ) { fputs( msg, fp ); fclose( fp ); }
	fputs( msg, stderr );
}
'''

# --- rows ---------------------------------------------------------------------------------------
for fname, fn, gf in (('partSalignmm.c', 'partA_row', 'gapfreq2'), ('Salignmm.c', 'A_row', 'gf2')):
    p = os.path.join(core, fname)
    s = open(p, encoding='latin-1').read()
    sig_start = s.index('static void %s( int i, int lgth2,' % fn)
    sig_end = s.index(')', s.index('int *MPI', sig_start)) + 1
    sig = s[sig_start:sig_end]
    s = s[:sig_start] + 'static int %s_vec = 1;\n' % fn + sig.replace('static void %s(' % fn, 'static void %s_impl(' % fn) + s[sig_end:]
    inc = '#include "dp.h"\n'
    assert s.count(inc) == 1
    s = s.replace(inc, inc + LOGFN)
    n = 0
    for pat in ('for( ; j+4<=lgth2; j+=4 )', 'for( ; j+3<=lgth2; j+=4 )'):
        n += s.count(pat)
        s = s.replace(pat, pat.replace('for( ; ', 'for( ; %s_vec && ' % fn))
    assert n == 2, (fname, n)
    ext = ', ext' if fn == 'A_row' else ''
    wrapper = r'''
static long %(fn)s_calls = 0, %(fn)s_cells = 0;
static void %(fn)s_report( void ) { char b[200]; sprintf( b, "AVXCHK %(fn)s: %%ld calls, %%ld cells, all identical\n", %(fn)s_calls, %(fn)s_cells ); avxchk_log( b ); }
%(sig)s
{
	size_t nd = (size_t)lgth2 + 2;
	double *c0 = malloc( sizeof( double ) * nd ), *m0 = malloc( sizeof( double ) * nd ), *MI0 = malloc( sizeof( double ) * nd );
	int *mp0 = malloc( sizeof( int ) * nd ), *ij0 = malloc( sizeof( int ) * nd ), *MPI0 = malloc( sizeof( int ) * nd );
	int j;
	if( %(fn)s_calls++ == 0 ) atexit( %(fn)s_report );
	%(fn)s_cells += lgth2;
	memcpy( c0, cur, sizeof( double ) * ( lgth2 + 1 ) );
	memcpy( m0, m, sizeof( double ) * ( lgth2 + 1 ) );
	memcpy( mp0, mp, sizeof( int ) * ( lgth2 + 1 ) );
	%(fn)s_vec = 0;
	%(fn)s_impl( i, lgth2, prev, c0, m0, mp0, ij0, fgcp2, ogcp2, %(gf)s, fgcp1va, ogcp1va, gf1va, gf1vapre, mi0%(ext)s, MI0, MPI0 );
	%(fn)s_vec = 1;
	%(fn)s_impl( i, lgth2, prev, cur, m, mp, ijrow, fgcp2, ogcp2, %(gf)s, fgcp1va, ogcp1va, gf1va, gf1vapre, mi0%(ext)s, MI, MPI );
	for( j=1; j<=lgth2; j++ )
	{
		if( memcmp( c0+j, cur+j, sizeof( double ) ) || memcmp( m0+j, m+j, sizeof( double ) ) || mp0[j] != mp[j] || ij0[j] != ijrow[j]
		 || memcmp( MI0+j, MI+j, sizeof( double ) ) || MPI0[j] != MPI[j] )
		{
			char b[400];
			sprintf( b, "AVXCHK %(fn)s MISMATCH: row %%d col %%d (lgth2 %%d): cur %%.17g/%%.17g m %%.17g/%%.17g mp %%d/%%d ij %%d/%%d\n",
			         i, j, lgth2, c0[j], cur[j], m0[j], m[j], mp0[j], mp[j], ij0[j], ijrow[j] );
			avxchk_log( b );
			abort();
		}
	}
	free( c0 ); free( m0 ); free( MI0 ); free( mp0 ); free( ij0 ); free( MPI0 );
}
''' % dict(fn=fn, sig=sig, gf=gf, ext=ext)
    impl_at = s.index('static void %s_impl(' % fn)
    body_open = s.index('\n{', impl_at)
    depth = 0; k = body_open + 1
    while True:
        if s[k] == '{': depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0: break
        k += 1
    s = s[:k+1] + '\n' + wrapper + s[k+1:]

    if fname == 'partSalignmm.c':
        # mc_match's scarr against the original loop over every letter
        sig2 = 'static void mc_match( double *match, double **cpmx1, int i1, int lgth2 )\n{\n'
        assert s.count(sig2) == 1
        a = s.index(sig2)
        anchor = '\tfor( g=0; g<mc_ngrp; g++ )\n'
        b = s.index(anchor, a)
        check = r'''	{
		static long calls = 0;
		int ll, jj;
		if( calls++ == 0 ) atexit( mcs_report );
		mcs_calls = calls;
		for( ll=0; ll<nalphabets; ll++ )
		{
			double ref = 0.0;
			for( jj=0; jj<nalphabets; jj++ ) ref = MULADD( n_dis_consweight_multi[jj][ll], cpmx1[jj][i1], ref );
			if( memcmp( &ref, scarr + ll, sizeof( double ) ) )
			{ char bb[200]; sprintf( bb, "AVXCHK mc_match scarr MISMATCH: letter %d: %.17g vs %.17g\n", ll, ref, scarr[ll] ); avxchk_log( bb ); abort(); }
		}
	}
'''
        s = s[:b] + check + s[b:]
        s = s[:a] + 'static long mcs_calls = 0;\nstatic void mcs_report( void ) { char b[200]; sprintf( b, "AVXCHK mc_match scarr: %ld calls, all identical\\n", mcs_calls ); avxchk_log( b ); }\n' + s[a:]
    open(p, 'w', encoding='latin-1').write(s)
    print('patched %s: %d vector loops' % (fname, n))

# --- scarr_fill ---------------------------------------------------------------------------------
p = os.path.join(core, 'mltaln9.c')
s = open(p, encoding='latin-1').read()
sig = 'void scarr_fill( double *scarr, double **mtx, double **cpmx1, int i1 )\n{\n'
assert s.count(sig) == 1
s = s.replace(sig, 'static void scarr_fill_impl( double *scarr, double **mtx, double **cpmx1, int i1 )\n{\n')
inc = '#include "mltaln.h"\n'
assert s.startswith(inc)
s = inc + LOGFN + s[len(inc):]
wrapper = r'''
static long sf_calls = 0;
static void sf_report( void ) { char b[200]; sprintf( b, "AVXCHK scarr_fill: %ld calls, all identical\n", sf_calls ); avxchk_log( b ); }
void scarr_fill( double *scarr, double **mtx, double **cpmx1, int i1 )
{
	double ref[0x100];
	int j, l;
	if( sf_calls++ == 0 ) atexit( sf_report );
	for( l=0; l<nalphabets; l++ )
	{
		ref[l] = 0.0;
		for( j=0; j<nalphabets; j++ )
			ref[l] += mtx[j][l] * cpmx1[j][i1];
	}
	scarr_fill_impl( scarr, mtx, cpmx1, i1 );
	for( l=0; l<nalphabets; l++ )
		if( memcmp( ref + l, scarr + l, sizeof( double ) ) )
		{ char b[200]; sprintf( b, "AVXCHK scarr_fill MISMATCH: letter %d: %.17g vs %.17g\n", l, ref[l], scarr[l] ); avxchk_log( b ); abort(); }
}
'''
# append after scarr_fill_impl's body (the end of the HAVE_SCARR_FILL block)
i = s.index('static void scarr_fill_impl(')
k = s.index('\n{', i) + 1
depth = 0
while True:
    if s[k] == '{': depth += 1
    elif s[k] == '}':
        depth -= 1
        if depth == 0: break
    k += 1
s = s[:k+1] + '\n' + wrapper + s[k+1:]
open(p, 'w', encoding='latin-1').write(s)
print('patched mltaln9.c: scarr_fill')

# --- gapruns --------------------------------------------------------------------------------------
p = os.path.join(core, 'mltaln9.c')
s = open(p, encoding='latin-1').read()
sig = 'static int gapruns( char *s, int len, int *st, int *en )\n{'
assert s.count(sig) == 1
s = s.replace(sig, 'static int gr_vec = 1;\nstatic int gapruns_impl( char *s, int len, int *st, int *en )\n{')
pat = 'for( ; i+32<=len; i+=32 )'
assert s.count(pat) == 1
s = s.replace(pat, 'for( ; gr_vec && i+32<=len; i+=32 )')
wrapper = r"""
static long gr_calls = 0;
static void gr_report( void ) { char b[200]; sprintf( b, "AVXCHK gapruns: %ld calls, all identical\n", gr_calls ); avxchk_log( b ); }
static int gapruns( char *s, int len, int *st, int *en )
{
	int *a = malloc( sizeof( int ) * ( len + 2 ) * 2 ), n0, n1, k;
	if( gr_calls++ == 0 ) atexit( gr_report );
	gr_vec = 0; n0 = gapruns_impl( s, len, a, a + len + 2 );
	gr_vec = 1; n1 = gapruns_impl( s, len, st, en );
	if( n0 != n1 ) { char b[200]; sprintf( b, "AVXCHK gapruns MISMATCH: n %d vs %d\n", n0, n1 ); avxchk_log( b ); abort(); }
	for( k=0; k<n1; k++ ) if( a[k] != st[k] || a[len+2+k] != en[k] ) { char b[200]; sprintf( b, "AVXCHK gapruns MISMATCH: run %d\n", k ); avxchk_log( b ); abort(); }
	free( a );
	return( n1 );
}
"""
anchor = 'static int *gapruns_buf( int len )'
assert s.count(anchor) == 1
s = s.replace(anchor, wrapper + '\n' + anchor)
open(p, 'w', encoding='latin-1').write(s)
print('patched mltaln9.c: gapruns')

# --- alignableReagion -----------------------------------------------------------------------------
p = os.path.join(core, 'fftFunctions.c')
s = open(p, encoding='latin-1').read()
sig = 'int alignableReagion( int    clus1, int    clus2, '
assert s.count(sig) == 1
s = s.replace(sig, LOGFN + 'static int ar_vec = 1;\nstatic int alignableReagion_impl( int    clus1, int    clus2, ')
n = 0
for pat, rep in (('for( i=0; i+32<=len; i+=32 )', 'for( i=0; ar_vec && i+32<=len; i+=32 )'),
                 ('for( q=0; common[q]; q++ )', 'for( q=0; ar_vec && common[q]; q++ )')):
    n += s.count(pat)
    s = s.replace(pat, rep)
assert n >= 3, n
wrapper = r"""
static long ar_calls = 0;
static void ar_report( void ) { char b[200]; sprintf( b, "AVXCHK alignableReagion: %ld calls, all identical\n", ar_calls ); avxchk_log( b ); }
int alignableReagion( int clus1, int clus2, char **seq1, char **seq2, double *eff1, double *eff2, Segment *seg )
{
	static Segment *s0 = NULL;
	static int s0n = 0;
	int v0, v1, k, cap = 100000;
	if( clus1 == 0 ) return( alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, seg ) );
	if( ar_calls++ == 0 ) atexit( ar_report );
	if( s0n < cap ) { s0 = realloc( s0, sizeof( Segment ) * cap ); s0n = cap; }
	ar_vec = 0; v0 = alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, s0 );
	ar_vec = 1; v1 = alignableReagion_impl( clus1, clus2, seq1, seq2, eff1, eff2, seg );
	if( v0 != v1 ) { char b[200]; sprintf( b, "AVXCHK alignableReagion MISMATCH: value %d vs %d\n", v0, v1 ); avxchk_log( b ); abort(); }
	for( k=0; k<v1; k++ )
		if( s0[k].start != seg[k].start || s0[k].end != seg[k].end || s0[k].center != seg[k].center || memcmp( &s0[k].score, &seg[k].score, sizeof( double ) ) )
		{ char b[200]; sprintf( b, "AVXCHK alignableReagion MISMATCH: segment %d\n", k ); avxchk_log( b ); abort(); }
	return( v1 );
}
"""
s = s + wrapper
open(p, 'w', encoding='latin-1').write(s)
print('patched fftFunctions.c: alignableReagion, %d gates' % n)
