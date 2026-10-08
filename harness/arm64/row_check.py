#!/usr/bin/env python3
"""Per-call check of the vector DP rows (partA_row in partSalignmm.c, A_row in Salignmm.c; any vector build).

Each call runs the row with the vector loop off on copies of cur/m/mp/MI/MPI/ijrow, then with it
on, and compares every output (cur[1..lgth2], m, mp, ijrow, MI, MPI) bit for bit; abort on any
difference.  Usage: row_check.py COREDIR  (rewrites partSalignmm.c and Salignmm.c in place)
"""
import sys, os

core = sys.argv[1]
for fname, fn, gf in (('partSalignmm.c', 'partA_row', 'gapfreq2'), ('Salignmm.c', 'A_row', 'gf2')):
    p = os.path.join(core, fname)
    s = open(p, encoding='latin-1').read()
    sig_start = s.index('static void %s( int i, int lgth2,' % fn)
    sig_end = s.index(')', s.index('int *MPI', sig_start)) + 1
    sig = s[sig_start:sig_end]
    s = s[:sig_start] + 'static int %s_vec = 1;\n' % fn + sig.replace('static void %s(' % fn, 'static void %s_impl(' % fn) + s[sig_end:]
    n = 0
    for pat in ('for( ; j+1<=lgth2; j+=2 )', 'for( ; j+7<=lgth2; j+=8 )', 'for( ; j+8<=lgth2; j+=8 )', 'for( ; j+2<=lgth2; j+=2 )', 'for( ; j+3<=lgth2; j+=4 )'):
        n += s.count(pat)
        s = s.replace(pat, pat.replace('for( ; ', 'for( ; %s_vec && ' % fn))
    assert n >= 1, (fname, n)
    ext = ', ext' if fn == 'A_row' else ''
    extp = ' double ext,' if fn == 'A_row' else ''
    wrapper = r'''
static long %(fn)s_calls = 0, %(fn)s_cells = 0;
static void %(fn)s_report( void ) { fprintf( stderr, "ROW_CHECK %(fn)s: %%ld calls, %%ld cells, all identical\n", %(fn)s_calls, %(fn)s_cells ); }
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
			fprintf( stderr, "ROW_CHECK %(fn)s: row %%d col %%d (lgth2 %%d): cur %%.17g/%%.17g m %%.17g/%%.17g mp %%d/%%d ij %%d/%%d\n",
			         i, j, lgth2, c0[j], cur[j], m0[j], m[j], mp0[j], mp[j], ij0[j], ijrow[j] );
			abort();
		}
	}
	free( c0 ); free( m0 ); free( MI0 ); free( mp0 ); free( ij0 ); free( MPI0 );
}
''' % dict(fn=fn, sig=sig, gf=gf, ext=ext)
    # put the wrapper right after the impl's body: find the next top-level function after the impl
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
    open(p, 'w', encoding='latin-1').write(s)
    print('patched %s: %d vector loops' % (fname, n))
