#!/usr/bin/env python3
"""Turn a copy of core/ into a per-call check of scarr_fill (opt8).

Every call of scarr_fill is compared with the stock statement it replaces,
    scarr[l] = 0.0; for( j=0; j<nalphabets; j++ ) scarr[l] += mtx[j][l] * cpmx1[j][i1];
compiled in the same file, so with the compiler's own contraction (the rounding class of the
build).  Every element must match bit for bit, else abort.  Each process reports
"SCARR_CHECK: N calls, all identical" at exit.
Usage: scarr_check.py CORE_DIR  (rewrites mltaln9.c in the copy)
"""
import os, sys

p = os.path.join(sys.argv[1], 'mltaln9.c')
s = open(p, encoding='latin-1').read()
head = 'void scarr_fill( double *scarr, double **mtx, double **cpmx1, int i1 )\n{'
assert s.count(head) == 1
s = s.replace(head, r'''static void scarr_fill_impl( double *scarr, double **mtx, double **cpmx1, int i1 );
static long scchk_calls = 0;
static void scchk_report( void ) { fprintf( stderr, "\nSCARR_CHECK: %ld calls, all identical\n", scchk_calls ); }
void scarr_fill( double *scarr, double **mtx, double **cpmx1, int i1 )
{
	double ref[0x100];
	int j, l;
	if( scchk_calls++ == 0 ) atexit( scchk_report );
	for( l=0; l<nalphabets; l++ )
	{
		ref[l] = 0.0;
		for( j=0; j<nalphabets; j++ ) ref[l] += mtx[j][l] * cpmx1[j][i1];
	}
	scarr_fill_impl( scarr, mtx, cpmx1, i1 );
	if( memcmp( ref, scarr, sizeof( double ) * nalphabets ) ) { fprintf( stderr, "SCARR_CHECK: differs at column %d\n", i1 ); abort(); }
}
static void scarr_fill_impl( double *scarr, double **mtx, double **cpmx1, int i1 )
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched scarr_fill')
