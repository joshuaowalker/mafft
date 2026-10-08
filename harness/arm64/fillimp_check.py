#!/usr/bin/env python3
"""Per-call check of fillimp_banded() (Linux arm64) against fillimp_track's original segment walk.

Each call: snapshot impmtx[0..lgth1)[0..lgth2) and rowlo/rowhi, run the banded walk, swap its
result out, restore the snapshot, run the original walk, then compare every cell (bitwise) and
rowlo/rowhi; abort on any difference.  Usage: fillimp_check.py COREDIR (rewrites mltaln9.c)
"""
import os, sys
p = os.path.join(sys.argv[1], 'mltaln9.c')
s = open(p, encoding='latin-1').read()

old = '''	if( fillimp_banded( impmtx, clus1, clus2, lgth1, eff1, eff2, eff1_kozo, eff2_kozo, effijx, localhom, swaplist, orinum1, orinum2, rowlo, rowhi, map1, map2, nres1, nres2 ) )
		goto fillimp_done;
'''
new = '''	double *FC_pre = malloc( sizeof( double ) * (size_t)lgth1 * lgth2 ), *FC_band = malloc( sizeof( double ) * (size_t)lgth1 * lgth2 );
	int *FC_lo = NULL, *FC_hi = NULL, *FC_blo = NULL, *FC_bhi = NULL, FC_did;
	{ int a; for( a=0; a<lgth1; a++ ) memcpy( FC_pre + (size_t)a*lgth2, impmtx[a], sizeof( double ) * lgth2 ); }
	if( rowlo )
	{
		FC_lo = malloc( sizeof( int ) * lgth1 ); FC_hi = malloc( sizeof( int ) * lgth1 );
		FC_blo = malloc( sizeof( int ) * lgth1 ); FC_bhi = malloc( sizeof( int ) * lgth1 );
		memcpy( FC_lo, rowlo, sizeof( int ) * lgth1 ); memcpy( FC_hi, rowhi, sizeof( int ) * lgth1 );
	}
	FC_did = fillimp_banded( impmtx, clus1, clus2, lgth1, eff1, eff2, eff1_kozo, eff2_kozo, effijx, localhom, swaplist, orinum1, orinum2, rowlo, rowhi, map1, map2, nres1, nres2 );
	if( FC_did )
	{
		int a;
		for( a=0; a<lgth1; a++ ) { memcpy( FC_band + (size_t)a*lgth2, impmtx[a], sizeof( double ) * lgth2 ); memcpy( impmtx[a], FC_pre + (size_t)a*lgth2, sizeof( double ) * lgth2 ); }
		if( rowlo ) { memcpy( FC_blo, rowlo, sizeof( int ) * lgth1 ); memcpy( FC_bhi, rowhi, sizeof( int ) * lgth1 ); memcpy( rowlo, FC_lo, sizeof( int ) * lgth1 ); memcpy( rowhi, FC_hi, sizeof( int ) * lgth1 ); }
	}
'''
assert s.count(old) == 1
s = s.replace(old, new)

old2 = '''fillimp_done:
#endif
	for( i=0; i<clus1; i++ ) free( map1[i] );'''
new2 = '''fillimp_done:
#endif
	if( FC_did )
	{
		static long calls = 0, cells = 0;
		int a, c;
		for( a=0; a<lgth1; a++ ) for( c=0; c<lgth2; c++ )
			if( memcmp( &impmtx[a][c], FC_band + (size_t)a*lgth2 + c, sizeof( double ) ) )
			{ fprintf( stderr, "FILLIMP_CHECK: cell %d,%d %.17g vs banded %.17g\\n", a, c, impmtx[a][c], FC_band[(size_t)a*lgth2+c] ); abort(); }
		if( rowlo && ( memcmp( rowlo, FC_blo, sizeof( int ) * lgth1 ) || memcmp( rowhi, FC_bhi, sizeof( int ) * lgth1 ) ) )
		{ fprintf( stderr, "FILLIMP_CHECK: rowlo/rowhi differ\\n" ); abort(); }
		calls++; cells += (long)lgth1 * lgth2;
		if( ( calls & ( calls - 1 ) ) == 0 ) fprintf( stderr, "FILLIMP_CHECK: %ld calls, %ld cells compared, all identical\\n", calls, cells );
	}
	else fprintf( stderr, "FILLIMP_CHECK: fallback (original walk only)\\n" );
	free( FC_pre ); free( FC_band ); free( FC_lo ); free( FC_hi ); free( FC_blo ); free( FC_bhi );
	for( i=0; i<clus1; i++ ) free( map1[i] );'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, 'w', encoding='latin-1').write(s)
print('patched fillimp_track')
