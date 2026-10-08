#!/usr/bin/env python3
"""Turn mltaln9.c into a per-call check of fillimp_track's contiguous walk (MAFFT_AVX512X).

Every call with row tracking (the refinement's call from partSalignmm.c) runs twice on the same
inputs: first with the contiguous-matrix walk switched off (the opt5 walk through the row
pointers), then, from the same starting matrix and row ranges, with it on.  Every cell of
impmtx[0..lgth1)[0..lgth2) and every rowlo/rowhi entry must match bit for bit, else abort.
Each process reports "FILLIMP_CHECK: N calls (C contiguous), M cells, all identical" at exit.
Usage: fillimp_check.py CORE_DIR_or_mltaln9.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'mltaln9.c')
s = open(p, encoding='latin-1').read()
head = 'void fillimp_track( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi )\n{'
assert s.count(head) == 1
s = s.replace(head, '''static int fichk_noc = 0; static long fichk_calls = 0, fichk_contig = 0, fichk_cells = 0;
static void fichk_report( void ) { fprintf( stderr, "\\nFILLIMP_CHECK: %ld calls (%ld contiguous), %ld cells, all identical\\n", fichk_calls, fichk_contig, fichk_cells ); }
static void fillimp_track_impl( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi, int *contig );
void fillimp_track( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi )
{
	double *pre, *ref; int *lo0, *hi0, *lo1, *hi1, a, c, contig = 0;
	size_t n = (size_t)lgth1 * lgth2;
	if( !rowlo ) { fillimp_track_impl( impmtx, imp, clus1, clus2, lgth1, lgth2, seq1, seq2, eff1, eff2, eff1_kozo, eff2_kozo, localhom, swaplist, forscore, orinum1, orinum2, rowlo, rowhi, &contig ); return; }
	if( fichk_calls++ == 0 ) atexit( fichk_report );
	pre = malloc( sizeof( double ) * n ); ref = malloc( sizeof( double ) * n );
	lo0 = malloc( sizeof( int ) * lgth1 ); hi0 = malloc( sizeof( int ) * lgth1 ); lo1 = malloc( sizeof( int ) * lgth1 ); hi1 = malloc( sizeof( int ) * lgth1 );
	for( a=0; a<lgth1; a++ ) memcpy( pre + (size_t)a * lgth2, impmtx[a], sizeof( double ) * lgth2 );
	memcpy( lo0, rowlo, sizeof( int ) * lgth1 ); memcpy( hi0, rowhi, sizeof( int ) * lgth1 );
	fichk_noc = 1;
	fillimp_track_impl( impmtx, imp, clus1, clus2, lgth1, lgth2, seq1, seq2, eff1, eff2, eff1_kozo, eff2_kozo, localhom, swaplist, forscore, orinum1, orinum2, rowlo, rowhi, &contig );
	fichk_noc = 0;
	for( a=0; a<lgth1; a++ ) { memcpy( ref + (size_t)a * lgth2, impmtx[a], sizeof( double ) * lgth2 ); memcpy( impmtx[a], pre + (size_t)a * lgth2, sizeof( double ) * lgth2 ); }
	memcpy( lo1, rowlo, sizeof( int ) * lgth1 ); memcpy( hi1, rowhi, sizeof( int ) * lgth1 );
	memcpy( rowlo, lo0, sizeof( int ) * lgth1 ); memcpy( rowhi, hi0, sizeof( int ) * lgth1 );
	fillimp_track_impl( impmtx, imp, clus1, clus2, lgth1, lgth2, seq1, seq2, eff1, eff2, eff1_kozo, eff2_kozo, localhom, swaplist, forscore, orinum1, orinum2, rowlo, rowhi, &contig );
	for( a=0; a<lgth1; a++ ) for( c=0; c<lgth2; c++ )
		if( memcmp( &impmtx[a][c], ref + (size_t)a * lgth2 + c, sizeof( double ) ) )
		{ fprintf( stderr, "FILLIMP_CHECK: cell %d,%d %a vs %a\\n", a, c, impmtx[a][c], ref[(size_t)a * lgth2 + c] ); abort(); }
	if( memcmp( rowlo, lo1, sizeof( int ) * lgth1 ) || memcmp( rowhi, hi1, sizeof( int ) * lgth1 ) ) { fprintf( stderr, "FILLIMP_CHECK: rowlo/rowhi differ\\n" ); abort(); }
	fichk_contig += contig; fichk_cells += n;
	free( pre ); free( ref ); free( lo0 ); free( hi0 ); free( lo1 ); free( hi1 );
}
static void fillimp_track_impl( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi, int *contig )
{''')
old = '''			for( i=2; i<lgth1 && fi_stride; i++ ) if( impmtx[i] != fi_base + (size_t)i * fi_stride ) fi_stride = 0;
		}
'''
assert s.count(old) == 1
s = s.replace(old, old + '''		if( fichk_noc ) fi_stride = 0;
		*contig = ( fi_stride != 0 );
''')
open(p, 'w', encoding='latin-1').write(s)
print('patched fillimp_track')
