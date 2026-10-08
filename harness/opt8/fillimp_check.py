#!/usr/bin/env python3
"""Turn a copy of core/ into a per-call check of fillimp_track's banded walk (opt8).

Every call that tracks the written ranges (rowlo != NULL, the partA__align path) runs twice on the
same inputs: first with the banded walk off, so the direct segment walk fills the matrix, then
as built (FILLIMP_BAND rows at a time).  The importance matrix (rows 0..lgth1-1, columns
0..lgth2-1) and rowlo/rowhi are restored in between, and the two results must match bit for bit,
else abort.  Each process reports "FILLIMP_CHECK: N calls, M cells, all identical" at exit.
Usage: fillimp_check.py CORE_DIR  (rewrites mltaln9.c in the copy)
"""
import os, sys

p = os.path.join(sys.argv[1], 'mltaln9.c')
s = open(p, encoding='latin-1').read()
gate = '\tif( fillimp_banded( impmtx,'
assert s.count(gate) == 1
s = s.replace(gate, '\tif( fillimp_band_on && fillimp_banded( impmtx,')
head = 'void fillimp_track( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi )\n{'
assert s.count(head) == 1
s = s.replace(head, r'''static int fillimp_band_on = 1;
static void fillimp_track_impl( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi );
static long fichk_calls = 0, fichk_cells = 0;
static void fichk_report( void ) { fprintf( stderr, "\nFILLIMP_CHECK: %ld calls, %ld cells, all identical\n", fichk_calls, fichk_cells ); }
void fillimp_track( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi )
{
	double *save, *ref;
	int *slo, *shi, *rlo, *rhi, i;
	size_t n = (size_t)lgth1 * lgth2;
	if( !rowlo || lgth1 < 1 || lgth2 < 1 )
	{
		fillimp_track_impl( impmtx, imp, clus1, clus2, lgth1, lgth2, seq1, seq2, eff1, eff2, eff1_kozo, eff2_kozo, localhom, swaplist, forscore, orinum1, orinum2, rowlo, rowhi );
		return;
	}
	if( fichk_calls++ == 0 ) atexit( fichk_report );
	save = malloc( sizeof( double ) * n ); ref = malloc( sizeof( double ) * n );
	slo = malloc( sizeof( int ) * lgth1 ); shi = malloc( sizeof( int ) * lgth1 ); rlo = malloc( sizeof( int ) * lgth1 ); rhi = malloc( sizeof( int ) * lgth1 );
	for( i=0; i<lgth1; i++ ) memcpy( save + (size_t)i * lgth2, impmtx[i], sizeof( double ) * lgth2 );
	memcpy( slo, rowlo, sizeof( int ) * lgth1 ); memcpy( shi, rowhi, sizeof( int ) * lgth1 );
	fillimp_band_on = 0;
	fillimp_track_impl( impmtx, imp, clus1, clus2, lgth1, lgth2, seq1, seq2, eff1, eff2, eff1_kozo, eff2_kozo, localhom, swaplist, forscore, orinum1, orinum2, rowlo, rowhi );
	for( i=0; i<lgth1; i++ ) { memcpy( ref + (size_t)i * lgth2, impmtx[i], sizeof( double ) * lgth2 ); memcpy( impmtx[i], save + (size_t)i * lgth2, sizeof( double ) * lgth2 ); }
	memcpy( rlo, rowlo, sizeof( int ) * lgth1 ); memcpy( rhi, rowhi, sizeof( int ) * lgth1 );
	memcpy( rowlo, slo, sizeof( int ) * lgth1 ); memcpy( rowhi, shi, sizeof( int ) * lgth1 );
	fillimp_band_on = 1;
	fillimp_track_impl( impmtx, imp, clus1, clus2, lgth1, lgth2, seq1, seq2, eff1, eff2, eff1_kozo, eff2_kozo, localhom, swaplist, forscore, orinum1, orinum2, rowlo, rowhi );
	for( i=0; i<lgth1; i++ )
	{
		if( memcmp( ref + (size_t)i * lgth2, impmtx[i], sizeof( double ) * lgth2 ) ) { fprintf( stderr, "FILLIMP_CHECK: impmtx row %d differs (lgth %d,%d)\n", i, lgth1, lgth2 ); abort(); }
		if( rlo[i] != rowlo[i] || rhi[i] != rowhi[i] ) { fprintf( stderr, "FILLIMP_CHECK: row range %d differs\n", i ); abort(); }
	}
	fichk_cells += (long)n;
	free( save ); free( ref ); free( slo ); free( shi ); free( rlo ); free( rhi );
}
static void fillimp_track_impl( double **impmtx, double *imp, int clus1, int clus2, int lgth1, int lgth2, char **seq1, char **seq2, double *eff1, double *eff2, double *eff1_kozo, double *eff2_kozo, LocalHom ***localhom, char *swaplist, int forscore, int *orinum1, int *orinum2, int *rowlo, int *rowhi )
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched fillimp_track')
