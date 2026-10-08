#!/usr/bin/env python3
"""Turn tddis.c into a per-call check of the masked cpmx_calc_new (MAFFT_AVX512X).

Every call also runs the original column loop into a scratch profile and compares all
nalphabets x lgth cells bit for bit, aborting on any difference.
Each process reports "CPMX_CHECK: N calls, M cells, all identical" at exit.
Usage: cpmx_check.py CORE_DIR_or_tddis.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'tddis.c')
s = open(p, encoding='latin-1').read()
head = 'void cpmx_calc_new( char **seq, double **cpmx, double *eff, int lgth, int clus ) // summ eff must be 1.0\n{'
assert s.count(head) == 1
s = s.replace(head, '''static long cx_calls = 0, cx_cells = 0;
static void cx_report( void ) { fprintf( stderr, "\\nCPMX_CHECK: %ld calls, %ld cells, all identical\\n", cx_calls, cx_cells ); }
static void cpmx_calc_new_impl( char **seq, double **cpmx, double *eff, int lgth, int clus );
void cpmx_calc_new( char **seq, double **cpmx, double *eff, int lgth, int clus )
{
	int i, j, k;
	double *ref = calloc( (size_t)nalphabets * lgth + 1, sizeof( double ) );
	if( cx_calls++ == 0 ) atexit( cx_report );
	for( k=0; k<clus; k++ ) for( j=0; j<lgth; j++ )
	{
		int l = (unsigned char)amino_n[(unsigned char)seq[k][j]];
		if( l < nalphabets ) ref[(size_t)l*lgth+j] += eff[k];
		else { fprintf( stderr, "CPMX_CHECK: letter outside the profile\\n" ); abort(); }
	}
	cpmx_calc_new_impl( seq, cpmx, eff, lgth, clus );
	for( i=0; i<nalphabets; i++ ) for( j=0; j<lgth; j++ )
		if( memcmp( &cpmx[i][j], ref + (size_t)i*lgth + j, sizeof( double ) ) )
		{ fprintf( stderr, "CPMX_CHECK: cell %d,%d %a vs %a\\n", i, j, cpmx[i][j], ref[(size_t)i*lgth+j] ); abort(); }
	cx_cells += (long)nalphabets * lgth;
	free( ref );
}
static void cpmx_calc_new_impl( char **seq, double **cpmx, double *eff, int lgth, int clus ) // summ eff must be 1.0
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched cpmx_calc_new')
