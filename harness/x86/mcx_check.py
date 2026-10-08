#!/usr/bin/env python3
"""Turn partSalignmm.c into a per-call check of mcx_apply (block-of-8 match_calc, MAFFT_AVX512X).

Every call of mc_match() computes the row with mcx_apply() into a scratch row and with the
grouped opt5 code (the scatter version) into match[]; all lgth2 values must match bit for bit.
Each process reports "MCX_CHECK: N calls, M cells, all identical" at exit.
Usage: mcx_check.py CORE_DIR_or_partSalignmm.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'partSalignmm.c')
s = open(p, encoding='latin-1').read()
old = '\tif( mcx_ok ) { mcx_apply( match, scarr, lgth2 ); return; }\n'
assert s.count(old) == 1
s = s.replace(old, '''	static long mx_calls = 0, mx_cells = 0; static double *mx_row = NULL; static int mx_cap = 0;
	if( mx_calls++ == 0 ) { void mx_report( void ); atexit( mx_report ); }
	if( mx_cap < lgth2 + 8 ) { free( mx_row ); mx_cap = lgth2 + 8; mx_row = malloc( sizeof( double ) * mx_cap ); }
	if( mcx_ok ) mcx_apply( mx_row, scarr, lgth2 );
''')
end = '\n}\n\nstatic void match_calc( double *match, double **cpmx1, double **cpmx2, int i1, int lgth2, double **doublework, int **intwork, int initialize )'
assert s.count(end) == 1
s = s.replace(end, '''
	if( mcx_ok )
	{
		for( j=0; j<lgth2; j++ ) if( memcmp( mx_row + j, match + j, sizeof( double ) ) )
		{ fprintf( stderr, "MCX_CHECK: row %d column %d %a vs %a\\n", i1, j, mx_row[j], match[j] ); abort(); }
		mx_cells += lgth2; mx_tot += lgth2; mx_n++;
	}
}
long mx_tot = 0, mx_n = 0;
void mx_report( void ) { fprintf( stderr, "\\nMCX_CHECK: %ld calls, %ld cells, all identical\\n", mx_n, mx_tot ); }

static void match_calc( double *match, double **cpmx1, double **cpmx2, int i1, int lgth2, double **doublework, int **intwork, int initialize )''')
# forward declarations for the counters used inside mc_match
s = s.replace('static void mc_match( double *match, double **cpmx1, int i1, int lgth2 )\n{', 'extern long mx_tot, mx_n;\nstatic void mc_match( double *match, double **cpmx1, int i1, int lgth2 )\n{', 1)
open(p, 'w', encoding='latin-1').write(s)
print('patched mc_match')
