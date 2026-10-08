#!/usr/bin/env python3
"""Turn partSalignmm.c into a per-call check of the banded importance gather (MAFFT_AVX512X).

Every call of part_imp_match_out_vead_gapmap() first computes the original result,
imp[j] + impmtx[i1][start2+gapmap2[j]] for every j with the scalar loop, into a scratch row,
then runs the banded version on imp; the two rows must match bit for bit, else abort.  It also
counts the calls that took the band path and the cells it skipped.
Each process reports "IMPBAND_CHECK: N calls (B banded), C cells, S skipped, all identical" at exit.
Usage: impband_check.py CORE_DIR_or_partSalignmm.c  (rewrites in place)
"""
import os, sys
p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'partSalignmm.c')
s = open(p, encoding='latin-1').read()
head = 'static void part_imp_match_out_vead_gapmap( double *imp, int i1, int lgth2, int start2, int *gapmap2 )\n{'
assert s.count(head) == 1
s = s.replace(head, '''static long ib_calls = 0, ib_band = 0, ib_cells = 0, ib_skip = 0;
static void ib_report( void ) { fprintf( stderr, "\\nIMPBAND_CHECK: %ld calls (%ld banded), %ld cells, %ld skipped, all identical\\n", ib_calls, ib_band, ib_cells, ib_skip ); }
static void part_imp_match_out_vead_gapmap_impl( double *imp, int i1, int lgth2, int start2, int *gapmap2 );
static void part_imp_match_out_vead_gapmap( double *imp, int i1, int lgth2, int start2, int *gapmap2 )
{
	int j;
	double *ref = malloc( sizeof( double ) * ( lgth2 + 1 ) );
	if( ib_calls++ == 0 ) atexit( ib_report );
	for( j=0; j<lgth2; j++ ) ref[j] = imp[j] + impmtx[i1][start2+gapmap2[j]];
	if( impclean )
	{
		ib_band++;
		for( j=0; j<lgth2; j++ ) if( start2+gapmap2[j] < improwlo[i1] || start2+gapmap2[j] > improwhi[i1] ) ib_skip++;
	}
	part_imp_match_out_vead_gapmap_impl( imp, i1, lgth2, start2, gapmap2 );
	if( memcmp( ref, imp, sizeof( double ) * lgth2 ) )
	{
		for( j=0; j<lgth2 && !memcmp( ref+j, imp+j, sizeof( double ) ); j++ ) ;
		fprintf( stderr, "IMPBAND_CHECK: row %d col %d: %a vs %a\\n", i1, j, ref[j], imp[j] ); abort();
	}
	ib_cells += lgth2;
	free( ref );
}
static void part_imp_match_out_vead_gapmap_impl( double *imp, int i1, int lgth2, int start2, int *gapmap2 )
{''')
open(p, 'w', encoding='latin-1').write(s)
print('patched part_imp_match_out_vead_gapmap')
