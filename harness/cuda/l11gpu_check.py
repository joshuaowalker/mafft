#!/usr/bin/env python3
"""Turn pairlocalalign.c into a per-call check of the GPU all-pairs local alignments (CUDA or Metal).

Every pair whose result the GPU delivered (status 0: aligned, or 1: no local alignment) is also
run through the CPU L__align11() on the same inputs, and the score (as a double, bit for bit),
both offsets and both aligned strings must match, else the process writes the pair to the log
and aborts.  Pairs the GPU did not take (status -1) go to the CPU as usual and are counted.
At exit one line per process goes to $L11CHK_LOG (if set; mafft --quiet hides stderr) and stderr:
  L11CHK: <gpu pairs checked> gpu-checked <cpu pairs> cpu-only <no-alignment cases> stop <cells> cells
Usage: l11gpu_check.py CORE_DIR_or_pairlocalalign.c   (rewrites in place)
"""
import os, sys

p = sys.argv[1]
if os.path.isdir(p): p = os.path.join(p, 'pairlocalalign.c')
s = open(p, encoding='latin-1').read()

head = 'static double L__align11_pair( int i, int j, char **mseq1, char **mseq2, int alloclen, int *off1pt, int *off2pt )\n{\n'
assert s.count(head) == 1
i0 = s.index(head)
i1 = s.index('\treturn( L__align11( n_dis_consweight_multi, 0.0, mseq1, mseq2, alloclen, off1pt, off2pt ) );\n}\n', i0)
tail_len = len('\treturn( L__align11( n_dis_consweight_multi, 0.0, mseq1, mseq2, alloclen, off1pt, off2pt ) );\n}\n')

new = r'''static long l11chk_gpu = 0, l11chk_cpu = 0, l11chk_stop = 0; static double l11chk_cells = 0.0;
static void l11chk_report( void )
{
	char *fn = getenv( "L11CHK_LOG" );
	FILE *fp = fn ? fopen( fn, "a" ) : NULL;
	if( fp ) { fprintf( fp, "L11CHK: %ld gpu-checked %ld cpu-only %ld stop %.0f cells\n", l11chk_gpu, l11chk_cpu, l11chk_stop, l11chk_cells ); fclose( fp ); }
	fprintf( stderr, "L11CHK: %ld gpu-checked %ld cpu-only %ld stop %.0f cells\n", l11chk_gpu, l11chk_cpu, l11chk_stop, l11chk_cells );
}
static void l11chk_fail( int i, int j, const char *what, double gs, double cs, int g1, int g2, int c1, int c2 )
{
	char *fn = getenv( "L11CHK_LOG" );
	FILE *fp = fn ? fopen( fn, "a" ) : NULL;
	if( fp ) { fprintf( fp, "L11CHK MISMATCH pair %d,%d: %s score %.17g/%.17g off %d,%d / %d,%d\n", i, j, what, gs, cs, g1, g2, c1, c2 ); fclose( fp ); }
	fprintf( stderr, "L11CHK MISMATCH pair %d,%d: %s score %.17g/%.17g off %d,%d / %d,%d\n", i, j, what, gs, cs, g1, g2, c1, c2 );
	abort();
}
static double L__align11_pair( int i, int j, char **mseq1, char **mseq2, int alloclen, int *off1pt, int *off2pt )
{
	static int registered = 0;
	if( !registered ) { registered = 1; atexit( l11chk_report ); }
#if defined(__APPLE__) || defined(MAFFT_CUDA)
	if( gpupairs && i < j && j < gpun )
	{
		l11res *r = gpupairs + gpuidx( i, j );
		if( r->status == 0 || r->status == 1 )
		{
			double gs = ( r->status == 0 ) ? (double)r->maxwm : 0.0, cs;
			int c1, c2, g1 = ( r->status == 0 ) ? r->off1 : 0, g2 = ( r->status == 0 ) ? r->off2 : 0;
			l11chk_cells += (double)strlen( mseq1[0] ) * strlen( mseq2[0] );
			cs = L__align11( n_dis_consweight_multi, 0.0, mseq1, mseq2, alloclen, &c1, &c2 );
			if( memcmp( &gs, &cs, sizeof( double ) ) ) l11chk_fail( i, j, "score", gs, cs, g1, g2, c1, c2 );
			if( g1 != c1 || g2 != c2 ) l11chk_fail( i, j, "offsets", gs, cs, g1, g2, c1, c2 );
			if( strcmp( mseq1[0], r->status == 0 ? r->s1 : "" ) ) l11chk_fail( i, j, "string 1", gs, cs, g1, g2, c1, c2 );
			if( strcmp( mseq2[0], r->status == 0 ? r->s2 : "" ) ) l11chk_fail( i, j, "string 2", gs, cs, g1, g2, c1, c2 );
			l11chk_gpu++;
			if( r->status == 1 ) l11chk_stop++;
			/* now deliver the GPU result exactly as the unpatched code does */
			if( r->status == 0 )
			{
				strcpy( mseq1[0], r->s1 );
				strcpy( mseq2[0], r->s2 );
				*off1pt = r->off1; *off2pt = r->off2;
				free( r->s1 ); free( r->s2 ); r->s1 = r->s2 = NULL; r->status = -1;
				return( (double)r->maxwm );
			}
			strcpy( mseq1[0], "" );
			strcpy( mseq2[0], "" );
			*off1pt = *off2pt = 0;
			r->status = -1;
			return( 0.0 );
		}
	}
#endif
	l11chk_cpu++;
	return( L__align11( n_dis_consweight_multi, 0.0, mseq1, mseq2, alloclen, off1pt, off2pt ) );
}
'''
s = s[:i0] + new + s[i1 + tail_len:]
open(p, 'w', encoding='latin-1').write(s)
print("patched", p)
