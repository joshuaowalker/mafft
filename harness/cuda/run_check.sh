#!/bin/bash
# run_check.sh PREFIX OUTDIR [JOBS] : run a build patched by l11gpu_check.py over the check cases.
#   L-INS-i (--localpair --maxiterate 1000 --thread 1) on every *.fasta in $LINSI_DIR under three
#   gap settings (default, --lexp 0, --lop -3.0 --lexp -0.5), on the protein test/sample
#   ($SAMPLE), and the placement cases in $PLACE_DIR (case*/{mode,q.fasta,shard.aln.fasta}).
# Each process appends its L11CHK line (pairs checked against the CPU L__align11, pairs left to
# the CPU) to OUTDIR/l11chk.log; a mismatch aborts that run and is logged there too.  Outputs go
# to OUTDIR/<case>.aln for comparison with a reference build.
set -u
prefix=$1; out=$2; jobs=${3:-2}
LINSI_DIR=${LINSI_DIR:-/work/mafft-canary-opt5/linsi}
PLACE_DIR=${PLACE_DIR:-/work/mafft-canary-opt5/place}
SAMPLE=${SAMPLE:-/work/src/cuda/test/sample}
mkdir -p $out; rm -f $out/l11chk.log
export L11CHK_LOG=$out/l11chk.log
{
  for f in $LINSI_DIR/*.fasta; do
    b=$(basename $f .fasta)
    echo "$b.def|--localpair --maxiterate 1000|$f"
    echo "$b.lexp0|--localpair --maxiterate 1000 --lexp 0|$f"
    echo "$b.lop3|--localpair --maxiterate 1000 --lop -3.0 --lexp -0.5|$f"
  done
  for o in def "lexp0|--lexp 0" "lop3|--lop -3.0 --lexp -0.5"; do
    n=${o%%|*}; a=""; [ "$o" != "$n" ] && a=${o#*|}
    echo "sample.$n|--localpair --maxiterate 1000 $a|$SAMPLE"
  done
} > $out/cases.txt
run1() {
  IFS='|' read -r name opts f <<< "$1"
  $prefix/bin/mafft $opts --quiet --thread 1 $f > $out/$name.aln 2>/dev/null || echo "FAILED $name" >> $out/l11chk.log
  echo "$name $(grep -c . $out/$name.aln)"
}
export -f run1; export prefix out
xargs -P $jobs -I{} bash -c 'run1 "$@"' _ {} < $out/cases.txt
for d in $PLACE_DIR/case*/; do
  n=$(basename $d)
  ( cd $d && $prefix/bin/mafft $(cat mode) q.fasta --keeplength --quiet --thread 1 shard.aln.fasta > $out/place.$n.aln 2>/dev/null ) || echo "FAILED place.$n" >> $out/l11chk.log
done
echo "== $(grep -c '^L11CHK:' $out/l11chk.log) processes; mismatches: $(grep -c MISMATCH $out/l11chk.log); failures: $(grep -c FAILED $out/l11chk.log)"
awk '/^L11CHK:/ { g += $2; c += $4; s += $6; x += $8 } END { printf "gpu-checked pairs %d, cpu-only pairs %d, of which no-alignment %d, cells %.4g\n", g, c, s, x }' $out/l11chk.log
