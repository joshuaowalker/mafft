# opt8 per-call verification harnesses

Each script rewrites a scratch copy of the source (`git archive` it; a `cp -R` of a built tree
leaves stale objects that `make` will not rebuild) so that every call of the rewritten code is
checked against a reference on the same inputs, bit for bit, aborting on the first difference.
Each process reports its counts on stderr at exit (strip `\r` from the progress output; run
without `--quiet`).

| script | checks | reference |
|---|---|---|
| bm_check.py | gapruns, makeresmap, cpmx_calc_new, alignableReagion | the same function with its byte-mask loops off (scalar tail only) |
| fillimp_check.py | fillimp_track's banded walk | the same call with the banded walk off (direct segment walk) |
| scarr_check.py | scarr_fill | the stock statement `scarr[l] += mtx[j][l] * cpmx1[j][i1]`, compiled in the same file |
| kb_check.py | Lfill_kb and the traceback through kb_ijv() (AVX-512 builds) | the scalar prefix-scan fill and Ltracking over its ijp |

The marker fill is checked by `harness/opt6/marks_check.py` (x86) or `harness/arm64/marks_check.py`
(arm64, which also gates the NEON and SVE loops).  The scripts can be combined in one copy.  For
the all-pairs fill on Apple silicon, run with `MAFFT_NOGPU=1` so the CPU fill does every pair.

## Which harnesses apply to which tag

Each harness directory was written against the source of the tag it verified, and patches
that source by matching its text. Against opt8, these were run: the four scripts above,
`harness/opt6/marks_check.py`, `harness/arm64/marks_check.py`, `harness/x86/igsx_check.py` and
`harness/cuda/l11gpu_check.py`. The other `harness/x86`, `harness/arm64` and `harness/cuda`
scripts verify `v7.526-opt7`, and `harness/opt5` verifies `v7.526-opt5`; several no longer
apply to opt8, whose shared code replaced what they patch. Run each against its own tag (with
`git archive`).
