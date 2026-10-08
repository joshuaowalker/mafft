# Linux arm64 (Graviton) per-call verification harnesses

Each script rewrites one or more files in a scratch copy of `core/` so that every call of a
rewritten function runs the new (vector) path and the original path on the same inputs, compares
all outputs bit for bit, and aborts on the first difference. Counts are reported on stderr
(at exit or at power-of-two call counts; strip `\r` from MAFFT's progress output to read them).

| script | function | file |
|---|---|---|
| marks_check.py | marker-based `Lfill_int`: NEON loop, SVE loop, the scalar head cells, the row-maximum search (also AVX2/AVX-512) | Lalign11.c |
| fillimp_check.py | `fillimp_banded` against the original segment walk (whole `impmtx` and `rowlo`/`rowhi`) | mltaln9.c |
| igs_check.py | `igs_pairscores` with the NEON `igs_prep` and table-lookup column sums off vs on | mltaln9.c |
| misc_check.py | NEON `makeresmap`; `part_imp_match_out_vead_gapmap` (contiguous-run loads) | mltaln9.c, partSalignmm.c |
| row_check.py | `partA_row`, `A_row` (the non-fusing NEON rows of gcc builds; any vector build) | partSalignmm.c, Salignmm.c |
| mcmatch_check.py | `mc_match` (NEON score table and column groups; also AVX-512) | partSalignmm.c |
| areg_check.py | `alignableReagion` presence and profile passes (NEON; also AVX-512BW) | fftFunctions.c |

Usage: copy the source tree, run `python3 SCRIPT.py <copy>/core` for each check wanted (they can
be combined), build as usual (`make CC=clang CFLAGS="-O3 -mcpu=neoverse-n2"`, `-mcpu=native`, or
`CC=gcc`), then run MAFFT without `--quiet` in several modes and compare its output with stock.
To exercise the SVE fill on a 128-bit SVE machine, add `-DLSVE_MINLANES=4`; to force the NEON
fill on a 256-bit one, `-DLSVE_MINLANES=99`.

The scripts patch the sources textually and may not apply to later revisions.
