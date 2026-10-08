# x86 (Zen 5 / Granite Rapids) per-call verification harnesses

The code these check is compiled only when `MAFFT_AVX512X` is defined (`core/mltaln.h`): AVX-512
F/BW/VL plus VPOPCNTDQ and VBMI2, e.g. `-march=x86-64-v4 -mavx512vpopcntdq -mavx512vbmi2` or a CPU
such as `znver4`, `znver5`, `icelake-server`, `sapphirerapids`, `graniterapids`. A plain
`-march=x86-64-v4` build compiles exactly the opt6 objects.

Each script rewrites one source file in a scratch copy of `core/` so that every call of the new
code also runs the original computation on the same inputs, compares all outputs bit for bit and
aborts on the first difference. Each process prints a `NAME_CHECK: ... all identical` line on
stderr at exit (strip `\r` from dvtditr's progress output to read it). The scripts combine.

| script | new code | file | compared against |
|---|---|---|---|
| kb_check.py | `Lfill_kb` (striped integer local-alignment fill, one kind byte per cell), `Ltracking_kb` | Lalign11.c | opt5's offset-storing `Lfill_int`, scalar loops only: every cell's offset, maxwm, end point; then the original `Ltracking` over that ijp vs `Ltracking_kb` (strings, offsets) |
| impband_check.py | importance gather restricted to the row's written range | partSalignmm.c | the full gather, every element |
| fillimp_check.py | `fillimp_track` walk through one base pointer (contiguous, huge-page matrix) | mltaln9.c | the opt5 walk from the same starting matrix: every cell and rowlo/rowhi |
| resmap_check.py | `makeresmap_avx512` | mltaln9.c | `makeresmap` |
| igsx_check.py | `igs_pairscores_x` (bit masks, popcount column sums) | mltaln9.c | the original `igs_pairscores` path, every pair score |
| colmask_check.py | `cpmx_colmask` (nonzero letters of each profile column) | mltaln9.c | the scalar `if( cpmx[l][j] )` test |
| mcx_check.py | `mcx_apply` (match_calc by blocks of 8 columns) | partSalignmm.c | the grouped `mc_match` (opt5), every cell |
| cpmx_check.py | masked `cpmx_calc_new` | tddis.c | the column loop, every cell |
| areg_check.py | bin-major profiles in `alignableReagion` | fftFunctions.c | the column-major profiles and site scores, every site |
| atexp_check.py | `at_expand` (traceback output rows by VBMI2 expand) | partSalignmm.c | the per-column loop, every row |

The partA_row prefix-scan change is covered by `harness/opt5/checks/row_check.py`, which applies
unchanged (it compares the vector row with the scalar row).

Usage: copy the source tree, run `python3 SCRIPT.py <copy>/core` for the checks wanted, build with
the flags above (and `-DMAFFT_STOCK_FMA=1` for the clang+FMA class), then run MAFFT without
`--quiet`, e.g. `MAFFT_BINARIES=<copy>/core mafft --localpair --maxiterate 1000 --thread 1 IN`.
The scripts patch the sources textually and may not apply to later revisions.
