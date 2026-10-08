# MAFFT 7.526 with exact speedups (unofficial fork)

> [!WARNING]
> **This is an unofficial fork of MAFFT. It is not affiliated with or endorsed by the MAFFT
> authors.**
>
> - Official MAFFT: **https://mafft.cbrc.jp/alignment/software/**
> - Official source: **https://gitlab.com/sysimm/mafft**
>
> For general use, install the official release. Please don't report problems with this
> fork to the MAFFT developers. Open an issue here instead.

This fork adds performance changes to MAFFT 7.526 under one constraint: for the same input
and options, the **output must be byte-identical to unmodified MAFFT** built for the same
platform and run with `--thread 1`. Downstream analyses calibrated on stock MAFFT output
are unaffected. Only how fast the result arrives changes.

The changes come from two projects that align fungal ITS sequences in production:

- **A phylogenetics pipeline** running L-INS-i (`--localpair --maxiterate 1000`) on
  alignments of typically about 180 sequences, on an Apple M1 and on AWS x86 instances with
  AVX-512 (opt1–opt6), then extended to more AWS architectures (opt7): Graviton (Linux
  arm64), AMD Zen 5, Intel Granite Rapids, and NVIDIA GPUs with CUDA.
- **[Dikarya](https://dikarya.us)**, a phylogenetics web service for fungal ITS barcodes,
  running `--auto` (mostly L-INS-i-style local pairs, sometimes FFT-NS-i) on a Broadwell
  server without AVX-512 (`dikarya1`).

The same source tree builds for all of these machines. Each vector path is chosen at compile
time, and every other target keeps the code it had before.

## Results

| platform | build | workload | speedup vs. stock 7.526 |
|---|---|---|---|
| Apple M1 (macOS) | `v7.526-opt4` and later (same arm64 code) | pipeline L-INS-i | 6.5× single run, 4.6× at 8 concurrent runs |
| AWS c7i / c7a (Sapphire Rapids, Zen 4), clang | `v7.526-opt5` | pipeline L-INS-i | 5.7–7.0× single run, 5.1–5.9× at 4 concurrent runs |
| AWS c7a / c7i (Zen 4, Sapphire Rapids), clang | `v7.526-opt6` | pipeline L-INS-i | a further 5% faster than opt5 at 4 concurrent runs on both; 9% (c7a) and 4% (c7i) in a single run (see [opt6](#opt6-the-marker-fill-on-avx-512)) |
| AWS c8a (AMD Zen 5), clang | `v7.526-opt7`, new x86 flags | pipeline L-INS-i | 9.8× at 4 concurrent runs, 12× single run; 1.6× over opt6 (see [opt7](#opt7-more-architectures)) |
| AWS c8i (Intel Granite Rapids), clang | `v7.526-opt7`, new x86 flags | pipeline L-INS-i | 8.2× at 4 concurrent runs, 9.3× single run; 1.4× over opt6 |
| AWS c8g / c7g (Graviton4 / Graviton3), clang | `v7.526-opt7` | pipeline L-INS-i | 4.8× / 5.1× at 4 concurrent runs, 5.4× / 5.6× single run; 1.3–1.4× over opt6 |
| AWS g4dn (NVIDIA T4, Cascade Lake host), clang + CUDA | `v7.526-opt7`, `CUDA=1` | pipeline L-INS-i | 5.8× at 4 concurrent runs; the GPU adds 1.2× over the CPU-only build |
| Intel Xeon E5-2690 v4 (Broadwell, AVX2, no AVX-512), gcc | `v7.526-opt5-dikarya1` | Dikarya `--auto` jobs | 6.4× on `--auto` local-pair jobs, 2.3× on `--auto` FFT-NS-i jobs, 5.5–6.9× on full L-INS-i (see [dikarya1](#dikarya1-avx2-for-x86-without-avx-512)) |

The speedups depend on the workload and the hardware: sequence count and length, the
alignment strategy, cache sizes and vector units. Other inputs may gain much less. The
technical report [`paper/mafft-apple-silicon.pdf`](paper/mafft-apple-silicon.pdf) describes
opt1–opt5: every change, how exactness was verified, the results, and what didn't work.

## The series

The changes are a linear series of commits on the `exact-speedups` branch, on top of upstream
`main` (7.526 plus commits from December 2025). Each build is tagged:

| tag | `mafft --version` | summary |
|---|---|---|
| `v7.526-opt1` | `v7.526` | bookkeeping fixes, NEON DP rows, matrix-product scores via Accelerate |
| `v7.526-opt2` | `v7.526-opt2` | refinement pair-score cache, vector prefix scan, run-based counts |
| `v7.526-opt3` | `v7.526-opt3` | all-pairs local alignment on the GPU (Metal) |
| `v7.526-opt4` | `v7.526-opt4` | shader compiled at build time, occupancy tuning |
| `v7.526-opt5` | `v7.526-opt5` | AVX-512 kernels for x86-64, multiply-add rounding matched per platform |
| `v7.526-opt5-dikarya1` | `v7.526-opt5-dikarya1` | AVX2 paths for x86-64 without AVX-512: marker-based `Lfill_int`, sparse `scarr_fill`, uncleared `Falign` work rows ([details](#dikarya1-avx2-for-x86-without-avx-512)) |
| `v7.526-opt6` | `v7.526-opt6` | dikarya1's marker-based `Lfill_int` on AVX-512 builds too ([details](#opt6-the-marker-fill-on-avx-512)) |
| `v7.526-opt7` | `v7.526-opt7` | Linux arm64 (NEON/SVE), x86 kernels for AVX-512 CPUs with VPOPCNTDQ and VBMI2, CUDA all-pairs stage, AVX2+FMA paths ([details](#opt7-more-architectures)) |

`v7.526-opt1` reports plain `v7.526`, so it can't be told apart from stock by version.
Prefer a later tag.

### dikarya1: AVX2 for x86 without AVX-512

The opt5 x86 kernels need AVX-512 (`-march=x86-64-v4`). On x86 CPUs without it (every
Intel server part before Skylake-SP, and many cloud and desktop machines today), opt5 falls
back to its SSE4.1 and scalar paths, which leaves most of the run time where it was in stock
MAFFT. `v7.526-opt5-dikarya1` adds AVX2 paths for that class of machine, under the same rule
as the rest of this fork.

They were written for, and measured on, the server that runs Dikarya:

- Intel Xeon E5-2690 v4 (Broadwell-EP), 2 vCPUs of a VM, AVX2 + FMA, no AVX-512
- Ubuntu 24.04, gcc 13.3, `make CC=gcc CFLAGS="-O3 -march=native"` (gcc with the Makefile's
  `-std=c99` never contracts `a*b+c`, so `MAFFT_STOCK_FMA` is 0 and every vector multiply-add
  below is a separate multiply and add)

The new code is guarded by `__AVX2__ && !MAFFT_STOCK_FMA`, and by `!__AVX512F__` where opt5
already had an AVX-512 path, so builds for other targets compiled exactly the code they did
before. (opt6 later enabled the marker fill on AVX-512 builds as well. Builds without
AVX-512 still compile exactly the dikarya1 code.)

| where | change | effect on this machine |
|---|---|---|
| `Lalign11.c`: `Lfill_int`, `Ltracking` | An AVX2 integer fill, 8 cells per step. A cell whose best move is a gap stores a marker instead of the gap's start offset; every DP row is kept instead of two alternating ones; `Ltracking`, which reads `ijp` only along its one path, recomputes the offset for a marked cell by replaying the original scalar rule over the stored rows. The arithmetic is integer, so the offsets are exactly the ones the full fill would store. This removes the first-index prefix scan and the `vmp[]` state from the inner loop. | 1.68 to 0.82 ns per DP cell (the all-pairs local alignments are ~60% of L-INS-i) |
| `mltaln9.c`: `scarr_fill()`; `Salignmm.c`, `partSalignmm.c`, `Dalignmm.c`, `SAalignmm.c`; `mc_match` | The 26x26 profile-score loop at the top of every `match_calc` (`scarr[l] += mtx[j][l] * cpmx1[j][i1]`): gcc vectorised it as gathers plus a serial add chain. It now skips the letters absent from the column (exact: the sum starts at +0, and under round-to-nearest a sum of nonzero terms is never -0, so adding +0 or -0 never changes it) and does 4 letters at a time, each lane with the same multiply-then-add sequence in the same order. | most of `A__align`, i.e. of FFT-NS-i |
| `Falign.c`, `mtxutl.c`: `AllocateCharMtxNoZero()` | `Falign` allocated eight `njob x alloclen` character matrices per call with `calloc`, i.e. cleared ~6 MB per call. They are only ever used as C strings written before they are read (`rndseq` is filled whole), so the rows are now `malloc`ed. `-DMAFFT_POISON_TEST` fills them with junk instead, which is how that claim was checked. | ~25% of an FFT-NS-i run |
| `Salignmm.c` `A_row`, `partSalignmm.c` `partA_row` | AVX2 (4 doubles) versions of opt5's AVX-512-only row fills. | small |
| `Lalign11.c` | 8-wide search for the first cell holding a row maximum. | small |

Tried and dropped: a straight AVX2 port of the opt5 `Lfill_int` prefix scans (only 1.17x
over SSE4.1: the loop is bound by the number of vector uops per cell, not by width); a
variant that computed the gap offsets only for blocks that needed them (slower: on related
ITS sequences gaps win across large areas, and the branch mispredicts); restructuring the
scan to shorten the loop-carried chain (no change).

**Results on the Broadwell server.** 14 real inputs from the service (49-444 ITS/LSU
sequences), `--adjustdirection --auto`, so `--auto` chose L-INS-i-style local pairs for 10
of them and FFT-NS-i for 4. Seconds, single runs:

| | stock 7.526 | `opt5` | `opt5-dikarya1` | vs. stock | vs. opt5 |
|---|---|---|---|---|---|
| `--auto` chose local-pair (10 inputs), `--thread 1` | 361 | 97 | 56 | 6.4x | 1.7x |
| `--auto` chose FFT-NS-i (4 inputs), `--thread 1` | 242 | 207 | 107 | 2.3x | 1.9x |
| all 14, `--thread 2` | not run | 199 | 99 | n/a | 2.0x |
| full L-INS-i, 176 seqs, `--thread 1` | 93.2 | 23.9 | 13.9 | 6.7x | 1.7x |

**How it was verified.**

- Byte-identical alignments to stock MAFFT (Ubuntu's 7.505 package and a gcc build of
  7.526) at `--thread 1` on 52 real inputs, both as built and with `-DMAFFT_POISON_TEST`.
- `harness/avx2/compare-builds.sh` runs six option sets (`--auto`, L-INS-i, G-INS-i,
  FFT-NS-i, FFT-NS-2 with `--retree 2`, `--adjustdirection`) against a reference build.
  The ones run were a synthetic DNA family, `test/sample` (protein) and three real inputs:
  all identical to stock 7.526.
- Multi-threaded runs are not reproducible in any MAFFT build, so `--thread 2` was checked
  for completion, not identity.
- Independently, on an AWS c7a (Zen 4), the dikarya1 gcc AVX2 build (`-march=x86-64-v3`)
  matched gcc stock 7.526 on all 183 shards of the pipeline's L-INS-i test set, and its
  `-DMAFFT_POISON_TEST` build on a 53-shard subset. There it ran 1.31× faster than opt5
  built the same way.

### opt6: the marker fill on AVX-512

dikarya1's marker fill is integer-only, so it gives the same result under every rounding
class, including the clang+FMA builds the pipeline runs. On an AVX-512 machine it also beat
opt5's AVX-512 prefix-scan fill, so opt6 uses it on AVX-512 builds too. That is a one-line
change to its compile guard. Builds without AVX-512 compile exactly the same objects as
`v7.526-opt5-dikarya1`, apart from the version string. This was checked for gcc
`-march=broadwell`, `x86-64-v3`, `x86-64-v2` and the default target, and for clang
`x86-64-v3` with `-DMAFFT_STOCK_FMA=1`. arm64 builds are unchanged.

On AWS, pipeline L-INS-i shards, clang `-O3 -march=x86-64-v4 -DMAFFT_STOCK_FMA=1`, means of
three interleaved rounds. A c7a.xlarge has 4 cores; a c7i.xlarge has 2 cores with 2 threads
each.

| build | c7a (Zen 4): 16 shards, 4 concurrent, wall / CPU | c7a: 2 single shards | c7i (Sapphire Rapids): 16 shards, 4 concurrent | c7i: 2 single shards |
|---|---|---|---|---|
| `v7.526-opt5` | 74.3 s / 281 s | 32.0 s | 121.3 s / 467 s | 33.6 s |
| `v7.526-opt6` | 70.3 s / 261 s | 29.2 s | 115.7 s / 444 s | 32.4 s |
| (tried) a 16-lane AVX-512 version of the marker fill | 70.3 s / 262 s | 29.1 s | 115.0 s / 443 s | 32.0 s |

The 16-lane version was no faster than the 8-lane one on either machine, so it was not
adopted.

Verification:

- `harness/opt6/marks_check.py` runs every call of the marker fill with and without its
  vector loops and compares the results bit for bit. It ran in three builds: gcc AVX2,
  clang AVX-512 with the 8-lane fill (opt6), and clang AVX-512 with the 16-lane variant.
  Each build ran 13 test alignments under three gap settings plus `test/sample` (732,696
  calls, 2.8 × 10^11 cells each), and the two AVX-512 builds also ran the 12 placement cases
  (`--add` and `--addfragments`). All were identical.
- The opt6 build matched the stored clang+FMA reference alignments on all 183 shards of the
  pipeline's test set on both the c7a and the c7i, and opt5's output on all 12 placement
  cases.

### opt7: more architectures

opt7 merges three parallel workstreams, each written and measured on its own AWS hardware.
Each one keeps the code that every other target compiles unchanged: an object-file comparison
(`harness/objcheck.sh`, which compiles every `core/*.c` of two trees with the same flags) shows
the M1 build, the x86 `-march=x86-64-v4` production build, the AVX2 builds and the baseline
builds compiling the same objects as `v7.526-opt6`, apart from the version string.

**x86 with newer AVX-512 (Zen 4+, Ice Lake+).** New kernels behind `MAFFT_AVX512X`, which is
defined only when VPOPCNTDQ and VBMI2 are enabled (for example `-march=x86-64-v4
-mavx512vpopcntdq -mavx512vbmi2`, or `-march=znver4`/`znver5`/`sapphirerapids`/
`graniterapids`; plain `-march=x86-64-v4` does not enable them):

- a striped (Farrar-layout) integer local-alignment fill: the horizontal-gap state is a running
  maximum along each lane plus a carry taken from the previous row, so there is no prefix scan
  in the loop; the traceback is one byte per cell, resolved with opt6's marker replay rules;
- the importance-matrix gather reads only the written column range of each row (the rest is
  +0.0), and the matrix is one 2 MB-aligned block marked for huge pages; `fillimp_track`
  walks it from one base pointer;
- `match_calc` in blocks of 8 columns with masked multiply-adds, keeping each column's chain;
- pair scores from per-letter bit masks and popcount (exact integer sums), traceback rows
  built with VBMI2 expand, and profile passes 64 columns at a time with masked adds.

| c8a.xlarge (Zen 5) / c8i.xlarge (Granite Rapids), clang+FMA | 16 shards, 4 concurrent: wall / CPU | 2 single shards |
|---|---|---|
| stock 7.526 | 283.7 s / 1062 s · 494.7 s / 1884 s | 57.5 + 84.0 s · 68.4 + 86.6 s |
| `v7.526-opt6` | 46.0 s / 174 s · 86.0 s / 333 s | 7.7 + 10.9 s · 10.4 + 15.2 s |
| `v7.526-opt7`, new flags | 29.0 s / 110 s · 60.7 s / 234 s | 4.8 + 6.8 s · 6.8 + 9.9 s |

Tried and dropped: the 16-lane marker fill (it helped on Zen 5 but the striped fill replaces
it), fills storing flag bits instead of the int traceback (10–25% slower), row-maximum
tracking inside the fill loop, a gather/scatter `fillimp` walk, and AMX (the only
matrix-shaped work was ~3% of the run and is now exact popcount sums).

**Linux arm64 (AWS Graviton).** Behind `__ARM_NEON && !__APPLE__`, `__ARM_FEATURE_SVE` or
`__linux__ && __aarch64__`, so the Apple M1 build is unchanged. clang builds on Linux arm64
reproduce the Mac's alignments (183/183 test shards), so they can share the Mac's alignment
cache; gcc builds form their own class, as on x86.

- dikarya1's marker fill with NEON loops, and a vector-length-agnostic SVE loop that is used at
  run time only when vectors are at least 256 bits (Graviton3; on Graviton4's 128-bit SVE2 the
  NEON loop is faster);
- the end cell of each local alignment is found once, after the last row, instead of
  searching every row whose maximum improved (all-pairs stage 10–12% faster);
- `fillimp_track` in bands of 128 importance-matrix rows (Graviton's L2 holds a band), the
  matrix in transparent huge pages, table-lookup pair-score sums, NEON `mc_match`,
  `alignableReagion` and gap-run passes, and the opt1 NEON row fills for gcc builds too.

| clang+FMA, `-mcpu=neoverse-v1` | c8g.xlarge (Graviton4): 16 shards, 4 concurrent | c8g: 2 singles | c7g.xlarge (Graviton3): 16 shards, 4 concurrent | c7g: 2 singles |
|---|---|---|---|---|
| stock 7.526 | 377.3 s / 1407 s | 75.5 + 98.7 s | 474.0 s / 1760 s | 96.0 + 122.6 s |
| `v7.526-opt6` | 105.7 s / 402 s | 21.0 + 26.8 s | 131.0 s / 496 s | 26.9 + 33.6 s |
| `v7.526-opt7` | 79.3 s / 301 s | 13.5 + 18.5 s | 92.3 s / 349 s | 16.4 + 22.7 s |

**NVIDIA GPUs (CUDA).** `make CUDA=1` adds a CUDA version of the Metal all-pairs kernel
(`core/l11gpu.cu`): the same integer recurrence, tie rules and traceback, one pair per warp,
each lane holding a strip of columns in registers. Pairs outside its limits, a missing GPU or
driver, or any CUDA error fall back to the CPU; `MAFFT_NOGPU=1` disables it. Builds without
`CUDA=1` are unchanged. The same workstream added AVX2 paths for clang+FMA builds of CPUs
without AVX-512 (`MAFFT_AVX2_PATHS`).

On AWS the GPU does not pay. GPU instances pair a large GPU with few CPU cores, and the
refinement stages, which stay on the CPU, dominate. On a g4dn.xlarge (T4; host Intel Cascade
Lake, 2 cores with 2 threads), 16 shards at 4 concurrent took 895 s with stock, 182 s with
opt7 on the CPU and 154 s with the GPU: about $1.40 per 1,000 shards, against about $0.11 on
a c8a.xlarge with opt7. The GPU path is for machines with a strong CPU and an idle GPU.

**Alan's AVX2 build is unchanged by default.** Two AVX2 passes added in opt7 (`gapruns` and
`alignableReagion`, 32 bytes at a time) would change the objects of the gcc AVX2 build that
dikarya1 verified on its Broadwell server. In that (non-contracting) class they are opt-in:
build with `-DMAFFT_AVX2_EXTRA=1` to try them, and compare with `harness/avx2/compare-builds.sh`
before relying on them. They were verified on AVX2-capable AWS CPUs, not on Broadwell.

**Verification.** Per-call harnesses for every rewritten function (`harness/x86`,
`harness/arm64`, `harness/cuda`), run across L-INS-i with three gap settings, the protein
`test/sample`, other strategies and the placement cases; then the merged tree on each machine:

| machine | build | 183 test shards | canary (13 windows + 12 placements) |
|---|---|---|---|
| c8a (Zen 5) | clang+FMA, new x86 flags; and `-march=x86-64-v4` | 183/183 = Mac reference (both) | 25/25 (both) |
| c8i (Granite Rapids) | clang+FMA, new x86 flags; and `-march=x86-64-v4` | 183/183 (new flags), 43/43 subset (v4) | 25/25 (both) |
| c8g / c7g (Graviton4 / 3) | clang `-mcpu=neoverse-v1` | 183/183 = Mac reference (both) | 25/25 (both) |
| all four | gcc | = stock gcc on the 43-shard subset | |
| g4dn (T4) | clang+FMA `-march=x86-64-v4` + `CUDA=1` | 183/183 = Mac reference | 25/25; also 25/25 for the AVX2+FMA builds with and without CUDA |
| g4dn (T4) | gcc AVX2 + `CUDA=1`, default and with `-DMAFFT_AVX2_EXTRA=1` | = stock gcc on the 13 canary windows (both) | |

## What "identical to stock" means

Stock MAFFT itself doesn't produce the same alignments on every platform. Apple's clang
fuses `a*b+c` into one rounding (FMA). gcc with the Makefile's `-std=c99` never does. On
our test set, a gcc build on x86 differed from the Mac on about a third of inputs. This
fork matches **the stock build of the same platform and rounding class**:

- On **arm64 with clang**, it matches stock built with the same compiler (fused
  multiply-adds, the default).
- On **x86-64**, clang with an FMA-capable `-march` fuses too. Build this fork the same way
  plus `-DMAFFT_STOCK_FMA=1`, and it matches that stock build. In our tests, those
  alignments also matched the Mac's. Built with gcc (no flag), it matches stock gcc
  instead. The flag must match what the compiler does to the rest of the code. A clang
  build for an FMA target *without* the flag matches neither.

Multi-threaded runs (`--thread` > 1) of the iterative methods are not reproducible even in
stock MAFFT. The guarantee covers single-threaded runs only.

## Building

The build is the same as upstream (see below), with these additions:

- **macOS (Apple silicon).** `cd core && make CC=clang && make install`. The Makefile links
  Accelerate and Metal automatically. If Xcode's Metal compiler is installed, the GPU kernel
  is compiled at build time; otherwise MAFFT compiles it at run time. Setting `MAFFT_NOGPU=1`
  disables the GPU path.
- **x86-64 with AVX-512** (verified on Intel Sapphire Rapids and AMD Zen 4):

      cd core
      make CC=clang CFLAGS="-O3 -march=x86-64-v4 -DMAFFT_STOCK_FMA=1"
      make install

- **x86-64 with AVX-512 and VPOPCNTDQ/VBMI2** (Zen 4 and later, Ice Lake and later; verified
  on AMD Zen 5 and Intel Granite Rapids). This enables opt7's x86 kernels:

      cd core
      make CC=clang CFLAGS="-O3 -march=x86-64-v4 -mavx512vpopcntdq -mavx512vbmi2 -DMAFFT_STOCK_FMA=1"
      make install

  These flags were not yet measured on Zen 4 (AWS c7a) or Sapphire Rapids (c7i), which also
  have both extensions; until they are, those machines keep the `-march=x86-64-v4` build.
- **Linux arm64** (verified on AWS Graviton3 and Graviton4 with clang 15; one binary for both):

      cd core
      make CC=clang CFLAGS="-O3 -mcpu=neoverse-v1"
      make install

- **NVIDIA GPU**: add `CUDA=1` to `make` and `make install` on any x86 or arm64 Linux build
  (needs the CUDA toolkit; `CUDA_HOME`, `NVCC` and `CUDA_ARCH` can be set). Verified on
  NVIDIA L4 and T4.
- **x86-64 with AVX2 but no AVX-512** (verified on Intel Broadwell with gcc 13.3, and with
  gcc 11.5 `-march=x86-64-v3` on AMD Zen 4):

      cd core
      make CC=gcc CFLAGS="-O3 -march=native"
      make install

- **Other targets.** The vector paths are selected at compile time (`__ARM_NEON`,
  `__AVX512F__`/`__AVX512BW__`/`__AVX512VL__`, `__AVX2__`, `__SSE4_1__`). Without them, the
  original scalar code is used, along with the portable algorithmic changes. These
  configurations compile but were not verified. Before relying on one, compare its output
  with a stock build made by the same compiler (`harness/avx2/compare-builds.sh` does this
  over six option sets).

On every target, don't add `-ffast-math` or `-ffp-contract=fast`, since they change results.
The version string doesn't record the compiler or rounding class. If you cache alignments,
include the compiler and rounding class in the cache key.

Only `core/` changed. `extensions/` is identical to upstream.

## For the MAFFT maintainers

The branch is upstream `main` (`0a2319b`, the newest official source as of 2026-10-07) plus
this series, so `git diff 0a2319b v7.526-opt7 -- core/` is the whole change against upstream.
Each tag's diff against the one before is one part of the work, for example
`git diff v7.526-opt5 v7.526-opt5-dikarya1 -- core/` for the AVX2 part. Each part keeps the
original code as the fallback for every other target.

## Credits and license

- **opt1–opt5** (everything up to `v7.526-opt5`): designed, implemented, verified and
  benchmarked by Claude Code (Claude Opus 5.5, Anthropic), with Josh Walker advising and
  setting the research direction. See `paper/`.
- **dikarya1** (the AVX2 additions): designed, implemented, verified and benchmarked by
  Claude Code (Claude Opus 5.5, Anthropic), with Alan Rockefeller directing the work for
  Dikarya.
- **opt6** (dikarya1's marker fill on AVX-512 builds): Claude Code (Claude Opus 5.5,
  Anthropic), with Josh Walker advising, building on Alan Rockefeller's dikarya1 work.
- **opt7** (Linux arm64, newer x86 AVX-512, CUDA): three Claude Code (Claude Opus 5.5,
  Anthropic) agents working in parallel, one per architecture, merged and verified by a
  coordinating Claude Code session, with Josh Walker advising and setting the research
  direction.
- MAFFT itself is by Kazutaka Katoh and colleagues. Please cite MAFFT as its authors ask
  (see the official site).

The code in `core/` is distributed under the BSD license in [`license`](license), and the
changes in this fork are offered under the same license. `extensions/` is covered by
[`license.extensions`](license.extensions).

The per-call verification harnesses are in [`harness/`](harness): `opt5` for the AVX-512
kernels, `avx2` for whole-build comparisons, `opt6` for the marker fill, and `x86`, `arm64`
and `cuda` for opt7.

---

*The upstream README follows, unchanged.*


# MAFFT version 7.526
Multiple sequence alignment program
<br>
https://mafft.cbrc.jp/alignment/software/

## COMPILE
     % cd core
     % make clean
     % make
     % cd ..

If you have the `./extensions` directory, which is for RNA alignments,

     % cd extensions
     % make clean
     % make
     % cd ..


## INSTALL (select a or b below)
###  a. Install to /usr/local/ using root account
     # cd core
     # make install
     # cd ..

If you have the `./extensions` directory,

     # cd extensions 
     # make install
     # cd ..

By this procedure (a), programs are installed into `/usr/local/bin/`. Some binaries, which are not directly used by a user, are installed into `/usr/local/libexec/mafft/`.

If the MAFFT_BINARIES environment variable is set to `/somewhare/else/`, the binaries in this directory are used, instead of those in `/usr/local/libexec/mafft/`.

### b. Install to non-default location (root account is not necessary)
     % cd core/
          Edit the first line of Makefile 
          From:
          PREFIX = /usr/local
          To:
          PREFIX = /home/your_home/somewhere

          Edit the third line of Makefile 
          From:
          BINDIR = $(PREFIX)/bin
          To:
          BINDIR = /home/your_home/bin 
                   (or elsewhere in your command-search path)
     % make clean
     % make
     % make install

If you have the `./extensions` directory,

     % cd ../extensions/
          Edit the first line of Makefile 
          From:
          PREFIX = /usr/local
          To:
          PREFIX = /home/your_home/somewhere
     % make clean
     % make
     % make install

The `MAFFT_BINARIES` environment variable *must not be* set.

If the `MAFFT_BINARIES` environment variable is set to `/somewhare/else/`, it overrides the setting of `PREFIX` (`/home/your_home/somewhere/` in the above example) in Makefile.

## CHECK
     % cd test
     % rehash                                                   # if necessary
     % mafft sample > test.fftns2                               # FFT-NS-2
     % mafft --maxiterate 100  sample > test.fftnsi             # FFT-NS-i
     % mafft --globalpair sample > test.gins1                   # G-INS-1 
     % mafft --globalpair --maxiterate 100  sample > test.ginsi # G-INS-i 
     % mafft --localpair sample > test.lins1                    # L-INS-1 
     % mafft --localpair --maxiterate 100  sample > test.linsi  # L-INS-i 
     % diff test.fftns2 sample.fftns2
     % diff test.fftnsi sample.fftnsi
     % diff test.gins1 sample.gins1
     % diff test.ginsi sample.ginsi
     % diff test.lins1 sample.lins1

If you have the `./extensions` directory,

     % mafft-qinsi samplerna > test.qinsi                       # Q-INS-i
     % mafft-xinsi samplerna > test.xinsi                       # X-INS-i
     % diff test.qinsi samplerna.qinsi
     % diff test.xinsi samplerna.xinsi

If you use the multithread version, the results of iterative refinement methods (`*-*-i`) are not always identical.  So try this test in the single-thread mode (`--thread 0`).


## INPUT FORMAT
Fasta format.

The type of input sequences (nucleotide or amino acid) is automatically recognized based on the frequency of A, T, G, C, U and N.


##  USAGE
     % /usr/local/bin/mafft input > output

See also https://mafft.cbrc.jp/alignment/software/


## UNINSTALL
     # rm -r /usr/local/libexec/mafft
     # rm /usr/local/bin/mafft
     # rm /usr/local/bin/fftns
     # rm /usr/local/bin/fftnsi
     # rm /usr/local/bin/nwns
     # rm /usr/local/bin/nwnsi
     # rm /usr/local/bin/linsi
     # rm /usr/local/bin/ginsi
     # rm /usr/local/bin/mafft-*
     # rm /usr/local/share/man/man1/mafft*


## LICENSE
See `./license` and `./license.extensions`.
