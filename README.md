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

This fork makes MAFFT 7.526 faster under one constraint: for the same input and options, the
**output is byte-identical to unmodified MAFFT** built for the same platform and run with
`--thread 1` ([what that means](#identical-to-stock)). Downstream analyses calibrated on stock
MAFFT output are unaffected. Only how fast the result arrives changes.

It was built for two projects that align fungal ITS sequences in production:

- **A phylogenetics pipeline** running L-INS-i (`--localpair --maxiterate 1000`) on
  alignments of about 180 sequences, on Apple silicon and on AWS (x86 and Graviton).
- **[Dikarya](https://dikarya.us)**, a phylogenetics web service for fungal ITS barcodes,
  running `--auto` (mostly L-INS-i-style local pairs, sometimes FFT-NS-i) on an Intel
  Broadwell server without AVX-512 (the Dikarya host).

One source tree builds for Apple silicon, Linux arm64, x86-64 with AVX-512 or AVX2, and NVIDIA
GPUs. The vector code is chosen at compile time; everything else is shared.

## Results

Speedup over stock MAFFT 7.526 built the same way on the same machine, wall time, `--thread 1`
per run. "Pipeline L-INS-i" is the pipeline's test set of real ITS alignments (about 180
sequences each): a batch of 16 run four at a time on a 4-vCPU instance (eight at a time on the
M1), and single runs. Where the current version (`v7.526-opt8`) was not measured on a machine, the
row gives the most recent version that was.

| machine | workload | version | several runs at once | single run |
|---|---|---|---|---|
| Apple M1 (macOS), clang, with the Metal GPU | pipeline L-INS-i | opt8 | 5.1× | 8.1× |
| AWS c7a (AMD Zen 4), clang | pipeline L-INS-i | opt8 | 8.6×&nbsp;† | 10.5×&nbsp;† |
| AWS c7i (Intel Sapphire Rapids), clang | pipeline L-INS-i | opt8 | 7.5×&nbsp;† | 8.1×&nbsp;† |
| AWS c8a (AMD Zen 5), clang | pipeline L-INS-i | opt7 | 9.8× | 12× |
| AWS c8i (Intel Granite Rapids), clang | pipeline L-INS-i | opt7 | 8.2× | 9.3× |
| AWS c8g (Graviton4), clang | pipeline L-INS-i | opt8 | 5.0×&nbsp;† | 5.6×&nbsp;† |
| AWS c7g (Graviton3), clang | pipeline L-INS-i | opt7 | 5.1× | 5.6× |
| AWS g6 (NVIDIA L4, AMD Zen 3 host), clang + CUDA | pipeline L-INS-i | opt8 | 8.0×&nbsp;‡ | 9.6×&nbsp;‡ |
| AWS g4dn (NVIDIA T4, Intel Cascade Lake host), clang + CUDA | pipeline L-INS-i | opt7 | 5.8× | |
| AMD Zen 3 with AVX2 only (the g6 host), gcc `-march=broadwell` | pipeline L-INS-i | opt8 | 5.9× | 7.1× |
| the same | FFT-NS-i on the same alignments | opt8 | 2.6× | |
| Dikarya host: Intel Xeon E5-2690 v4 (Broadwell, AVX2), gcc | Dikarya `--auto` jobs | dikarya1 | | 6.4× on local-pair jobs, 2.3× on FFT-NS-i jobs, 6.7× on full L-INS-i |

† Stock was timed on the same instance type in an earlier session; on c8g, opt7 timed in both
sessions agreed within 1%, but c7i speed has drifted by up to 15% between sessions.<br>
‡ Against a stock gcc build (clang stock is 2–6% faster). The GPU's share: the same build
without CUDA was 6.5× and 7.6×.

The speedups depend on the workload and the hardware: sequence count and length, the
alignment strategy, cache sizes and vector units. Other inputs may gain much less. On AWS,
GPU instances pair the GPU with few CPU cores, and refinement, which stays on the CPU,
dominates: at on-demand prices, and at the throughput of the 16-alignment batches, the L4 costs
about $1.10 per 1,000 pipeline alignments, against about $0.17 on a c7a. The GPU path is for machines with a strong CPU and an idle GPU.

The technical report [`paper/mafft-apple-silicon.pdf`](paper/mafft-apple-silicon.pdf)
describes the methods, the verification and what didn't work in detail, including how the
per-architecture work of opt7 was done by parallel agents and consolidated in opt8.

## Building

The build is the same as upstream (see below); choose flags for your machine:

| machine | build in `core/` |
|---|---|
| macOS, Apple silicon | `make CC=clang && make install` |
| Linux arm64 (e.g. Graviton3/4) | `make CC=clang CFLAGS="-O3 -mcpu=neoverse-v1" && make install` |
| x86-64 with AVX-512 | `make CC=clang CFLAGS="-O3 -march=x86-64-v4 -DMAFFT_STOCK_FMA=1" && make install` |
| x86-64 with AVX2, no AVX-512 | `make CC=gcc CFLAGS="-O3 -march=native" && make install` |
| any Linux build with an NVIDIA GPU | add `CUDA=1` to both `make` commands |

- **macOS:** Accelerate and Metal are linked automatically. If Xcode's Metal compiler is
  installed, the GPU kernel is compiled at build time; otherwise at run time.
- **AVX-512:** on CPUs with VBMI2 (AMD Zen 4 and later, Intel Ice Lake and later), adding
  `-mavx512vpopcntdq -mavx512vbmi2` enables one more vector path; it was about 2% faster on
  Zen 4 and is how the c8a and c8i results above were built.
- **AVX2:** the gcc build is the Dikarya host's. clang works too, with `-DMAFFT_STOCK_FMA=1`
  when the target has FMA (see [below](#identical-to-stock)); it is the CPU side of the g6
  results above.
- **CUDA:** needs the CUDA toolkit; `CUDA_HOME`, `NVCC` and `CUDA_ARCH` can be set. Pairs
  outside the kernel's limits, a missing GPU or any CUDA error fall back to the CPU.
- **GPU off:** `MAFFT_NOGPU=1` makes the Metal and CUDA builds run everything on the CPU.
- **Other targets** compile the same algorithms in scalar code. They were not verified;
  before relying on one, compare its output with a stock build made by the same compiler
  (`harness/avx2/compare-builds.sh` runs six option sets).

On every target, don't add `-ffast-math` or `-ffp-contract=fast`: they change results. The
version string doesn't record the compiler or rounding class, so if you cache alignments,
include both in the cache key. Only `core/` differs from upstream; `extensions/` is unchanged.

## Identical to stock

Stock MAFFT itself doesn't produce the same alignments on every platform. Apple's clang
fuses `a*b+c` into one rounding (FMA); gcc with the Makefile's `-std=c99` never does. On the
pipeline's test set, a gcc build on x86 differed from the Mac on about a third of inputs. This
fork matches **the stock build of the same platform and rounding class**:

- **arm64 with clang** (macOS or Linux): stock built with the same compiler, which fuses by
  default. Linux arm64 clang builds reproduce the Mac's alignments.
- **x86-64 with clang** and an FMA-capable `-march`: clang fuses too. Build this fork the same
  way plus `-DMAFFT_STOCK_FMA=1` and it matches that stock build (and, in our tests, the
  Mac's alignments).
- **gcc** (any target): stock gcc, no flag needed.

`MAFFT_STOCK_FMA` must match what the compiler does to the rest of the code: a clang build
for an FMA target *without* the flag matches neither class. Multi-threaded runs
(`--thread` > 1) of the iterative methods are not reproducible even in stock MAFFT, so the
guarantee covers single-threaded runs.

## How it works

Every change keeps the original arithmetic: the same operations on the same values in the
same order wherever floating point is involved, or integer arithmetic where MAFFT's scores
are provably integral. The main pieces, by stage of an L-INS-i run:

- **All-pairs local alignment** (`Lalign11.c`, about 60% of stock L-INS-i). An integer fill
  that stores a marker instead of the gap-start offset in cells where a gap wins, and
  recomputes those offsets along the one traceback path by replaying the original rule;
  vectorized for NEON, SVE, AVX2 and AVX-512, plus a striped (Farrar-layout) fill on AVX-512.
  On Apple silicon and NVIDIA GPUs the same integer recurrence, tie rules and traceback run
  on the GPU (Metal, CUDA).
- **Refinement** (`dvtditr`, `partSalignmm.c`, `tddis.c`). A cache of per-pair scores that
  reuses a score while neither row has changed; closed-form pair scores (matrix products via
  Accelerate on macOS, exact integer sums elsewhere); vector DP rows whose multiply-adds
  round like the stock build's; the importance matrix filled in cache-sized bands and read
  back only over the columns written; substitution profiles skipping the letters absent from
  a column.
- **Bookkeeping everywhere.** Gap runs, residue maps, column profiles and anchor search
  scan 16–64 bytes at a time with byte-compare masks; gap-column removal moves runs instead
  of characters; work buffers are no longer cleared when every byte is written before it is
  read.

All vector code tests the feature macros defined once in `core/mafft_simd.h`; machines
without a vector unit run the same algorithms in scalar code.

## Verification

- **Per-call harnesses** ([`harness/`](harness)) patch a scratch copy of the source so that
  every call of a rewritten function is also computed the original way, compared bit for bit,
  and aborted on the first difference. They run across L-INS-i with three gap settings, the
  protein `test/sample`, FFT-NS-i and placement (`--add`, `--addfragments`).
- **Whole alignments** are compared with stock references: the pipeline's 183-alignment
  test set, and a canary kit of 13 L-INS-i windows and 12 placement cases.
- **Object comparison** (`harness/objcheck.sh`) shows which compiled objects a change touches
  in each build configuration.

State of `v7.526-opt8`:

| machine | build | result |
|---|---|---|
| Apple M1 | clang, with and without the GPU | canary 25/25 (13/13 without the GPU); harnesses identical |
| AWS c7a (Zen 4) | clang+FMA AVX-512, with and without VBMI2 flags | 183/183 test alignments = Mac reference; canary 25/25; harnesses identical |
| AWS c7a | gcc `x86-64-v3`, `x86-64-v4` | = stock gcc on a 43-alignment subset |
| AWS c7i (Sapphire Rapids) | clang+FMA AVX-512 | canary 25/25 |
| AWS c8g (Graviton4) | clang `-mcpu=neoverse-v1` | 183/183 = Mac reference; harnesses identical |
| AWS c8g | gcc | harnesses identical (gcc builds match stock gcc, not the Mac) |
| AWS g6 (L4, Zen 3 host) | clang+FMA `x86-64-v3` + CUDA | 183/183 = Mac reference; canary 25/25 with and without the GPU; 733,956 GPU alignments = CPU |
| AWS g6 host (Zen 3, AVX2 only) | gcc `-march=broadwell` (the Dikarya host's build class) | 183/183 = `v7.526-opt5-dikarya1`; 13/13 canary windows = stock gcc; FFT-NS-i 16/16 = dikarya1 |

`v7.526-opt8` has not yet been run on the Dikarya host itself. In its build class on the Zen 3
host it was 1.2× faster than dikarya1 (L-INS-i and FFT-NS-i) with identical output; if it is
slower on Broadwell, `v7.526-opt5-dikarya1` builds the earlier code. Some harness runs were on
an opt8 commit before the final tuning commit; `harness/opt8/README.md` lists which harness
scripts apply to which version.

## Versions

Each version is a tag on the `exact-speedups` branch, a linear series on upstream `main`
(7.526 plus commits from December 2025). `mafft --version` reports the tag's suffix
(`v7.526-opt1` reports plain `v7.526`). Use the newest unless you need to reproduce an older
build.

| tag | adds |
|---|---|
| `v7.526-opt1` | bookkeeping rewrites, NEON DP rows, closed-form pair scores via Accelerate (Apple silicon) |
| `v7.526-opt2` | refinement pair-score cache, vector prefix scan, run-based gap counts |
| `v7.526-opt3` | all-pairs local alignment on the GPU (Metal) |
| `v7.526-opt4` | GPU kernel compiled at build time, occupancy tuning |
| `v7.526-opt5` | x86-64 AVX-512 kernels; multiply-adds rounded per platform |
| `v7.526-opt5-dikarya1` | AVX2 paths for x86-64 without AVX-512, for the Dikarya host: the marker fill, sparse profile scores, uncleared `Falign` work rows |
| `v7.526-opt6` | the marker fill on AVX-512 builds too |
| `v7.526-opt7` | Linux arm64 (NEON/SVE), newer x86 AVX-512 kernels, the CUDA all-pairs stage |
| `v7.526-opt8` | one feature-test header and shared portable code for every target; the AVX-512 kernels without extra flags |

## For the MAFFT maintainers

`git diff 0a2319b v7.526-opt8 -- core/` is the whole change against upstream `main`
(`0a2319b`, the newest official source as of 2026-10-07). The feature tests are all in
`core/mafft_simd.h`, and each vector path keeps a scalar version of the same algorithm.

## Credits and license

- **opt1–opt5**: designed, implemented, verified and benchmarked by Claude Code (Claude Opus
  5.5, Anthropic), with Josh Walker advising and setting the research direction. See
  `paper/`.
- **dikarya1** (the AVX2 paths): designed, implemented, verified and benchmarked by Claude
  Code (Claude Opus 5.5, Anthropic), with Alan Rockefeller directing the work for Dikarya.
- **opt6–opt8**: Claude Code (Claude Opus 5.5, Anthropic), with Josh Walker advising, building
  on Alan Rockefeller's dikarya1 work. opt7 was written by three Claude Code agents in
  parallel, one per architecture, and merged by a coordinating session.
- MAFFT itself is by Kazutaka Katoh and colleagues. Please cite MAFFT as its authors ask
  (see the official site).

The code in `core/` is distributed under the BSD license in [`license`](license), and the
changes in this fork are offered under the same license. `extensions/` is covered by
[`license.extensions`](license.extensions).

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
