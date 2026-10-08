#!/bin/bash
# objcheck.sh OLD_TREE NEW_TREE [CONFIG...]
#
# Compile every core/*.c of two MAFFT source trees with the same compiler and flags and report
# which object files differ.  Used to show that a change for one platform leaves the code that
# other platforms compile untouched.  Each CONFIG is NAME, built in a scratch directory:
#
#   on macOS (Apple clang):
#     mac-arm64      clang -O3                                     (the M1 build)
#     x86-v4-fma     clang -arch x86_64 -O3 -march=x86-64-v4 -DMAFFT_STOCK_FMA=1  (AWS production)
#     x86-v4x-fma    clang -arch x86_64 -O3 -march=x86-64-v4 -mavx512vpopcntdq -mavx512vbmi2 -DMAFFT_STOCK_FMA=1
#     x86-v3-fma     clang -arch x86_64 -O3 -march=x86-64-v3 -DMAFFT_STOCK_FMA=1
#     x86-broadwell  clang -arch x86_64 -O3 -march=broadwell
#     x86-v2         clang -arch x86_64 -O3 -march=x86-64-v2
#     x86-base       clang -arch x86_64 -O3
#   on Linux, any of the above without -arch, plus gcc-<march> (gcc -O3 -march=<march>) and
#   clang-<march>, e.g. gcc-x86-64-v3, gcc-broadwell, gcc-native, clang-armv8.2-a.
#
# Defaults: mac-arm64 x86-v4-fma x86-v3-fma x86-broadwell x86-v2 x86-base (macOS), or
# gcc-native clang-native (Linux).  Exit status 1 if any object differs.
set -u
old=$1 new=$2; shift 2
os=$(uname)
if [ $# -eq 0 ]; then
  if [ $os = Darwin ]; then set -- mac-arm64 x86-v4-fma x86-v3-fma x86-broadwell x86-v2 x86-base
  else set -- gcc-native clang-native; fi
fi
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
rc=0
for cfg in "$@"; do
  arch=""; [ $os = Darwin ] && arch="-arch x86_64"
  case $cfg in
    mac-arm64)     cc="clang"; fl="-O3";;
    x86-v4-fma)    cc="clang $arch"; fl="-O3 -march=x86-64-v4 -DMAFFT_STOCK_FMA=1";;
    x86-v4x-fma)   cc="clang $arch"; fl="-O3 -march=x86-64-v4 -mavx512vpopcntdq -mavx512vbmi2 -DMAFFT_STOCK_FMA=1";;
    x86-v3-fma)    cc="clang $arch"; fl="-O3 -march=x86-64-v3 -DMAFFT_STOCK_FMA=1";;
    x86-broadwell) cc="clang $arch"; fl="-O3 -march=broadwell";;
    x86-v2)        cc="clang $arch"; fl="-O3 -march=x86-64-v2";;
    x86-base)      cc="clang $arch"; fl="-O3";;
    gcc-*)         cc="gcc"; fl="-O3 -march=${cfg#gcc-}";;
    clang-*)       cc="clang"; fl="-O3 -march=${cfg#clang-}";;
    *) echo "unknown config $cfg" >&2; exit 2;;
  esac
  n=0; d=0; list=""
  for side in old new; do
    src=$old; [ $side = new ] && src=$new
    mkdir -p $tmp/$cfg/$side
    for c in $src/core/*.c; do
      $cc $fl -std=c99 -Denablemultithread -c $c -o $tmp/$cfg/$side/$(basename $c .c).o 2>/dev/null \
        || echo "  compile failed ($side): $(basename $c)" >&2
    done
  done
  for o in $tmp/$cfg/old/*.o; do
    n=$((n+1)); b=$(basename $o)
    cmp -s $o $tmp/$cfg/new/$b || { d=$((d+1)); list="$list $b"; }
  done
  echo "$cfg: $n objects, $d differ${list:+:$list}"
  [ $d -gt 0 ] && rc=1
done
exit $rc
