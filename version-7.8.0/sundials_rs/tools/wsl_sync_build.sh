#!/usr/bin/env bash
# Sync the Windows-side working copy into the WSL Ubuntu x86-64 build sandbox
# and run the requested cargo step there.
#
#   ~/sdl/examples -> symlink to the read-only upstream SUNDIALS 7.8.0
#                     C tree's examples/ directory (verify_examples.sh reads
#                     ../examples/<solver>/<dir>/*.out)
#   ~/sdl/port     -> this workspace
#
# usage: bash tools/wsl_sync_build.sh <step>
set -u
export PATH="$HOME/.cargo/bin:/usr/bin:/bin:/usr/local/bin"
WIN_REPO=/mnt/c/Users/nsh/Developer/github/SUNDIALS_7_8_Rust_port_for_Linux
UPSTREAM=/mnt/c/Users/nsh/Developer/sundials-7.8.0
SB="$HOME/sdl"

mkdir -p "$SB/port"
ln -sfn "$UPSTREAM/examples" "$SB/examples"
rsync -a --delete --exclude target --exclude logs --exclude .git \
      "$WIN_REPO/" "$SB/port/"
cd "$SB/port" || exit 1
mkdir -p logs
# The Windows working copy may carry CRLF (git core.autocrlf); shell
# scripts must be LF for /usr/bin/env to find `bash`.
for f in tools/*.sh; do sed -i 's/\r$//' "$f"; done
chmod +x tools/*.sh

case "${1:-build}" in
  build) cargo build --workspace 2>&1 | tee logs/build.log | tail -40 ;;
  test)  cargo test --workspace --lib 2>&1 | tee logs/test.log | tail -40 ;;
  rel)   cargo build --release --workspace --examples 2>&1 | tee logs/build-rel.log | tail -40 ;;
  gate)  tools/verify_examples.sh all > logs/gate-run.log 2>&1
         echo "gate exit: $?"; tail -4 logs/summary.txt ;;
  pow)   tools/pow_differential.sh all ;;
  # Copy the run artefacts back into the Windows working copy under
  # evidence/, where they are tracked (logs/ itself is gitignored).
  evidence)
         D="$WIN_REPO/evidence/linux-x86_64-glibc239"
         mkdir -p "$D"
         cp logs/summary.txt logs/pow_differential.log "$D"/
         tools/classify_diffs.sh > "$D/classify_diffs.txt" 2>&1
         { uname -srm; ldd --version | head -1; gcc --version | head -1
           rustc -V; grep -m1 'model name' /proc/cpuinfo; } > "$D/host.txt"
         tools/compare_pristine_c.sh        > "$D/pristine_c_comparison.txt" 2>&1
         tools/compare_lapack_substituted.sh >> "$D/pristine_c_comparison.txt" 2>&1
         # Cross-distribution artefacts, if those sweeps have been run.
         [ -f logs/glibc-sweep.txt ] && cp logs/glibc-sweep.txt "$D"/
         for g in logs/gate-*.txt; do [ -e "$g" ] && cp "$g" "$D"/; done
         ls -l "$D" ;;
  sync)  echo "synced only" ;;
  *)     "$@" ;;
esac
