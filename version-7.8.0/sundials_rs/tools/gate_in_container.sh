#!/usr/bin/env bash
# gate_in_container.sh — run the full 199-variant example gate inside
# another distribution's container, natively.
#
# tools/glibc_sweep.sh fingerprints each distribution's libm and shows
# where they disagree. Where they do, the only way to know whether the
# disagreement is output-observable is to run the gate there. This script
# does that: it copies the workspace into a container, installs a minimal
# Rust toolchain, builds the release examples and runs
# tools/verify_examples.sh, then prints the IDENTICAL / DIFF / EXCLUDED
# tally and the list of variants that differ from the host's result.
#
#   tools/gate_in_container.sh debian:12 archlinux:latest
#   tools/gate_in_container.sh --platform linux/arm64 debian:13
#
# --platform runs the images for another architecture (or $GATE_PLATFORM).
# On a foreign architecture that needs user-mode emulation registered in
# binfmt_misc; the script checks for it up front and says what to install,
# because the failure without it is the unhelpful "exec container process
# /bin/sh: Exec format error" *after* pulling the image.
#
# Requires docker or podman, and network access (each container downloads
# rustup). $CONTAINER_RUNTIME overrides the auto-detection.
#
# The container is entered through /bin/sh, not bash: Alpine ships busybox ash
# and would fail before the installer could add bash. Everything before the
# installer line must therefore stay POSIX; tools/*.sh are invoked as
# `bash tools/...` explicitly.
# Nothing is installed on the host and nothing is written into the
# workspace; the container gets read-only mounts and copies what it needs.
# Per-distribution summaries are written to logs/gate-<image>.txt, or
# logs/gate-<image>-<arch>.txt when --platform names a foreign architecture,
# so an emulated run can never overwrite the native one.
set -u
cd "$(dirname "$0")/.."
WS_ROOT="$PWD"
UP="$(cd .. && pwd)"
LOGS="$WS_ROOT/logs"
mkdir -p "$LOGS"
rc=0

PLATFORM="${GATE_PLATFORM:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --platform) PLATFORM="${2:?--platform needs an argument, e.g. linux/arm64}"; shift 2 ;;
    --platform=*) PLATFORM="${1#--platform=}"; shift ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: gate_in_container.sh [--platform linux/arm64] <image>..." >&2; exit 2; }

# docker is not the only game: podman takes the same arguments for everything
# used here, and is what is installed on the Ubuntu 26.04 host.
RT="${CONTAINER_RUNTIME:-}"
if [ -z "$RT" ]; then
  for c in docker podman; do command -v "$c" >/dev/null 2>&1 && { RT=$c; break; }; done
fi
[ -n "$RT" ] || { echo "no container runtime: install docker or podman"; exit 1; }
"$RT" info >/dev/null 2>&1 || { echo "$RT is installed but its daemon is not reachable"; exit 1; }
echo "container runtime: $RT"

# Emulation preflight. Without this the run pulls a few hundred MB and then
# dies with "Exec format error", which says nothing about the cause.
PLAT_ARGS=""
if [ -n "$PLATFORM" ]; then
  PLAT_ARGS="--platform $PLATFORM"
  want_arch="${PLATFORM##*/}"
  host_arch="$(uname -m)"
  case "$host_arch:$want_arch" in
    x86_64:amd64|x86_64:x86_64|aarch64:arm64|aarch64:aarch64) native=yes ;;
    *) native=no ;;
  esac
  if [ "$native" = no ]; then
    case "$want_arch" in
      arm64|aarch64) handler=qemu-aarch64 ;;
      arm|armv7|armhf) handler=qemu-arm ;;
      ppc64le) handler=qemu-ppc64le ;;
      s390x) handler=qemu-s390x ;;
      riscv64) handler=qemu-riscv64 ;;
      *) handler="qemu-$want_arch" ;;
    esac
    if [ ! -e "/proc/sys/fs/binfmt_misc/$handler" ]; then
      cat >&2 <<EOF
$PLATFORM needs user-mode emulation on this $host_arch host, and
/proc/sys/fs/binfmt_misc/$handler is not registered.

  sudo apt install qemu-user-binfmt     # Debian/Ubuntu; qemu-user-static elsewhere

Registered handlers here: $(ls /proc/sys/fs/binfmt_misc/ 2>/dev/null | grep -v '^register$\|^status$' | tr '\n' ' ')
EOF
      exit 1
    fi
    echo "platform: $PLATFORM (emulated on $host_arch via $handler)"
    EMU=" [EMULATED]"
  else
    echo "platform: $PLATFORM (native)"
    EMU=""
  fi
else
  EMU=""
fi

# The example tree used to live only in the parent directory. It is vendored
# at the workspace root now, and verify_examples.sh prefers that copy, so the
# parent is a fallback rather than a requirement -- this used to exit here.
if [ -d "$WS_ROOT/examples/cvode/serial" ]; then
  EXAMPLES="$(readlink -f "$WS_ROOT/examples")"
elif [ -d "$UP/examples/cvode/serial" ]; then
  EXAMPLES="$(readlink -f "$UP/examples")"
else
  echo "no upstream examples/ at $WS_ROOT/examples or $UP/examples"; exit 1
fi

installer() {
  case "$1" in
    debian*|ubuntu*) echo 'export DEBIAN_FRONTEND=noninteractive; apt-get -qq update >/dev/null && apt-get -qq install -y gcc curl ca-certificates diffutils >/dev/null' ;;
    fedora*)         echo 'dnf -y -q install gcc curl diffutils >/dev/null' ;;
    archlinux*)      echo 'pacman -Sy --noconfirm --quiet gcc curl diffutils >/dev/null' ;;
    # Alpine is musl, and the only image here without bash: busybox ash is
    # /bin/sh, so bash has to be installed before tools/*.sh can run at all.
    # musl-dev is what lets rustup-init pick the musl host triple.
    alpine*)         echo 'apk add --no-cache bash gcc musl-dev curl ca-certificates diffutils tar >/dev/null' ;;
    *)               echo 'true' ;;
  esac
}

for image in "$@"; do
  tag="${image//[:\/]/-}"
  [ -n "$PLATFORM" ] && tag="$tag-${PLATFORM##*/}"
  echo "=== $image ==="
  # shellcheck disable=SC2086  # PLAT_ARGS is intentionally word-split
  "$RT" run --rm $PLAT_ARGS \
    -v "$WS_ROOT:/src:ro" -v "$EXAMPLES:/w/examples:ro" \
    "$image" sh -c "
      set -e
      $(installer "$image")
      curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path >/dev/null
      export PATH=\$HOME/.cargo/bin:\$PATH
      mkdir -p /w/port
      # copy without target/ and .git rather than copying then deleting them:
      # target/ alone is several GB of release artefacts.
      tar -C /src --exclude=./target --exclude=./.git --exclude=./logs -cf - . | tar -C /w/port -xf -
      cd /w/port
      for f in tools/*.sh; do sed -i 's/\r\$//' \"\$f\"; done
      # musl's ldd writes its version to stderr, so 2>/dev/null blanked this
      # field on Alpine. 2>&1 keeps it for both libcs.
      echo \"--- \$(uname -m)$EMU / \$(ldd --version 2>&1 | head -1) / \$(rustc -V) ---\"
      cargo build --workspace 2>&1 | grep -E '^(warning|error)' | head -20 || true
      bash tools/verify_examples.sh all >/dev/null 2>&1 || true
      echo 'IDENTICAL:' \$(grep -c 'IDENTICAL\$' logs/summary.txt)
      echo 'DIFF:     ' \$(grep -c 'DIFF(' logs/summary.txt)
      echo 'EXCLUDED: ' \$(grep -c 'EXCLUDED' logs/summary.txt)
      echo 'FAIL:     ' \$(grep -c 'FAIL(' logs/summary.txt)
      echo '--- variants reported DIFF here ---'
      grep 'DIFF(' logs/summary.txt | awk '{print \$1, \$2}' | sort
    " 2>&1 | tee "$LOGS/gate-$tag.txt"

  # A container that cannot start still exits the pipeline cleanly and leaves
  # a log with an error in it instead of a tally. That happened: an earlier
  # `pull --platform linux/arm64 alpine:3.20` replaced the local image, the
  # next native run died with "Exec format error", and the empty result was
  # copied over good committed evidence. Refuse to pass in that state.
  if ! grep -q '^IDENTICAL:' "$LOGS/gate-$tag.txt"; then
    echo "FAILED: $image produced no gate result -- see $LOGS/gate-$tag.txt" >&2
    if grep -q 'Exec format error\|does not match the expected platform' "$LOGS/gate-$tag.txt"; then
      echo "  the local image is for another architecture; re-pull it:" >&2
      echo "    $RT pull --platform ${PLATFORM:-linux/amd64} $image" >&2
    fi
    rc=1
  fi
  echo
done
exit "${rc:-0}"
