/* libm_probe.c — fingerprint a host's libm, one hash per function.
 *
 * SUNDIALS_7_8_Rust_port_for_Linux.
 *
 * The port's byte-identity claim rests on one platform property: the libm
 * behind `f64`'s transcendental methods must be the libm that generated
 * the upstream reference .out files. README.md § "Distribution coverage"
 * argues that this holds across the Debian, Arch and Fedora families
 * because they all ship glibc. This program turns that argument into a
 * measurement.
 *
 * For each function the library and the examples actually call, it
 * evaluates a deterministic corpus and prints an FNV-1a hash of the
 * result BIT PATTERNS. Two hosts whose hashes all match cannot disagree
 * about any of those results — so a verification gate established on one
 * of them transfers to the other. A single differing hash localises the
 * disagreement to that function.
 *
 * `sqrt` is included as a control: it is IEEE-754 specified and must
 * match everywhere, including on non-glibc hosts. If `sqrt` ever differs,
 * the harness is broken, not the libm.
 *
 *   cc -O2 -o libm_probe tools/libm_probe.c -lm && ./libm_probe
 *
 * Driven across distributions by tools/glibc_sweep.sh.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 1000000u

static uint64_t sm_state;

static uint64_t sm_next(void)
{
  uint64_t z = (sm_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static double unit(void) { return (double)(sm_next() >> 11) * 0x1p-53; }

static uint64_t fnv_init(void) { return 0xcbf29ce484222325ull; }

static uint64_t fnv_push(uint64_t h, double d)
{
  uint64_t b;
  memcpy(&b, &d, sizeof b);
  for (int i = 0; i < 8; i++)
  {
    h ^= (b >> (8 * i)) & 0xff;
    h *= 0x100000001b3ull;
  }
  return h;
}

/* Each entry: name, seed, and the domain the port actually reaches.
 * `lo`/`hi` bound a uniform draw; `pow` is handled separately because it
 * takes two arguments and its y is a reciprocal of an integrator order. */
struct probe
{
  const char *name;
  uint64_t seed;
  double lo, hi;
  double (*f)(double);
};

static const struct probe PROBES[] = {
  {"sqrt",  11, 0.0, 1e6, sqrt},   /* IEEE-754 control — must match everywhere */
  {"exp",   12, -40.0, 40.0, exp},
  {"log",   13, 1e-12, 1e3, log},
  {"sin",   14, -100.0, 100.0, sin},
  {"cos",   15, -100.0, 100.0, cos},
  {"asin",  16, -1.0, 1.0, asin},
  {"acos",  17, -1.0, 1.0, acos},
  {"atan",  18, -100.0, 100.0, atan},
  {"sinh",  19, -20.0, 20.0, sinh},
  {"cosh",  20, -20.0, 20.0, cosh},
  {"acosh", 21, 1.0, 100.0, acosh},
};

int main(void)
{
  /* pow over the SUNDIALS operating domain — same construction as
   * tools/pow_oracle.c's `domain` corpus. */
  sm_state = 1;
  uint64_t h = fnv_init();
  for (unsigned i = 0; i < N; i++)
  {
    double x = unit() * 100.0;
    if (x == 0.0) { x = 100.0; }
    uint64_t s = sm_next();
    double y;
    if (s % 14 == 0) { y = unit() * 2.0 - 1.0; }
    else
    {
      y = 1.0 / (double)((s % 13) + 1);
      if (s & 0x100) { y = -y; }
    }
    h = fnv_push(h, pow(x, y));
  }
  printf("%-6s %016llx\n", "pow", (unsigned long long)h);

  for (unsigned p = 0; p < sizeof PROBES / sizeof PROBES[0]; p++)
  {
    sm_state = PROBES[p].seed;
    h = fnv_init();
    for (unsigned i = 0; i < N; i++)
    {
      double x = PROBES[p].lo + unit() * (PROBES[p].hi - PROBES[p].lo);
      h = fnv_push(h, PROBES[p].f(x));
    }
    printf("%-6s %016llx\n", PROBES[p].name, (unsigned long long)h);
  }
  return 0;
}
