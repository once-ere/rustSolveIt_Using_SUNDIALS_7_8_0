/* pow_oracle.c — native glibc/x86-64 reference generator for the
 * deterministic `pow` in crates/sundials_core/src/sundials_math.rs.
 *
 * SUNDIALS_7_8_Rust_port_for_Linux.  Build and run on the *target* host
 * (Linux, glibc, Intel/AMD x86-64); this program is the oracle the Rust
 * routine is measured against, so it must never be cross-built.
 *
 *   cc -O2 -o pow_oracle tools/pow_oracle.c -lm
 *   ./pow_oracle domain > /tmp/pow_domain.bin      # 5,900,000 results
 *   ./pow_oracle random > /tmp/pow_random.bin      # 20,000,000 results
 *
 * It calls the host's `pow` — on glibc >= 2.28 / x86-64 that is the ifunc
 * dispatch in sysdeps/x86_64/fpu/multiarch/e_pow.c, which selects
 * __ieee754_pow_fma (built from the same source with -mfma -mavx2
 * -ffp-contract=fast) whenever the CPU reports FMA.  That FMA-contracted
 * build is exactly the target the Rust port reproduces, which is why the
 * measurement is only meaningful when made here rather than on arm64.
 *
 * Output is a raw little-endian stream of the IEEE-754 bit patterns of
 * pow(x, y), one uint64_t per corpus element, in corpus order.  The
 * corpus itself is *not* transmitted: both sides regenerate it from the
 * same splitmix64 recurrence, so the two programs cannot disagree about
 * which inputs they evaluated.  Keep the two generators in lockstep — the
 * Rust twin is `pow_corpus` in sundials_math.rs's test module.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N_DOMAIN 5900000u
#define N_RANDOM 20000000u

static uint64_t sm_state;

static uint64_t sm_next(void)
{
  uint64_t z = (sm_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static double bits_to_f64(uint64_t b)
{
  double d;
  memcpy(&d, &b, sizeof d);
  return d;
}

static uint64_t f64_to_bits(double d)
{
  uint64_t b;
  memcpy(&b, &d, sizeof b);
  return b;
}

/* [0,1) from the top 53 bits, the usual construction. */
static double unit(uint64_t u) { return (double)(u >> 11) * 0x1p-53; }

/* Corpus A — the domain SUNDIALS actually evaluates.  SUNRpowerR is called
 * from the step-size controllers as pow(bias*dsm, +-1/order), so x lies in
 * (0, ~100] and |y| <= 1, with y overwhelmingly a reciprocal of an
 * integrator order 1..13. */
static void domain_pair(double *x, double *y)
{
  double xv = unit(sm_next()) * 100.0;
  if (xv == 0.0) { xv = 100.0; }
  uint64_t s = sm_next();
  double yv;
  if (s % 14 == 0)
  {
    yv = unit(sm_next()) * 2.0 - 1.0;
  }
  else
  {
    yv = 1.0 / (double)((s % 13) + 1);
    if (s & 0x100) { yv = -yv; }
  }
  *x = xv;
  *y = yv;
}

/* Corpus B — unrestricted finite bit patterns, far outside the operating
 * domain; used to bound the residual disagreement, not to gate anything. */
static void random_pair(double *x, double *y)
{
  for (;;)
  {
    double xv = bits_to_f64(sm_next());
    double yv = bits_to_f64(sm_next());
    if (isfinite(xv) && isfinite(yv))
    {
      *x = xv;
      *y = yv;
      return;
    }
  }
}

int main(int argc, char **argv)
{
  int domain = (argc > 1 && strcmp(argv[1], "random") == 0) ? 0 : 1;
  unsigned n = domain ? N_DOMAIN : N_RANDOM;
  sm_state = domain ? 1ull : 2ull;

  for (unsigned i = 0; i < n; i++)
  {
    double x, y;
    if (domain) { domain_pair(&x, &y); }
    else { random_pair(&x, &y); }
    uint64_t r = f64_to_bits(pow(x, y));
    if (fwrite(&r, sizeof r, 1, stdout) != 1) { return 1; }
  }
  return 0;
}
