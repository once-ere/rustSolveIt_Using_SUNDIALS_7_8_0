/* libm_oracle.c — reference generator for the pure-Rust elementary
 * functions in `crates/sundials_core/src/sundials_libm.rs`.
 *
 * Built with the *host* compiler and linked against the *host* C library,
 * so `sin`, `exp`, … below are whatever glibc this machine ships. For each
 * sampled input it writes a 4 x u64 little-endian record:
 *
 *     [0] input               x, as bits
 *     [1] host result         f(x) computed by the host libm
 *     [2] rounded reference   the correctly rounded binary64 nearest to the
 *                             113-bit __float128 value of f(x)
 *     [3] residual            (quad_value - reference) / ulp(reference),
 *                             a double in [-0.5, 0.5]; lets the Rust side
 *                             recover a real-valued ulp error for any
 *                             candidate result without needing quad itself
 *
 * Record [2] and [3] require libquadmath (ships with gcc). Without it,
 * compile with -DNO_QUADMATH: those two slots are then filled with the host
 * result and 0, and the Rust side reports agreement only, not accuracy.
 *
 * Usage:  libm_oracle <function> <corpus> <count> <outfile>
 * The corpus definitions here and in `sundials_libm.rs`'s test module are
 * *not* required to agree, because the inputs are transmitted in the file.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NO_QUADMATH
#include <quadmath.h>
#endif

/* ---------------- deterministic sample stream (splitmix64) ------------- */
static uint64_t g_state;

static uint64_t nxt(void)
{
  uint64_t z = (g_state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static double unit(void) { return (double)(nxt() >> 11) * 0x1p-53; }

/* ---------------- corpora ---------------------------------------------- */
/* kind 0: linear   x = lo + (hi-lo)*u
   kind 1: logspace x = +-(1+u) * 2^k, k uniform in [lo, hi]
   kind 2: logspace, positive only                                        */
struct spec {
  const char *name;
  int kind;
  double lo, hi;
};

static double sample(const struct spec *s)
{
  if (s->kind == 0) return s->lo + (s->hi - s->lo) * unit();
  int span = (int)(s->hi - s->lo) + 1;
  int k = (int)s->lo + (int)(nxt() % (uint64_t)span);
  double m = 1.0 + unit();
  double x = ldexp(m, k);
  if (s->kind == 1 && (nxt() & 1)) x = -x;
  return x;
}

/* ---------------- function table --------------------------------------- */
typedef double (*fn_d)(double);
#ifndef NO_QUADMATH
typedef __float128 (*fn_q)(__float128);
#endif

struct entry {
  const char *name;
  fn_d f;
#ifndef NO_QUADMATH
  fn_q q;
#endif
  struct spec domain;
  struct spec wide;
};

#ifndef NO_QUADMATH
#define E(n, f, q, dk, dlo, dhi, wk, wlo, whi) \
  {n, f, q, {n, dk, dlo, dhi}, {n, wk, wlo, whi}}
#else
#define E(n, f, q, dk, dlo, dhi, wk, wlo, whi) \
  {n, f, {n, dk, dlo, dhi}, {n, wk, wlo, whi}}
#endif

static const struct entry TABLE[] = {
    E("exp", exp, expq, 0, -745.0, 710.0, 0, -40.0, 40.0),
    E("log", log, logq, 2, -1060.0, 1020.0, 0, 1e-3, 1e3),
    E("expm1", expm1, expm1q, 0, -40.0, 40.0, 0, -1.0, 1.0),
    E("log1p", log1p, log1pq, 0, -0.9999, 10.0, 2, -60.0, 60.0),
    E("sin", sin, sinq, 0, -3.15, 3.15, 1, -20.0, 60.0),
    E("cos", cos, cosq, 0, -3.15, 3.15, 1, -20.0, 60.0),
    E("atan", atan, atanq, 0, -1.5, 1.5, 1, -30.0, 30.0),
    E("asin", asin, asinq, 0, -1.0, 1.0, 0, -0.02, 0.02),
    E("acos", acos, acosq, 0, -1.0, 1.0, 0, 0.98, 1.0),
    E("sinh", sinh, sinhq, 0, -710.0, 710.0, 0, -1.0, 1.0),
    E("cosh", cosh, coshq, 0, -710.0, 710.0, 0, -1.0, 1.0),
    E("acosh", acosh, acoshq, 0, 1.0, 1e6, 0, 1.0, 1.125),
};
#define NTABLE ((int)(sizeof(TABLE) / sizeof(TABLE[0])))

static uint64_t bits(double d)
{
  uint64_t u;
  memcpy(&u, &d, 8);
  return u;
}

int main(int argc, char **argv)
{
  if (argc != 5) {
    fprintf(stderr, "usage: %s <function> <domain|wide> <count> <outfile>\n",
            argv[0]);
    fprintf(stderr, "functions:");
    for (int i = 0; i < NTABLE; i++) fprintf(stderr, " %s", TABLE[i].name);
    fprintf(stderr, "\n");
    return 2;
  }
  const struct entry *e = NULL;
  for (int i = 0; i < NTABLE; i++)
    if (strcmp(TABLE[i].name, argv[1]) == 0) e = &TABLE[i];
  if (!e) {
    fprintf(stderr, "unknown function %s\n", argv[1]);
    return 2;
  }
  int wide = strcmp(argv[2], "wide") == 0;
  const struct spec *sp = wide ? &e->wide : &e->domain;
  long n = atol(argv[3]);

  /* Seed depends on the function name and the corpus, so no two streams
     coincide; the value itself is arbitrary but fixed. */
  g_state = 0x12345678ABCDEF01ULL + (uint64_t)(wide ? 0x5BD1E995u : 0u);
  for (const char *p = e->name; *p; p++) g_state = g_state * 1000003u + (unsigned char)*p;

  FILE *out = fopen(argv[4], "wb");
  if (!out) {
    perror(argv[4]);
    return 1;
  }
  for (long i = 0; i < n; i++) {
    double x = sample(sp);
    double got = e->f(x);
    uint64_t rec[4];
    rec[0] = bits(x);
    rec[1] = bits(got);
#ifndef NO_QUADMATH
    __float128 qv = e->q((__float128)x);
    double ref = (double)qv;
    double u = nextafter(ref, INFINITY) - ref;
    double resid = 0.0;
    if (u != 0.0 && isfinite(ref) && isfinite((double)qv)) {
      resid = (double)((qv - (__float128)ref) / (__float128)u);
    }
    rec[2] = bits(ref);
    rec[3] = bits(resid);
#else
    rec[2] = rec[1];
    rec[3] = bits(0.0);
#endif
    if (fwrite(rec, 8, 4, out) != 4) {
      perror("write");
      return 1;
    }
  }
  fclose(out);
  return 0;
}
