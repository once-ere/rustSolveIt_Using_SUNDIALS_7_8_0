//! Port of `src/sundials/sundials_math.c` + `include/sundials/sundials_math.h`
//! (double-precision branch).

use crate::sundials_libm::SunMath;
use crate::sundials_types::*;

/// C macro `SUNMIN(A,B)`: `((A) < (B) ? (A) : (B))`.
pub fn SUNMIN<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

/// C macro `SUNMAX(A,B)`: `((A) > (B) ? (A) : (B))`.
pub fn SUNMAX<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

/// C macro `SUNSQR(A)`: `((A) * (A))`.
pub fn SUNSQR(a: sunrealtype) -> sunrealtype {
    a * a
}

/// C macro `SUNRsqrt(x)`: 0 for `x <= 0`, else `sqrt(x)`.
pub fn SUNRsqrt(x: sunrealtype) -> sunrealtype {
    if x <= 0.0 {
        0.0
    } else {
        x.sqrt()
    }
}

/// C macro `SUNRabs(x)`: `fabs(x)`.
pub fn SUNRabs(x: sunrealtype) -> sunrealtype {
    x.abs()
}

/// C macro `SUNRexp(x)`: `exp(x)`.
pub fn SUNRexp(x: sunrealtype) -> sunrealtype {
    x.sun_exp()
}

/// C macro `SUNRceil(x)`: `ceil(x)`.
pub fn SUNRceil(x: sunrealtype) -> sunrealtype {
    x.ceil()
}

/// C macro `SUNRround(x)`: `round(x)` (halfway cases away from zero).
pub fn SUNRround(x: sunrealtype) -> sunrealtype {
    x.round()
}

/// C macro `SUNRcopysign(x, y)`: `copysign(x, y)`.
pub fn SUNRcopysign(x: sunrealtype, y: sunrealtype) -> sunrealtype {
    x.copysign(y)
}

/// C macro `SUNRsamesign(x, y)`: `signbit(x) == signbit(y)`.
pub fn SUNRsamesign(x: sunrealtype, y: sunrealtype) -> sunbooleantype {
    x.is_sign_negative() == y.is_sign_negative()
}

/// C macro `SUNRdifferentsign(x, y)`: `!SUNRsamesign(x, y)`.
pub fn SUNRdifferentsign(x: sunrealtype, y: sunrealtype) -> sunbooleantype {
    !SUNRsamesign(x, y)
}

/// C macro `SUNRpowerR(base, exponent)`: `pow(base, exponent)`.
///
/// Uses the deterministic `pow` implementation below instead of
/// `f64::powf` (platform libm): reference `.out` files were generated
/// against glibc, whose `pow` (>= 2.28) is the ARM optimized-routines
/// algorithm. Platform libms disagree with it by 1 ulp on rare inputs
/// (e.g. macOS libm returns 14 incorrectly rounded results over the
/// 6174 `pow` calls of `cvVdp_auto_nls`), which flips marginal
/// step-size/error-test decisions and breaks byte-identical output.
///
/// **Scope.** This makes `pow` — and only `pow` — independent of the host
/// libm. `SUNRexp` (below), `arkode_lsrkstep`'s `SUNRlog`/`SUNRsinh`/
/// `SUNRcosh`/`SUNRacosh`, and every `sin`/`cos`/`asin`/`acos`/`atan`/`exp`/
/// `ln` in the examples still resolve to the host's libm through `f64`'s
/// unspecified-precision methods. On this port's target — Linux on x86-64
/// with glibc — that host libm *is* the one the upstream reference outputs
/// were generated with, which is why byte-identical output is claimed here
/// and scoped to glibc (see `README.md` § "Platform scope"). The `pow`
/// routine below is measured bit-exact against the native glibc `pow` by
/// `tools/pow_differential.sh`; see `POW_FMA_EXACTNESS.md` §0.
/// `SUNRsqrt` is *not* in that set: `f64::sqrt` is IEEE-754 `squareRoot`,
/// correctly rounded and identical on every target — as are `SUNRceil`,
/// `SUNRround`, `SUNRabs`, `SUNRcopysign` and `f64::mul_add`. The routine
/// below is portable Rust and carries no architecture assumption.
pub fn SUNRpowerR(base: sunrealtype, exponent: sunrealtype) -> sunrealtype {
    pow_glibc(base, exponent)
}

pub fn SUNIpowerI(base: i32, exponent: i32) -> i32 {
    let mut prod: i32 = 1;
    let mut i = 1;
    while i <= exponent {
        prod *= base;
        i += 1;
    }
    prod
}

pub fn SUNRpowerI(base: sunrealtype, exponent: i32) -> sunrealtype {
    let mut prod: sunrealtype = 1.0;
    let expt = exponent.abs();
    let mut i = 1;
    while i <= expt {
        prod *= base;
        i += 1;
    }
    if exponent < 0 {
        prod = 1.0 / prod;
    }
    prod
}

pub fn SUNRCompare(a: sunrealtype, b: sunrealtype) -> sunbooleantype {
    SUNRCompareTol(a, b, 10.0 * SUN_UNIT_ROUNDOFF)
}

pub fn SUNRCompareTol(a: sunrealtype, b: sunrealtype, tol: sunrealtype) -> sunbooleantype {
    /* If a and b are exactly equal.
     * This also covers the case where a and b are both inf under IEEE 754. */
    if a == b {
        return SUNFALSE;
    }

    let diff = SUNRabs(a - b);
    let norm = SUNMIN(SUNRabs(a + b), SUN_BIG_REAL);

    /* C uses !isless(diff, max) so NaNs compare "not equal" (true);
     * Rust `!(diff < max)` has identical semantics for NaN. */
    !(diff < SUNMAX(10.0 * SUN_UNIT_ROUNDOFF, tol * norm))
}

/// C `SUNStrToReal`: `strtod` semantics — parse the longest valid leading
/// float, ignoring leading whitespace and trailing junk; 0.0 if nothing
/// parses.
pub fn SUNStrToReal(str_: &str) -> sunrealtype {
    /* strtod skips C-locale (ASCII) whitespace only */
    let s = str_.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r']);
    let b = s.as_bytes();
    let mut i = 0usize;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let lower = s[i..].to_ascii_lowercase();
    if lower.starts_with("infinity") {
        return s[..i + 8].parse::<f64>().unwrap_or(0.0);
    }
    if lower.starts_with("inf") {
        return s[..i + 3].parse::<f64>().unwrap_or(0.0);
    }
    if lower.starts_with("nan") {
        return f64::NAN;
    }
    let start_digits = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    // at least one digit must appear in the mantissa
    if i == start_digits || !s[start_digits..i].bytes().any(|c| c.is_ascii_digit()) {
        return 0.0;
    }
    let mantissa_end = i;
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let exp_digits = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_digits {
            i = j;
        } else {
            i = mantissa_end;
        }
    }
    s[..i].parse::<f64>().unwrap_or(0.0)
}

/* =====================================================================
 * Deterministic double-precision pow
 * =====================================================================
 * Port of the ARM optimized-routines `pow` (via musl `src/math/pow.c`,
 * Copyright (c) 2018, Arm Limited, SPDX-License-Identifier: MIT) — the
 * same algorithm and tables as glibc >= 2.28
 * (`sysdeps/ieee754/dbl-64/e_pow.c`), i.e. the libm that produced the
 * upstream reference outputs. Worst-case error 0.54 ulp. Config mirrors
 * the x86-64 glibc build: TOINT_INTRINSICS=0, EXP_USE_TOINT_NARROW=0,
 * WANT_ROUNDING=1. The three explicit fused multiply-adds of the C are
 * `f64::mul_add` (guaranteed fused in Rust); every other expression
 * keeps the exact C evaluation order and is never contracted.
 *
 * Portability: this is plain `f64` arithmetic plus `f64::mul_add`, which is
 * fused and correctly rounded on every Rust target, and Rust never contracts
 * arithmetic on its own — so the routine carries no architecture assumption
 * despite the "ARM optimized-routines" provenance. On x86-64 that provenance
 * is in fact the target: glibc ifunc-dispatches `pow` to `__ieee754_pow_fma`,
 * this same source rebuilt with `-mfma -mavx2 -ffp-contract=fast`, and the
 * FMA-contraction map below reproduces that build. Measured against the
 * native glibc `pow` on Linux/x86-64: 0 mismatches over 5,900,000 inputs in
 * the domain SUNDIALS evaluates and 0 over 20,000,000 unrestricted finite
 * inputs (`tools/pow_differential.sh`; POW_FMA_EXACTNESS.md §0). The map
 * itself was originally derived against oracle binaries built on arm64;
 * §§1–7 of that document record how.
 */

const POW_LOG_LN2HI: u64 = 0x3fe62e42fefa3800;
const POW_LOG_LN2LO: u64 = 0x3d2ef35793c76730;
const POW_LOG_POLY: [u64; 7] = [
    0xbfe0000000000000,
    0xbfe5555555555560,
    0x3fe0000000000006,
    0x3fe999999959554e,
    0xbfe555555529a47a,
    0xbff2495b9b4845e9,
    0x3ff0002b8b263fc3,
];
/* (invc, logc, logctail) */
const POW_LOG_TAB: [(u64, u64, u64); 128] = [
    (0x3ff6a00000000000, 0xbfd62c82f2b9c800, 0x3cfab42428375680),
    (0x3ff6800000000000, 0xbfd5d1bdbf580800, 0xbd1ca508d8e0f720),
    (0x3ff6600000000000, 0xbfd5767717455800, 0xbd2362a4d5b6506d),
    (0x3ff6400000000000, 0xbfd51aad872df800, 0xbce684e49eb067d5),
    (0x3ff6200000000000, 0xbfd4be5f95777800, 0xbd041b6993293ee0),
    (0x3ff6000000000000, 0xbfd4618bc21c6000, 0x3d13d82f484c84cc),
    (0x3ff5e00000000000, 0xbfd404308686a800, 0x3cdc42f3ed820b3a),
    (0x3ff5c00000000000, 0xbfd3a64c55694800, 0x3d20b1c686519460),
    (0x3ff5a00000000000, 0xbfd347dd9a988000, 0x3d25594dd4c58092),
    (0x3ff5800000000000, 0xbfd2e8e2bae12000, 0x3d267b1e99b72bd8),
    (0x3ff5600000000000, 0xbfd2895a13de8800, 0x3d15ca14b6cfb03f),
    (0x3ff5600000000000, 0xbfd2895a13de8800, 0x3d15ca14b6cfb03f),
    (0x3ff5400000000000, 0xbfd22941fbcf7800, 0xbd165a242853da76),
    (0x3ff5200000000000, 0xbfd1c898c1699800, 0xbd1fafbc68e75404),
    (0x3ff5000000000000, 0xbfd1675cababa800, 0x3d1f1fc63382a8f0),
    (0x3ff4e00000000000, 0xbfd1058bf9ae4800, 0xbd26a8c4fd055a66),
    (0x3ff4c00000000000, 0xbfd0a324e2739000, 0xbd0c6bee7ef4030e),
    (0x3ff4a00000000000, 0xbfd0402594b4d000, 0xbcf036b89ef42d7f),
    (0x3ff4a00000000000, 0xbfd0402594b4d000, 0xbcf036b89ef42d7f),
    (0x3ff4800000000000, 0xbfcfb9186d5e4000, 0x3d0d572aab993c87),
    (0x3ff4600000000000, 0xbfcef0adcbdc6000, 0x3d2b26b79c86af24),
    (0x3ff4400000000000, 0xbfce27076e2af000, 0xbd172f4f543fff10),
    (0x3ff4200000000000, 0xbfcd5c216b4fc000, 0x3d21ba91bbca681b),
    (0x3ff4000000000000, 0xbfcc8ff7c79aa000, 0x3d27794f689f8434),
    (0x3ff4000000000000, 0xbfcc8ff7c79aa000, 0x3d27794f689f8434),
    (0x3ff3e00000000000, 0xbfcbc286742d9000, 0x3d194eb0318bb78f),
    (0x3ff3c00000000000, 0xbfcaf3c94e80c000, 0x3cba4e633fcd9066),
    (0x3ff3a00000000000, 0xbfca23bc1fe2b000, 0xbd258c64dc46c1ea),
    (0x3ff3a00000000000, 0xbfca23bc1fe2b000, 0xbd258c64dc46c1ea),
    (0x3ff3800000000000, 0xbfc9525a9cf45000, 0xbd2ad1d904c1d4e3),
    (0x3ff3600000000000, 0xbfc87fa06520d000, 0x3d2bbdbf7fdbfa09),
    (0x3ff3400000000000, 0xbfc7ab890210e000, 0x3d2bdb9072534a58),
    (0x3ff3400000000000, 0xbfc7ab890210e000, 0x3d2bdb9072534a58),
    (0x3ff3200000000000, 0xbfc6d60fe719d000, 0xbd10e46aa3b2e266),
    (0x3ff3000000000000, 0xbfc5ff3070a79000, 0xbd1e9e439f105039),
    (0x3ff3000000000000, 0xbfc5ff3070a79000, 0xbd1e9e439f105039),
    (0x3ff2e00000000000, 0xbfc526e5e3a1b000, 0xbd20de8b90075b8f),
    (0x3ff2c00000000000, 0xbfc44d2b6ccb8000, 0x3d170cc16135783c),
    (0x3ff2c00000000000, 0xbfc44d2b6ccb8000, 0x3d170cc16135783c),
    (0x3ff2a00000000000, 0xbfc371fc201e9000, 0x3cf178864d27543a),
    (0x3ff2800000000000, 0xbfc29552f81ff000, 0xbd248d301771c408),
    (0x3ff2600000000000, 0xbfc1b72ad52f6000, 0xbd2e80a41811a396),
    (0x3ff2600000000000, 0xbfc1b72ad52f6000, 0xbd2e80a41811a396),
    (0x3ff2400000000000, 0xbfc0d77e7cd09000, 0x3d0a699688e85bf4),
    (0x3ff2400000000000, 0xbfc0d77e7cd09000, 0x3d0a699688e85bf4),
    (0x3ff2200000000000, 0xbfbfec9131dbe000, 0xbd2575545ca333f2),
    (0x3ff2000000000000, 0xbfbe27076e2b0000, 0x3d2a342c2af0003c),
    (0x3ff2000000000000, 0xbfbe27076e2b0000, 0x3d2a342c2af0003c),
    (0x3ff1e00000000000, 0xbfbc5e548f5bc000, 0xbd1d0c57585fbe06),
    (0x3ff1c00000000000, 0xbfba926d3a4ae000, 0x3d253935e85baac8),
    (0x3ff1c00000000000, 0xbfba926d3a4ae000, 0x3d253935e85baac8),
    (0x3ff1a00000000000, 0xbfb8c345d631a000, 0x3d137c294d2f5668),
    (0x3ff1a00000000000, 0xbfb8c345d631a000, 0x3d137c294d2f5668),
    (0x3ff1800000000000, 0xbfb6f0d28ae56000, 0xbd269737c93373da),
    (0x3ff1600000000000, 0xbfb51b073f062000, 0x3d1f025b61c65e57),
    (0x3ff1600000000000, 0xbfb51b073f062000, 0x3d1f025b61c65e57),
    (0x3ff1400000000000, 0xbfb341d7961be000, 0x3d2c5edaccf913df),
    (0x3ff1400000000000, 0xbfb341d7961be000, 0x3d2c5edaccf913df),
    (0x3ff1200000000000, 0xbfb16536eea38000, 0x3d147c5e768fa309),
    (0x3ff1000000000000, 0xbfaf0a30c0118000, 0x3d2d599e83368e91),
    (0x3ff1000000000000, 0xbfaf0a30c0118000, 0x3d2d599e83368e91),
    (0x3ff0e00000000000, 0xbfab42dd71198000, 0x3d1c827ae5d6704c),
    (0x3ff0e00000000000, 0xbfab42dd71198000, 0x3d1c827ae5d6704c),
    (0x3ff0c00000000000, 0xbfa77458f632c000, 0xbd2cfc4634f2a1ee),
    (0x3ff0c00000000000, 0xbfa77458f632c000, 0xbd2cfc4634f2a1ee),
    (0x3ff0a00000000000, 0xbfa39e87b9fec000, 0x3cf502b7f526feaa),
    (0x3ff0a00000000000, 0xbfa39e87b9fec000, 0x3cf502b7f526feaa),
    (0x3ff0800000000000, 0xbf9f829b0e780000, 0xbd2980267c7e09e4),
    (0x3ff0800000000000, 0xbf9f829b0e780000, 0xbd2980267c7e09e4),
    (0x3ff0600000000000, 0xbf97b91b07d58000, 0xbd288d5493faa639),
    (0x3ff0400000000000, 0xbf8fc0a8b0fc0000, 0xbcdf1e7cf6d3a69c),
    (0x3ff0400000000000, 0xbf8fc0a8b0fc0000, 0xbcdf1e7cf6d3a69c),
    (0x3ff0200000000000, 0xbf7fe02a6b100000, 0xbd19e23f0dda40e4),
    (0x3ff0200000000000, 0xbf7fe02a6b100000, 0xbd19e23f0dda40e4),
    (0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000),
    (0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000),
    (0x3fefc00000000000, 0x3f80101575890000, 0xbd10c76b999d2be8),
    (0x3fef800000000000, 0x3f90205658938000, 0xbd23dc5b06e2f7d2),
    (0x3fef400000000000, 0x3f98492528c90000, 0xbd2aa0ba325a0c34),
    (0x3fef000000000000, 0x3fa0415d89e74000, 0x3d0111c05cf1d753),
    (0x3feec00000000000, 0x3fa466aed42e0000, 0xbd2c167375bdfd28),
    (0x3fee800000000000, 0x3fa894aa149fc000, 0xbd197995d05a267d),
    (0x3fee400000000000, 0x3faccb73cdddc000, 0xbd1a68f247d82807),
    (0x3fee200000000000, 0x3faeea31c006c000, 0xbd0e113e4fc93b7b),
    (0x3fede00000000000, 0x3fb1973bd1466000, 0xbd25325d560d9e9b),
    (0x3feda00000000000, 0x3fb3bdf5a7d1e000, 0x3d2cc85ea5db4ed7),
    (0x3fed600000000000, 0x3fb5e95a4d97a000, 0xbd2c69063c5d1d1e),
    (0x3fed400000000000, 0x3fb700d30aeac000, 0x3cec1e8da99ded32),
    (0x3fed000000000000, 0x3fb9335e5d594000, 0x3d23115c3abd47da),
    (0x3fecc00000000000, 0x3fbb6ac88dad6000, 0xbd1390802bf768e5),
    (0x3feca00000000000, 0x3fbc885801bc4000, 0x3d2646d1c65aacd3),
    (0x3fec600000000000, 0x3fbec739830a2000, 0xbd2dc068afe645e0),
    (0x3fec400000000000, 0x3fbfe89139dbe000, 0xbd2534d64fa10afd),
    (0x3fec000000000000, 0x3fc1178e8227e000, 0x3d21ef78ce2d07f2),
    (0x3febe00000000000, 0x3fc1aa2b7e23f000, 0x3d2ca78e44389934),
    (0x3feba00000000000, 0x3fc2d1610c868000, 0x3d039d6ccb81b4a1),
    (0x3feb800000000000, 0x3fc365fcb0159000, 0x3cc62fa8234b7289),
    (0x3feb400000000000, 0x3fc4913d8333b000, 0x3d25837954fdb678),
    (0x3feb200000000000, 0x3fc527e5e4a1b000, 0x3d2633e8e5697dc7),
    (0x3feae00000000000, 0x3fc6574ebe8c1000, 0x3d19cf8b2c3c2e78),
    (0x3feac00000000000, 0x3fc6f0128b757000, 0xbd25118de59c21e1),
    (0x3feaa00000000000, 0x3fc7898d85445000, 0xbd1c661070914305),
    (0x3fea600000000000, 0x3fc8beafeb390000, 0xbd073d54aae92cd1),
    (0x3fea400000000000, 0x3fc95a5adcf70000, 0x3d07f22858a0ff6f),
    (0x3fea000000000000, 0x3fca93ed3c8ae000, 0xbd28724350562169),
    (0x3fe9e00000000000, 0x3fcb31d8575bd000, 0xbd0c358d4eace1aa),
    (0x3fe9c00000000000, 0x3fcbd087383be000, 0xbd2d4bc4595412b6),
    (0x3fe9a00000000000, 0x3fcc6ffbc6f01000, 0xbcf1ec72c5962bd2),
    (0x3fe9600000000000, 0x3fcdb13db0d49000, 0xbd2aff2af715b035),
    (0x3fe9400000000000, 0x3fce530effe71000, 0x3cc212276041f430),
    (0x3fe9200000000000, 0x3fcef5ade4dd0000, 0xbcca211565bb8e11),
    (0x3fe9000000000000, 0x3fcf991c6cb3b000, 0x3d1bcbecca0cdf30),
    (0x3fe8c00000000000, 0x3fd07138604d5800, 0x3cf89cdb16ed4e91),
    (0x3fe8a00000000000, 0x3fd0c42d67616000, 0x3d27188b163ceae9),
    (0x3fe8800000000000, 0x3fd1178e8227e800, 0xbd2c210e63a5f01c),
    (0x3fe8600000000000, 0x3fd16b5ccbacf800, 0x3d2b9acdf7a51681),
    (0x3fe8400000000000, 0x3fd1bf99635a6800, 0x3d2ca6ed5147bdb7),
    (0x3fe8200000000000, 0x3fd214456d0eb800, 0x3d0a87deba46baea),
    (0x3fe7e00000000000, 0x3fd2bef07cdc9000, 0x3d2a9cfa4a5004f4),
    (0x3fe7c00000000000, 0x3fd314f1e1d36000, 0xbd28e27ad3213cb8),
    (0x3fe7a00000000000, 0x3fd36b6776be1000, 0x3d116ecdb0f177c8),
    (0x3fe7800000000000, 0x3fd3c25277333000, 0x3d183b54b606bd5c),
    (0x3fe7600000000000, 0x3fd419b423d5e800, 0x3d08e436ec90e09d),
    (0x3fe7400000000000, 0x3fd4718dc271c800, 0xbd2f27ce0967d675),
    (0x3fe7200000000000, 0x3fd4c9e09e173000, 0xbd2e20891b0ad8a4),
    (0x3fe7000000000000, 0x3fd522ae0738a000, 0x3d2ebe708164c759),
    (0x3fe6e00000000000, 0x3fd57bf753c8d000, 0x3d1fadedee5d40ef),
    (0x3fe6c00000000000, 0x3fd5d5bddf596000, 0xbd0a0b2a08a465dc),
];
pub(crate) const EXP_INVLN2N: u64 = 0x40671547652b82fe;
pub(crate) const EXP_SHIFT: u64 = 0x4338000000000000;
pub(crate) const EXP_NEGLN2HIN: u64 = 0xbf762e42fefa0000;
pub(crate) const EXP_NEGLN2LON: u64 = 0xbd0cf79abc9e3b3a;
pub(crate) const EXP_POLY: [u64; 4] = [
    0x3fdffffffffffdbd,
    0x3fc555555555543c,
    0x3fa55555cf172b91,
    0x3f81111167a4d017,
];
pub(crate) const EXP_TAB: [u64; 256] = [
    0x0000000000000000, 0x3ff0000000000000, 0x3c9b3b4f1a88bf6e, 0x3feff63da9fb3335,
    0xbc7160139cd8dc5d, 0x3fefec9a3e778061, 0xbc905e7a108766d1, 0x3fefe315e86e7f85,
    0x3c8cd2523567f613, 0x3fefd9b0d3158574, 0xbc8bce8023f98efa, 0x3fefd06b29ddf6de,
    0x3c60f74e61e6c861, 0x3fefc74518759bc8, 0x3c90a3e45b33d399, 0x3fefbe3ecac6f383,
    0x3c979aa65d837b6d, 0x3fefb5586cf9890f, 0x3c8eb51a92fdeffc, 0x3fefac922b7247f7,
    0x3c3ebe3d702f9cd1, 0x3fefa3ec32d3d1a2, 0xbc6a033489906e0b, 0x3fef9b66affed31b,
    0xbc9556522a2fbd0e, 0x3fef9301d0125b51, 0xbc5080ef8c4eea55, 0x3fef8abdc06c31cc,
    0xbc91c923b9d5f416, 0x3fef829aaea92de0, 0x3c80d3e3e95c55af, 0x3fef7a98c8a58e51,
    0xbc801b15eaa59348, 0x3fef72b83c7d517b, 0xbc8f1ff055de323d, 0x3fef6af9388c8dea,
    0x3c8b898c3f1353bf, 0x3fef635beb6fcb75, 0xbc96d99c7611eb26, 0x3fef5be084045cd4,
    0x3c9aecf73e3a2f60, 0x3fef54873168b9aa, 0xbc8fe782cb86389d, 0x3fef4d5022fcd91d,
    0x3c8a6f4144a6c38d, 0x3fef463b88628cd6, 0x3c807a05b0e4047d, 0x3fef3f49917ddc96,
    0x3c968efde3a8a894, 0x3fef387a6e756238, 0x3c875e18f274487d, 0x3fef31ce4fb2a63f,
    0x3c80472b981fe7f2, 0x3fef2b4565e27cdd, 0xbc96b87b3f71085e, 0x3fef24dfe1f56381,
    0x3c82f7e16d09ab31, 0x3fef1e9df51fdee1, 0xbc3d219b1a6fbffa, 0x3fef187fd0dad990,
    0x3c8b3782720c0ab4, 0x3fef1285a6e4030b, 0x3c6e149289cecb8f, 0x3fef0cafa93e2f56,
    0x3c834d754db0abb6, 0x3fef06fe0a31b715, 0x3c864201e2ac744c, 0x3fef0170fc4cd831,
    0x3c8fdd395dd3f84a, 0x3feefc08b26416ff, 0xbc86a3803b8e5b04, 0x3feef6c55f929ff1,
    0xbc924aedcc4b5068, 0x3feef1a7373aa9cb, 0xbc9907f81b512d8e, 0x3feeecae6d05d866,
    0xbc71d1e83e9436d2, 0x3feee7db34e59ff7, 0xbc991919b3ce1b15, 0x3feee32dc313a8e5,
    0x3c859f48a72a4c6d, 0x3feedea64c123422, 0xbc9312607a28698a, 0x3feeda4504ac801c,
    0xbc58a78f4817895b, 0x3feed60a21f72e2a, 0xbc7c2c9b67499a1b, 0x3feed1f5d950a897,
    0x3c4363ed60c2ac11, 0x3feece086061892d, 0x3c9666093b0664ef, 0x3feeca41ed1d0057,
    0x3c6ecce1daa10379, 0x3feec6a2b5c13cd0, 0x3c93ff8e3f0f1230, 0x3feec32af0d7d3de,
    0x3c7690cebb7aafb0, 0x3feebfdad5362a27, 0x3c931dbdeb54e077, 0x3feebcb299fddd0d,
    0xbc8f94340071a38e, 0x3feeb9b2769d2ca7, 0xbc87deccdc93a349, 0x3feeb6daa2cf6642,
    0xbc78dec6bd0f385f, 0x3feeb42b569d4f82, 0xbc861246ec7b5cf6, 0x3feeb1a4ca5d920f,
    0x3c93350518fdd78e, 0x3feeaf4736b527da, 0x3c7b98b72f8a9b05, 0x3feead12d497c7fd,
    0x3c9063e1e21c5409, 0x3feeab07dd485429, 0x3c34c7855019c6ea, 0x3feea9268a5946b7,
    0x3c9432e62b64c035, 0x3feea76f15ad2148, 0xbc8ce44a6199769f, 0x3feea5e1b976dc09,
    0xbc8c33c53bef4da8, 0x3feea47eb03a5585, 0xbc845378892be9ae, 0x3feea34634ccc320,
    0xbc93cedd78565858, 0x3feea23882552225, 0x3c5710aa807e1964, 0x3feea155d44ca973,
    0xbc93b3efbf5e2228, 0x3feea09e667f3bcd, 0xbc6a12ad8734b982, 0x3feea012750bdabf,
    0xbc6367efb86da9ee, 0x3fee9fb23c651a2f, 0xbc80dc3d54e08851, 0x3fee9f7df9519484,
    0xbc781f647e5a3ecf, 0x3fee9f75e8ec5f74, 0xbc86ee4ac08b7db0, 0x3fee9f9a48a58174,
    0xbc8619321e55e68a, 0x3fee9feb564267c9, 0x3c909ccb5e09d4d3, 0x3feea0694fde5d3f,
    0xbc7b32dcb94da51d, 0x3feea11473eb0187, 0x3c94ecfd5467c06b, 0x3feea1ed0130c132,
    0x3c65ebe1abd66c55, 0x3feea2f336cf4e62, 0xbc88a1c52fb3cf42, 0x3feea427543e1a12,
    0xbc9369b6f13b3734, 0x3feea589994cce13, 0xbc805e843a19ff1e, 0x3feea71a4623c7ad,
    0xbc94d450d872576e, 0x3feea8d99b4492ed, 0x3c90ad675b0e8a00, 0x3feeaac7d98a6699,
    0x3c8db72fc1f0eab4, 0x3feeace5422aa0db, 0xbc65b6609cc5e7ff, 0x3feeaf3216b5448c,
    0x3c7bf68359f35f44, 0x3feeb1ae99157736, 0xbc93091fa71e3d83, 0x3feeb45b0b91ffc6,
    0xbc5da9b88b6c1e29, 0x3feeb737b0cdc5e5, 0xbc6c23f97c90b959, 0x3feeba44cbc8520f,
    0xbc92434322f4f9aa, 0x3feebd829fde4e50, 0xbc85ca6cd7668e4b, 0x3feec0f170ca07ba,
    0x3c71affc2b91ce27, 0x3feec49182a3f090, 0x3c6dd235e10a73bb, 0x3feec86319e32323,
    0xbc87c50422622263, 0x3feecc667b5de565, 0x3c8b1c86e3e231d5, 0x3feed09bec4a2d33,
    0xbc91bbd1d3bcbb15, 0x3feed503b23e255d, 0x3c90cc319cee31d2, 0x3feed99e1330b358,
    0x3c8469846e735ab3, 0x3feede6b5579fdbf, 0xbc82dfcd978e9db4, 0x3feee36bbfd3f37a,
    0x3c8c1a7792cb3387, 0x3feee89f995ad3ad, 0xbc907b8f4ad1d9fa, 0x3feeee07298db666,
    0xbc55c3d956dcaeba, 0x3feef3a2b84f15fb, 0xbc90a40e3da6f640, 0x3feef9728de5593a,
    0xbc68d6f438ad9334, 0x3feeff76f2fb5e47, 0xbc91eee26b588a35, 0x3fef05b030a1064a,
    0x3c74ffd70a5fddcd, 0x3fef0c1e904bc1d2, 0xbc91bdfbfa9298ac, 0x3fef12c25bd71e09,
    0x3c736eae30af0cb3, 0x3fef199bdd85529c, 0x3c8ee3325c9ffd94, 0x3fef20ab5fffd07a,
    0x3c84e08fd10959ac, 0x3fef27f12e57d14b, 0x3c63cdaf384e1a67, 0x3fef2f6d9406e7b5,
    0x3c676b2c6c921968, 0x3fef3720dcef9069, 0xbc808a1883ccb5d2, 0x3fef3f0b555dc3fa,
    0xbc8fad5d3ffffa6f, 0x3fef472d4a07897c, 0xbc900dae3875a949, 0x3fef4f87080d89f2,
    0x3c74a385a63d07a7, 0x3fef5818dcfba487, 0xbc82919e2040220f, 0x3fef60e316c98398,
    0x3c8e5a50d5c192ac, 0x3fef69e603db3285, 0x3c843a59ac016b4b, 0x3fef7321f301b460,
    0xbc82d52107b43e1f, 0x3fef7c97337b9b5f, 0xbc892ab93b470dc9, 0x3fef864614f5a129,
    0x3c74b604603a88d3, 0x3fef902ee78b3ff6, 0x3c83c5ec519d7271, 0x3fef9a51fbc74c83,
    0xbc8ff7128fd391f0, 0x3fefa4afa2a490da, 0xbc8dae98e223747d, 0x3fefaf482d8e67f1,
    0x3c8ec3bc41aa2008, 0x3fefba1bee615a27, 0x3c842b94c3a9eb32, 0x3fefc52b376bba97,
    0x3c8a64a931d185ee, 0x3fefd0765b6e4540, 0xbc8e37bae43be3ed, 0x3fefdbfdad9cbe14,
    0x3c77893b4d91cd9d, 0x3fefe7c1819e90d8, 0x3c5305c14160cc89, 0x3feff3c22b8f71f1,
];

const POW_OFF: u64 = 0x3fe6955500000000;
/* SIGN_BIAS = 0x800 << EXP_TABLE_BITS */
const POW_SIGN_BIAS: u32 = 0x800 << 7;
const BITS_ONE: u64 = 0x3ff0000000000000; /* 1.0 */
const BITS_INF: u64 = 0x7ff0000000000000; /* infinity */

/// Top 12 bits of a double (sign and exponent bits).
fn pow_top12(x: f64) -> u32 {
    (x.to_bits() >> 52) as u32
}

/// C `issignaling_inline`.
fn pow_issignaling(ix: u64) -> bool {
    (ix & 0x7ff8000000000000) == 0x7ff0000000000000 && (ix & 0x0007ffffffffffff) != 0
}

/// C `xflow`: force an overflow/underflow with the correct sign.
fn pow_xflow(sign: u32, y: f64) -> f64 {
    (if sign != 0 { -y } else { y }) * y
}

fn pow_math_uflow(sign: u32) -> f64 {
    pow_xflow(sign, f64::from_bits(0x1000000000000000)) /* 0x1p-767 */
}

fn pow_math_oflow(sign: u32) -> f64 {
    pow_xflow(sign, f64::from_bits(0x7000000000000000)) /* 0x1p769 */
}

/// C `log_inline`: returns `(y, tail)` with y+tail = log(x) and about 15
/// extra bits of precision in tail. `ix` is the bit representation of x,
/// normalized in the subnormal range using the sign bit for the exponent.
fn pow_log_inline(ix: u64) -> (f64, f64) {
    /* x = 2^k z; where z is in range [OFF,2*OFF) and exact.
       The range is split into N subintervals.
       The ith subinterval contains z and c is near its center. */
    let tmp = ix.wrapping_sub(POW_OFF);
    let i = ((tmp >> (52 - 7)) % 128) as usize; /* POW_LOG_TABLE_BITS = 7 */
    let k = (tmp as i64) >> 52; /* arithmetic shift */
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let z = f64::from_bits(iz);
    let kd = k as f64;

    /* log(x) = k*Ln2 + log(c) + log1p(z/c-1). */
    let invc = f64::from_bits(POW_LOG_TAB[i].0);
    let logc = f64::from_bits(POW_LOG_TAB[i].1);
    let logctail = f64::from_bits(POW_LOG_TAB[i].2);
    let ln2hi = f64::from_bits(POW_LOG_LN2HI);
    let ln2lo = f64::from_bits(POW_LOG_LN2LO);

    /* Note: 1/c is j/N or j/N/2 where j is an integer in [N,2N) and
       |z/c - 1| < 1/N, so r = z/c - 1 is exactly representible. */
    let r = z.mul_add(invc, -1.0); /* __FP_FAST_FMA */

    /* k*Ln2 + log(c) + r. */
    let t1 = kd * ln2hi + logc;
    let t2 = t1 + r;
    let lo1 = kd * ln2lo + logctail;
    let lo2 = t1 - t2 + r;

    /* Evaluation is optimized assuming superscalar pipelined execution. */
    let a0 = f64::from_bits(POW_LOG_POLY[0]); /* -0.5 */
    let a1 = f64::from_bits(POW_LOG_POLY[1]);
    let a2 = f64::from_bits(POW_LOG_POLY[2]);
    let a3 = f64::from_bits(POW_LOG_POLY[3]);
    let a4 = f64::from_bits(POW_LOG_POLY[4]);
    let a5 = f64::from_bits(POW_LOG_POLY[5]);
    let a6 = f64::from_bits(POW_LOG_POLY[6]);
    let ar = a0 * r;
    let ar2 = r * ar;
    let ar3 = r * ar2;
    /* k*Ln2 + log(c) + r + A[0]*r*r. */
    let hi = t2 + ar2;
    let lo3 = ar.mul_add(r, -ar2); /* __FP_FAST_FMA */
    let lo4 = t2 - hi + ar2;
    /* p = log1p(r) - r - A[0]*r*r. */
    let p = ar3 * (a1 + r * a2 + ar2 * (a3 + r * a4 + ar2 * (a5 + r * a6)));
    let lo = lo1 + lo2 + lo3 + lo4 + p;
    let y = hi + lo;
    let tail = hi - y + lo;
    (y, tail)
}

/// C `specialcase`: handle exp results that overflow or underflow the
/// normal scale*(1+tmp) evaluation.
fn pow_exp_specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if (ki & 0x80000000) == 0 {
        /* k > 0, the exponent of scale might have overflowed by <= 460. */
        let sbits = sbits.wrapping_sub(1009u64 << 52);
        let scale = f64::from_bits(sbits);
        /* Contracted in the reference build, exactly as in the non-special
        path below. Unfused, this branch is 1 ulp low on ~1 input in 200k. */
        return f64::from_bits(0x7f00000000000000) * scale.mul_add(tmp, scale); /* 0x1p1009 */
    }
    /* k < 0, need special care in the subnormal range. */
    let sbits = sbits.wrapping_add(1022u64 << 52);
    /* Note: sbits is signed scale. */
    let scale = f64::from_bits(sbits);
    let mut y = scale + scale * tmp;
    if y.abs() < 1.0 {
        /* Round y to the right precision before scaling it into the
           subnormal range to avoid double rounding. */
        let one: f64 = if y < 0.0 { -1.0 } else { 1.0 };
        let mut lo = scale - y + scale * tmp;
        let hi = one + y;
        lo = one - hi + y + lo;
        y = (hi + lo) - one;
        /* Fix the sign of 0. */
        if y == 0.0 {
            y = f64::from_bits(sbits & 0x8000000000000000);
        }
    }
    f64::from_bits(0x0010000000000000) * y /* 0x1p-1022 */
}

/// C `exp_inline`: computes sign*exp(x+xtail) where |xtail| < 2^-8/N and
/// |xtail| <= |x|. `sign_bias` is `POW_SIGN_BIAS` or 0.
fn pow_exp_inline(x: f64, xtail: f64, sign_bias: u32) -> f64 {
    let mut abstop = pow_top12(x) & 0x7ff;
    /* top12(0x1p-54) = 0x3c9, top12(512.0) = 0x408, top12(1024.0) = 0x409 */
    if abstop.wrapping_sub(0x3c9) >= 0x408u32.wrapping_sub(0x3c9) {
        if abstop.wrapping_sub(0x3c9) >= 0x80000000 {
            /* Avoid spurious underflow for tiny x. Note: 0 is common input. */
            let one = 1.0 + x; /* WANT_ROUNDING */
            return if sign_bias != 0 { -one } else { one };
        }
        if abstop >= 0x409 {
            /* Note: inf and nan are already handled. */
            if x.to_bits() >> 63 != 0 {
                return pow_math_uflow(sign_bias);
            } else {
                return pow_math_oflow(sign_bias);
            }
        }
        /* Large x is special cased below. */
        abstop = 0;
    }

    /* exp(x) = 2^(k/N) * exp(r), with exp(r) in [2^(-1/2N),2^(1/2N)]. */
    /* x = ln2/N*k + r, with int k and r in [-ln2/2N, ln2/2N]. */
    let z = f64::from_bits(EXP_INVLN2N) * x;
    /* z - kd is in [-1, 1] in non-nearest rounding modes
       (TOINT_INTRINSICS=0, EXP_USE_TOINT_NARROW=0 path). */
    let shift = f64::from_bits(EXP_SHIFT);
    let mut kd = z + shift;
    let ki = kd.to_bits();
    kd -= shift;
    let mut r = x + kd * f64::from_bits(EXP_NEGLN2HIN) + kd * f64::from_bits(EXP_NEGLN2LON);
    /* The code assumes 2^-200 < |xtail| < 2^-8/N. */
    r += xtail;
    /* 2^(k/N) ~= scale * (1 + tail). */
    let idx = (2 * (ki % 128)) as usize; /* EXP_TABLE_BITS = 7 */
    let top = ki.wrapping_add(sign_bias as u64) << (52 - 7);
    let tail = f64::from_bits(EXP_TAB[idx]);
    /* This is only a valid scale when -1023*N < k < 1024*N. */
    let sbits = EXP_TAB[idx + 1].wrapping_add(top);
    /* exp(x) = 2^(k/N) * exp(r) ~= scale + scale * (tail + exp(r) - 1). */
    let r2 = r * r;
    let c2 = f64::from_bits(EXP_POLY[0]);
    let c3 = f64::from_bits(EXP_POLY[1]);
    let c4 = f64::from_bits(EXP_POLY[2]);
    let c5 = f64::from_bits(EXP_POLY[3]);
    /* glibc builds this translation unit with -ffp-contract=fast, so the
    compiler contracts this polynomial into fused multiply-adds. Measured
    against a gcc -ffp-contract=fast build of the same C: this association
    is the one the compiler emits (see POW_FMA_EXACTNESS.md). */
    let tmp = (r2 * r2).mul_add(r.mul_add(c5, c4), r2.mul_add(r.mul_add(c3, c2), tail + r));
    if abstop == 0 {
        return pow_exp_specialcase(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    /* Note: tmp == 0 or |tmp| > 2^-200 and scale > 2^-739, so there
       is no spurious underflow here even without fma. */
    /* glibc's x86-64 `e_pow-fma.c` is built with `-mfma -mavx2
       -ffp-contract=fast`, so this final `scale + scale * tmp` — the last
       operation before the result is rounded — is emitted as a single fused
       multiply-add. It must be fused here too or the returned double lands
       on the wrong side of a rounding boundary whenever the exact result is
       within a fraction of an ulp of the midpoint. */
    scale.mul_add(tmp, scale)
}

/// C `checkint`: 0 if not an integer, 1 if odd, 2 if even; `iy` is the
/// bit representation of a non-zero finite value.
fn pow_checkint(iy: u64) -> i32 {
    let e = ((iy >> 52) & 0x7ff) as i32;
    if e < 0x3ff {
        return 0;
    }
    if e > 0x3ff + 52 {
        return 2;
    }
    if iy & ((1u64 << (0x3ff + 52 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1u64 << (0x3ff + 52 - e)) != 0 {
        return 1;
    }
    2
}

/// C `zeroinfnan`: true if the bit pattern is 0, infinity or nan.
fn pow_zeroinfnan(i: u64) -> bool {
    i.wrapping_mul(2).wrapping_sub(1) >= BITS_INF.wrapping_mul(2).wrapping_sub(1)
}

/// glibc-compatible `pow(x, y)` (see block comment above).
pub(crate) fn pow_glibc(x: f64, y: f64) -> f64 {
    let mut sign_bias: u32 = 0;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = pow_top12(x);
    let topy = pow_top12(y);
    if topx.wrapping_sub(0x001) >= 0x7ff - 0x001
        || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be
    {
        /* Note: if |y| > 1075 * ln2 * 2^53 ~= 0x1.749p62 then pow(x,y) = inf/0
           and if |y| < 2^-54 / 1075 ~= 0x1.e7b6p-65 then pow(x,y) = +-1. */
        /* Special cases: (x < 0x1p-126 or inf or nan) or
           (|y| < 0x1p-65 or |y| >= 0x1p63 or nan). */
        if pow_zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return if pow_issignaling(ix) { x + y } else { 1.0 };
            }
            if ix == BITS_ONE {
                return if pow_issignaling(iy) { x + y } else { 1.0 };
            }
            if ix.wrapping_mul(2) > BITS_INF.wrapping_mul(2)
                || iy.wrapping_mul(2) > BITS_INF.wrapping_mul(2)
            {
                return x + y;
            }
            if ix.wrapping_mul(2) == BITS_ONE.wrapping_mul(2) {
                return 1.0;
            }
            if (ix.wrapping_mul(2) < BITS_ONE.wrapping_mul(2)) == (iy >> 63 == 0) {
                return 0.0; /* |x|<1 && y==inf or |x|>1 && y==-inf. */
            }
            return y * y;
        }
        if pow_zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix >> 63 != 0 && pow_checkint(iy) == 1 {
                x2 = -x2;
            }
            return if iy >> 63 != 0 { 1.0 / x2 } else { x2 };
        }
        /* Here x and y are non-zero finite. */
        if ix >> 63 != 0 {
            /* Finite x < 0. */
            let yint = pow_checkint(iy);
            if yint == 0 {
                return f64::NAN; /* __math_invalid */
            }
            if yint == 1 {
                sign_bias = POW_SIGN_BIAS;
            }
            ix &= 0x7fffffffffffffff;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
            /* Note: sign_bias == 0 here because y is not odd. */
            if ix == BITS_ONE {
                return 1.0;
            }
            if (topy & 0x7ff) < 0x3be {
                /* |y| < 2^-65, x^y ~= 1 + y*log(x). */
                return if ix > BITS_ONE { 1.0 + y } else { 1.0 - y }; /* WANT_ROUNDING */
            }
            return if (ix > BITS_ONE) == (topy < 0x800) {
                pow_math_oflow(0)
            } else {
                pow_math_uflow(0)
            };
        }
        if topx == 0 {
            /* Normalize subnormal x so exponent becomes negative. */
            ix = (x * f64::from_bits(0x4330000000000000)).to_bits(); /* 0x1p52 */
            ix &= 0x7fffffffffffffff;
            ix = ix.wrapping_sub(52u64 << 52);
        }
    }

    let (hi, lo) = pow_log_inline(ix);
    /* __FP_FAST_FMA */
    let ehi = y * hi;
    let elo = y * lo + y.mul_add(hi, -ehi);
    pow_exp_inline(ehi, elo, sign_bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers() {
        assert_eq!(SUNIpowerI(2, 10), 1024);
        assert_eq!(SUNIpowerI(3, 0), 1);
        assert_eq!(SUNRpowerI(2.0, -2), 0.25);
        assert_eq!(SUNRpowerI(10.0, 3), 1000.0);
    }

    #[test]
    fn pow_exact() {
        assert_eq!(SUNRpowerR(2.0, 10.0), 1024.0);
        assert_eq!(SUNRpowerR(4.0, 0.5), 2.0);
        assert_eq!(SUNRpowerR(1.0, 123.456), 1.0);
        assert_eq!(SUNRpowerR(7.25, 0.0), 1.0);
        assert_eq!(SUNRpowerR(0.0, 2.0), 0.0);
        assert_eq!(SUNRpowerR(-2.0, 3.0), -8.0);
        assert_eq!(SUNRpowerR(-2.0, 2.0), 4.0);
        assert!(SUNRpowerR(-2.0, 0.5).is_nan());
    }

    #[test]
    fn pow_glibc_bits() {
        /* (x, y, result) bit-pattern triples recorded from the C build of
        the same algorithm (musl pow interposed into cvVdp_auto_nls, whose
        run reproduces the upstream reference byte-for-byte). */
        const TRIPLES: [(u64, u64, u64); 5] = [
            (0x3ea1c9676388326e, 0x3fe0000000000000, 0x3f47db7e7ebb7d59),
            (0x3fed70a44b135d5c, 0x3fe0000000000000, 0x3feeb17dc53b4a93),
            (0x3fa5fb20341c94e9, 0x3fd5555555555555, 0x3fd668ebabd8e3dd),
            (0x3fcca8448d464163, 0x3fe0000000000000, 0x3fde4855e322ff6f),
            (0x40008a8214b0d2cf, 0x3fd5555555555555, 0x3ff46229aa6e2e7a),
        ];
        for (bx, by, br) in TRIPLES {
            let r = SUNRpowerR(f64::from_bits(bx), f64::from_bits(by));
            assert_eq!(r.to_bits(), br, "pow({bx:016x}, {by:016x})");
        }
    }

    /* ---------------------------------------------------------------
     * Differential test against the *native* host `pow`.
     *
     * The oracle is `tools/pow_oracle.c`, built and run on the target
     * host (Linux / glibc / x86-64) by `tools/pow_differential.sh`; see
     * POW_FMA_EXACTNESS.md.  The corpora are regenerated here from the
     * same splitmix64 recurrence rather than transmitted, so the two
     * programs cannot disagree about which inputs they evaluated —
     * keep `pow_corpus` below byte-for-byte in step with the C twin.
     *
     * Without the oracle files the test reports "not run" and passes:
     * `cargo test` must stay green on hosts where no oracle was built
     * (and on non-glibc hosts, where the oracle would be meaningless).
     * --------------------------------------------------------------- */

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /* [0,1) from the top 53 bits — the C twin's `unit()`.
        0x1p-53 is written as a bit pattern so no libm call appears. */
        fn unit(&mut self) -> f64 {
            ((self.next() >> 11) as f64) * f64::from_bits(0x3ca0000000000000)
        }
    }

    fn pow_corpus(domain: bool) -> impl Iterator<Item = (f64, f64)> {
        let mut rng = SplitMix64(if domain { 1 } else { 2 });
        let n = if domain { 5_900_000usize } else { 20_000_000usize };
        (0..n).map(move |_| {
            if domain {
                /* the domain SUNRpowerR is actually evaluated on:
                pow(bias*dsm, +-1/order) => x in (0, ~100], |y| <= 1 */
                let mut x = rng.unit() * 100.0;
                if x == 0.0 {
                    x = 100.0;
                }
                let s = rng.next();
                let y = if s % 14 == 0 {
                    rng.unit() * 2.0 - 1.0
                } else {
                    let v = 1.0 / ((s % 13) + 1) as f64;
                    if s & 0x100 != 0 { -v } else { v }
                };
                (x, y)
            } else {
                loop {
                    let x = f64::from_bits(rng.next());
                    let y = f64::from_bits(rng.next());
                    if x.is_finite() && y.is_finite() {
                        break (x, y);
                    }
                }
            }
        })
    }

    fn run_pow_differential(var: &str, domain: bool) -> Option<(usize, usize)> {
        let path = std::env::var(var).ok()?;
        let blob = std::fs::read(&path).unwrap_or_else(|e| panic!("{var}={path}: {e}"));
        let mut mismatches = 0usize;
        let mut total = 0usize;
        for (i, (x, y)) in pow_corpus(domain).enumerate() {
            let off = i * 8;
            if off + 8 > blob.len() {
                break;
            }
            let mut w = [0u8; 8];
            w.copy_from_slice(&blob[off..off + 8]);
            let want = u64::from_le_bytes(w);
            let got = pow_glibc(x, y).to_bits();
            total += 1;
            /* NaN payloads are not architecturally specified; both-NaN
            counts as agreement (documented in POW_FMA_EXACTNESS.md). */
            if got != want && !(f64::from_bits(want).is_nan() && f64::from_bits(got).is_nan()) {
                if mismatches < 10 {
                    eprintln!(
                        "pow mismatch: x={:016x} y={:016x} oracle={want:016x} port={got:016x}",
                        x.to_bits(),
                        y.to_bits()
                    );
                }
                mismatches += 1;
            }
        }
        Some((total, mismatches))
    }

    #[test]
    fn pow_glibc_vs_native_oracle_domain() {
        match run_pow_differential("SUNDIALS_POW_ORACLE_DOMAIN", true) {
            None => eprintln!("pow domain differential: not run (no oracle)"),
            Some((total, bad)) => {
                eprintln!("pow domain differential: {total} inputs, {bad} mismatches");
                assert_eq!(bad, 0, "deterministic pow disagrees with the host libm");
            }
        }
    }

    #[test]
    fn pow_glibc_vs_native_oracle_random() {
        match run_pow_differential("SUNDIALS_POW_ORACLE_RANDOM", false) {
            None => eprintln!("pow random differential: not run (no oracle)"),
            Some((total, bad)) => {
                /* Out-of-domain corpus: reported, and bounded, but not a
                gate — see POW_FMA_EXACTNESS.md §5. */
                eprintln!("pow random differential: {total} inputs, {bad} mismatches");
                assert!(bad * 1_000_000 <= total, "residual disagreement above 1e-6");
            }
        }
    }

    #[test]
    fn compare() {
        assert!(!SUNRCompare(1.0, 1.0));
        assert!(SUNRCompare(1.0, 1.001));
        assert!(SUNRCompare(f64::NAN, 1.0));
        assert!(!SUNRCompare(f64::INFINITY, f64::INFINITY));
    }

    #[test]
    fn str_to_real() {
        assert_eq!(SUNStrToReal("1e-3"), 1e-3);
        assert_eq!(SUNStrToReal("  -2.5rest"), -2.5);
        assert_eq!(SUNStrToReal("junk"), 0.0);
        assert_eq!(SUNStrToReal(".5"), 0.5);
        assert_eq!(SUNStrToReal("3."), 3.0);
        assert_eq!(SUNStrToReal("1e"), 1.0);
        assert_eq!(SUNStrToReal("inf"), f64::INFINITY);
        assert_eq!(SUNStrToReal("-Infinity"), f64::NEG_INFINITY);
    }

    #[test]
    fn sqrt_guard() {
        assert_eq!(SUNRsqrt(-4.0), 0.0);
        assert_eq!(SUNRsqrt(4.0), 2.0);
    }
}
