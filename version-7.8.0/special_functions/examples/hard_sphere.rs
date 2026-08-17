//! Doc-example check: hard-sphere partial-wave cross-section -> 4 pi a^2.
use special_functions::sph_bessel::{sph_j, sph_y};

fn hard_sphere_cross_section(k: f64, a: f64, l_max: i32) -> Result<f64, String> {
    let ka = k * a;
    let mut sigma = 0.0;
    for l in 0..=l_max {
        let delta = (sph_j(l, ka)? / sph_y(l, ka)?).atan();
        sigma += (2 * l + 1) as f64 * delta.sin().powi(2);
    }
    Ok(4.0 * std::f64::consts::PI / (k * k) * sigma)
}

fn main() {
    let a = 1.0;
    let geometric = std::f64::consts::PI * a * a;
    for &k in &[1e-3, 1e-2, 1e-1] {
        let s = hard_sphere_cross_section(k, a, 6).unwrap();
        println!("  ka={:<6} sigma/(pi a^2) = {:.6}", k * a, s / geometric);
    }
    let s = hard_sphere_cross_section(1e-3, a, 6).unwrap();
    assert!((s / geometric - 4.0).abs() < 1e-4, "low-energy limit is 4 pi a^2");
    println!("SUCCESS: low-energy hard-sphere cross-section -> 4 pi a^2");
}
