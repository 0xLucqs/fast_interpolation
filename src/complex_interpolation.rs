use num::{Complex, Zero};
use stwo_prover::core::fields::m31::M31;

fn complex_interpolation(xs: &[Complex<f64>], ys: &[Complex<f64>]) -> Complex<M31> {
    let n = xs.len();
    Complex::new(M31::zero(), M31::zero())
}
