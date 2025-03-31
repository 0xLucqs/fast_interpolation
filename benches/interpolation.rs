use ark_ff::One;
use ark_ff_optimized::fp31::Fp;
use ark_poly::{DenseUVPolynomial, Polynomial, polynomial::univariate::DensePolynomial};
use bitvec::{field::BitField, order::Lsb0, vec::BitVec};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fast_interpolation::fast_interpolation;

fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");
    group.significance_level(0.1).sample_size(10);
    let bytes = include_bytes!("../blob");
    let bitvec = BitVec::<u8, Lsb0>::from_slice(bytes);
    let mut coeffs = bitvec
        .chunks(30)
        .map(|chunk| {
            let value = chunk.load::<u32>();
            Fp(value)
        })
        .collect::<Vec<_>>();
    let length = coeffs.len().next_power_of_two();

    coeffs.resize(length, Fp::one());

    let poly = DensePolynomial::from_coefficients_vec(coeffs);
    let mut xs = Vec::with_capacity(poly.coeffs.len());
    let mut ys = Vec::with_capacity(poly.coeffs.len());
    println!("computing xs and ys");
    for i in 0..poly.degree() + 1 {
        xs.push(Fp::from(i as u32));
        let y = poly.evaluate(&xs[i]);
        ys.push(y);
    }
    println!("Done");
    group.bench_function("full blob".to_string(), |b| {
        b.iter(|| {
            let f = fast_interpolation(black_box(&xs), black_box(&ys));
            assert_eq!(f.degree(), poly.degree());
            assert_eq!(f, poly);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_interpolation);
criterion_main!(benches);
