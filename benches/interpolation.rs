use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fast_interpolation::complex_interpolation::interpolate_qm31;
use frieda::utils::polynomial_from_bytes;
use stwo_prover::core::poly::circle::CanonicCoset;

fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");
    group.significance_level(0.1).sample_size(10);
    let poly = polynomial_from_bytes(include_bytes!("../blob"));
    let bigger_domain = CanonicCoset::new(poly.log_size() + 10).circle_domain();
    let point_evals = (0..=(1 << poly.log_size()))
        .map(|i| {
            let point = bigger_domain.at(i);
            let eval = poly.eval_at_point(point.into_ef());
            (point, eval)
        })
        .collect::<Vec<_>>();
    let xs = point_evals
        .clone()
        .into_iter()
        .map(|(x, _)| x)
        .collect::<Vec<_>>();
    let ys = point_evals.into_iter().map(|(_, y)| y).collect::<Vec<_>>();

    group.bench_function("full blob".to_string(), |b| {
        b.iter(|| {
            let f = interpolate_qm31(black_box(&xs), black_box(&ys));

            assert_eq!(f.0[0].coeffs, poly.0[0].coeffs);
            assert_eq!(f.0[1].coeffs, poly.0[1].coeffs);
            assert_eq!(f.0[2].coeffs, poly.0[2].coeffs);
            assert_eq!(f.0[3].coeffs, poly.0[3].coeffs);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_interpolation);
criterion_main!(benches);
