use lambdaworks_math::circle::cosets::Coset;

use lambdaworks_math::circle::point::CirclePoint;
use lambdaworks_math::circle::polynomial::{evaluate_cfft, interpolate_cfft};
use stwo_prover::core::circle::CirclePoint as SCirclePoint;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::extensions::Degree2ExtensionField;

use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
use lambdaworks_math::polynomial::Polynomial;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use stwo_prover::core::fields::m31::M31;

use stwo_prover::core::fields::qm31::QM31;
use stwo_prover::core::poly::circle::CanonicCoset;

use crate::{build_subproduct_tree, eval_tree, linear_combination};

pub fn interpolate_qm31(pks: &[SCirclePoint<M31>], extvks: &[QM31]) -> Vec<M31> {
    println!("pks: {:?}", pks.len());
    println!("extvks: {:?}", extvks.len());
    let zks = &pks
        .iter()
        .map(|zk| {
            FieldElement::<Degree2ExtensionField>::new([
                FieldElement::<Mersenne31Field>::new(zk.x.0),
                FieldElement::<Mersenne31Field>::new(zk.y.0),
            ])
        })
        .collect::<Vec<_>>();
    let ((first_col, second_col), (third_col, fourth_col)) = rayon::join(
        || {
            interpolate_cm31(
                zks,
                &extvks
                    .iter()
                    .map(|vk| {
                        FieldElement::<Degree2ExtensionField>::new([
                            FieldElement::<Mersenne31Field>::new(vk.0.0.0),
                            FieldElement::<Mersenne31Field>::new(vk.0.1.0),
                        ])
                    })
                    .collect::<Vec<_>>(),
            )
        },
        || {
            interpolate_cm31(
                zks,
                &extvks
                    .iter()
                    .map(|vk| {
                        FieldElement::<Degree2ExtensionField>::new([
                            FieldElement::<Mersenne31Field>::new(vk.1.0.0),
                            FieldElement::<Mersenne31Field>::new(vk.1.1.0),
                        ])
                    })
                    .collect::<Vec<_>>(),
            )
        },
    );

    // concatenate the 4 columns
    first_col
        .into_iter()
        .zip(second_col)
        .zip(third_col)
        .zip(fourth_col)
        .flat_map(|(((a, b), c), d)| [a, b, c, d])
        .collect::<Vec<_>>()
}
pub fn interpolate_cm31(
    zks: &[FieldElement<Degree2ExtensionField>],
    vks: &[FieldElement<Degree2ExtensionField>],
) -> (Vec<M31>, Vec<M31>) {
    println!("zks len = {}", zks.len());
    assert_eq!(zks.len(), vks.len());
    assert!((zks.len() - 1).is_power_of_two());

    let n = zks.len() - 1;
    println!("computing vprimes");
    let vprimes = vks
        .into_par_iter()
        .zip_eq(zks)
        .map(|(vk, zk)| vk.clone() * zk.pow(n as u128 / 2))
        .collect::<Vec<_>>();

    println!("computing zks");
    let mut pol = fast_interpolation(zks, &vprimes);
    // assert_eq!(
    //     CM31::from_u32_unchecked(
    //         pol.coefficients[0].value()[0].representative(),
    //         pol.coefficients[0].value()[1].representative()
    //     ),
    //     CM31::from_u32_unchecked(
    //         pol.coefficients[pol.coefficients.len() - 1].value()[0].representative(),
    //         pol.coefficients[pol.coefficients.len() - 1].value()[1].representative()
    //     )
    //     .complex_conjugate()
    // );
    // assert_eq!(
    //     CM31::from_u32_unchecked(
    //         pol.coefficients[1].value()[0].representative(),
    //         pol.coefficients[1].value()[1].representative()
    //     ),
    //     CM31::from_u32_unchecked(
    //         pol.coefficients[pol.coefficients.len() - 2].value()[0].representative(),
    //         pol.coefficients[pol.coefficients.len() - 2].value()[1].representative()
    //     )
    //     .complex_conjugate()
    // );
    assert_eq!(
        zks.iter().map(|z| pol.evaluate(z)).collect::<Vec<_>>(),
        vprimes
    );

    let last_coeff = pol.coefficients.pop().unwrap();

    // let last_term = pol.coefficients.pop().unwrap();
    // println!("last term = {:?}", last_term);
    // println!("computing domain");
    println!("poly len = {}", pol.coefficients.len());
    let coset = Coset::new_standard(n.ilog2());
    let stwo_coset = CanonicCoset::new(n.ilog2()).coset();
    let points = Coset::get_coset_points(&coset);
    assert_eq!(
        stwo_coset
            .iter()
            .map(|p| CirclePoint::new(
                FieldElement::<Mersenne31Field>::new(p.x.0),
                FieldElement::<Mersenne31Field>::new(p.y.0)
            )
            .unwrap())
            .collect::<Vec<_>>(),
        points
    );
    assert_eq!(points.len(), n);

    println!("evaluating");

    // assert_eq!(pol.coefficients.len(), n);
    // Evaluate p(z) (or p(z) - c_N * z^N, with c_N added back as last_term) on the domain.
    // let evals: Vec<ComplexCircleFp> = domain.fft(&pol.coeffs);
    let f_values = points
        .iter()
        .map(|z| {
            let z = FieldElement::<Degree2ExtensionField>::new([z.x, z.y]);
            ((pol.evaluate(&z) + &last_coeff * z.pow(n)) / z.pow(n as u128 / 2)).unwrap()
        })
        .collect::<Vec<_>>();

    let re_col = f_values.iter().map(|x| x.value()[0]).collect::<Vec<_>>();

    let im_col = f_values.iter().map(|x| x.value()[1]).collect::<Vec<_>>();

    println!("interpolating");
    let re_fft = interpolate_cfft(re_col.clone());
    let re_evaluations = evaluate_cfft(re_fft.clone());
    assert_eq!(re_evaluations, re_col);
    let im_fft = interpolate_cfft(im_col);

    (
        re_fft
            .into_iter()
            .map(|x| M31::from_u32_unchecked(*x.value()))
            .collect(),
        im_fft
            .into_iter()
            .map(|x| M31::from_u32_unchecked(*x.value()))
            .collect(),
    )
}

pub fn fast_interpolation(
    u: &[FieldElement<Degree2ExtensionField>],
    v: &[FieldElement<Degree2ExtensionField>],
) -> Polynomial<FieldElement<Degree2ExtensionField>> {
    let z0 = &u[0];
    let v0 = &v[0];
    let u = &u[1..];
    let v = &v[1..];
    assert_eq!(u.len(), v.len());
    assert!(u.len().is_power_of_two());

    let tree = build_subproduct_tree(u);

    let k = u.len().ilog2() as usize;
    let m = &tree[k][0];

    let m_deriv = m.differentiate();
    let m_deriv_vals = eval_tree(&m_deriv, &tree, u, 0);

    let c: Vec<FieldElement<Degree2ExtensionField>> = v
        .par_iter()
        .zip(m_deriv_vals.par_iter())
        .map(|(vi, m_prime_ui)| vi * &m_prime_ui.inv().unwrap())
        .collect();

    let poly = linear_combination(u, &c, &tree, k, 0);

    let lambda = (v0 - poly.evaluate(z0)) * m.evaluate(z0).inv().unwrap();

    assert_eq!(&lambda * m.evaluate(z0) + poly.evaluate(z0), *v0);

    poly + m * lambda
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use frieda::{
        proof::{generate_proof, get_queries_from_proof},
        utils::{self, felts_to_bytes_le},
    };
    use lambdaworks_math::field::fields::mersenne31::{
        extensions::Degree2ExtensionField, field::Mersenne31Field,
    };
    use num_traits::Pow;
    use rand::Rng;
    use stwo_prover::core::{
        backend::CpuBackend,
        circle::Coset as SCoset,
        fields::m31::BaseField,
        fri::FriConfig,
        pcs::PcsConfig,
        poly::circle::{CircleDomain, CirclePoly, SecureCirclePoly},
        utils::bit_reverse_index,
    };
    const QUERY_PER_SAMPLE: usize = 20;
    const PCS_CONFIG: PcsConfig = PcsConfig {
        pow_bits: 1,
        fri_config: FriConfig {
            log_blowup_factor: 4,
            log_last_layer_degree_bound: 1,
            n_queries: QUERY_PER_SAMPLE,
        },
    };

    use super::*;

    #[test]
    fn test_simple_interpolation() {
        // Create a simple polynomial with known coefficients
        // Each coefficient is a pair of field elements (real, imaginary)
        const POLY_SIZE: usize = 16;

        let one = BaseField::from(1);
        let coeffs = (0..POLY_SIZE).map(|_| (one, one)).collect::<Vec<_>>();

        println!("coeffs.len(): {:?}", coeffs.len());

        let circle_poly = CirclePoly::<CpuBackend>::new(vec![one; POLY_SIZE]);

        let mut rng = rand::rng();

        let random_points = (0..=coeffs.len())
            .map(|_| {
                let index = rng.random_range(0..(2.pow(coeffs.len().ilog2() + 10) as usize));
                CanonicCoset::new(coeffs.len().ilog2() + 10)
                    .circle_domain()
                    .at(index)
            })
            .collect::<Vec<_>>();

        let random_evals = random_points
            .par_iter()
            .map(|point| {
                let eval = circle_poly.eval_at_point(point.into_ef());
                println!("eval = {:#?}", eval);
                FieldElement::<Degree2ExtensionField>::new([
                    FieldElement::<Mersenne31Field>::new(eval.0.0.0),
                    FieldElement::<Mersenne31Field>::new(eval.0.1.0),
                ])
            })
            .collect::<Vec<_>>();

        let random_points = random_points
            .iter()
            .map(|point| {
                FieldElement::<Degree2ExtensionField>::new([
                    FieldElement::<Mersenne31Field>::new(point.x.0),
                    FieldElement::<Mersenne31Field>::new(point.y.0),
                ])
            })
            .collect::<Vec<_>>();
        // Interpolate back
        let (re_col, im_col) = interpolate_cm31(&random_points, &random_evals);
        let interpolated = re_col.into_iter().zip(im_col);

        // Compare results
        println!("interpolated.len(): {:?}", interpolated.len());
        println!("coeffs.len(): {:?}", coeffs.len());
        println!("coeffs: {:?}", &coeffs);
        println!();
        println!("interp: {:?}", &interpolated);

        for (i, ((bre, bim), (ire, iim))) in coeffs.iter().zip(interpolated).enumerate() {
            assert_eq!(bre, &ire, "failed real part at {}", i);
            assert_eq!(bim, &iim, "failed imaginary part at {}", i);
        }
    }

    #[test]
    fn test_interpolate_cm31() {
        let data = include_bytes!("../blob")[..(2048. * 3.75) as usize].to_vec();
        let mut coeffs = frieda::utils::bytes_to_felt_le(&data);
        let next_power_of_2 = 1 << ((coeffs.len() as f64).log2().ceil() as u32).max(2);

        coeffs.resize(next_power_of_2, BaseField::from(1));
        let real_circle_poly =
            CirclePoly::<CpuBackend>::new(coeffs.iter().step_by(2).copied().collect());
        let imaginary_circle_poly =
            CirclePoly::<CpuBackend>::new(coeffs.iter().skip(1).step_by(2).copied().collect());
        let empty_circle_poly =
            CirclePoly::<CpuBackend>::new(vec![BaseField::from(0); coeffs.len()]);
        let circle_poly = SecureCirclePoly::<CpuBackend>([
            real_circle_poly,
            imaginary_circle_poly,
            empty_circle_poly.clone(),
            empty_circle_poly,
        ]);
        // evaluate the polynomial at the even coefficients

        let mut rng = rand::rng();
        let domain_size = 2_usize.pow(coeffs.len().ilog2() + 10);

        let mut used_indices = std::collections::HashSet::new();
        let mut random_points = Vec::with_capacity(coeffs.len() / 2 + 1);
        let canonic_coset = CanonicCoset::new(coeffs.len().ilog2() + 10);

        while random_points.len() <= coeffs.len() / 2 {
            let index = rng.random_range((coeffs.len() + 1)..domain_size);
            if used_indices.insert(index) {
                random_points.push(canonic_coset.circle_domain().at(index));
            }
        }
        random_points.sort();

        let random_evals = random_points
            .par_iter()
            .map(|point| {
                let eval = circle_poly.eval_at_point(point.into_ef());
                FieldElement::<Degree2ExtensionField>::new([
                    FieldElement::<Mersenne31Field>::new(eval.0.0.0),
                    FieldElement::<Mersenne31Field>::new(eval.0.1.0),
                ])
            })
            .collect::<Vec<_>>();

        let random_points = random_points
            .iter()
            .map(|point| {
                FieldElement::<Degree2ExtensionField>::new([
                    FieldElement::<Mersenne31Field>::new(point.x.0),
                    FieldElement::<Mersenne31Field>::new(point.y.0),
                ])
            })
            .collect::<Vec<_>>();

        let (re_col, im_col) = interpolate_cm31(&random_points, &random_evals);
        let interpolated = re_col
            .into_iter()
            .zip(im_col)
            .flat_map(|(a, b)| [a, b])
            .collect::<Vec<_>>();
        println!("interpolated.len(): {:?}", interpolated.len());
        println!("coeffs.len(): {:?}", coeffs.len() / 2);
        println!("coeffs[..10]: {:?}", &coeffs[..10]);
        println!("interp[..10]: {:?}", &interpolated[..10]);
        assert_eq!(interpolated, coeffs);
    }

    #[test]
    fn test_something() {
        let data = include_bytes!("../blob").to_vec();
        let poly = utils::polynomial_from_bytes(&data);

        // we should have polys.len().next_power_of_two() samples
        let samples_nb = (1 << (poly.log_size() + 1)) / QUERY_PER_SAMPLE;
        println!("samples_nb: {:?}", samples_nb);
        let proofs_pos = (0..=samples_nb)
            .into_par_iter()
            .map(|i| {
                let proof = generate_proof(&data, Some(i as u64), PCS_CONFIG);
                let queries = get_queries_from_proof(proof.clone(), Some(i as u64));
                (proof, queries)
            })
            .collect::<Vec<_>>();

        let domain = CircleDomain::new(SCoset::half_odds(proofs_pos[0].0.coset_log_size));
        let mut pos_set = HashSet::new();
        let mut pos_vec = Vec::new();
        let mut evals_vec = Vec::new();
        for (proof, (_log_size, pos)) in proofs_pos {
            for (i, p) in pos.iter().enumerate() {
                let point = domain.at(bit_reverse_index(*p, domain.log_size()));
                if pos_set.insert(point) {
                    pos_vec.push(point);
                    evals_vec.push(proof.evaluations[i]);
                }
            }
        }

        let pos = pos_vec;
        let evals = evals_vec;
        let evals_nb = (1 << poly.log_size()) + 1;
        println!("evals_nb: {:?}", evals_nb);

        println!("evals.len(): {:?}", evals.len());

        let interpolated = interpolate_qm31(&pos[..evals_nb], &evals[..evals_nb]);
        let interpolated_bytes = felts_to_bytes_le(&interpolated);
        interpolated_bytes
            .into_iter()
            .zip(data)
            .enumerate()
            .for_each(|(i, (a, b))| {
                if a != b {
                    println!("failed at {}", i);
                }
            });
        // assert_eq!(data, interpolated_bytes[..data.len()]);
        // assert_eq!(interpolated.0[0].coeffs[..500], poly.0[0].coeffs[..500]);
        // assert_eq!(interpolated.0[1].coeffs[..500], poly.0[1].coeffs[..500]);
        // assert_eq!(interpolated.0[2].coeffs[..500], poly.0[2].coeffs[..500]);
        // assert_eq!(interpolated.0[3].coeffs[..500], poly.0[3].coeffs[..500]);
    }
}
