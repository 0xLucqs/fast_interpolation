use lambdaworks_math::circle::cosets::Coset;

use lambdaworks_math::circle::point::CirclePoint;
use lambdaworks_math::circle::polynomial::{evaluate_cfft, interpolate_cfft};

use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::extensions::Degree2ExtensionField;

use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
use lambdaworks_math::polynomial::Polynomial;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use stwo_prover::core::backend::CpuBackend;
use stwo_prover::core::backend::cpu::bit_reverse;

use stwo_prover::core::fields::cm31::CM31;
use stwo_prover::core::fields::m31::M31;

use stwo_prover::core::fields::ComplexConjugate;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};

use crate::{build_subproduct_tree, eval_tree, linear_combination};

// pub fn interpolate_qm31(pks: &[CirclePoint<M31>], extvks: &[QM31]) -> Vec<M31> {
//     println!("pks: {:?}", pks.len());
//     println!("extvks: {:?}", extvks.len());
//     let _zks = &pks
//         .iter()
//         .map(|zk| CM31::from_m31(zk.x, zk.y))
//         .collect::<Vec<_>>();
//     // let first_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.0).collect::<Vec<_>>());
//     // let second_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.1).collect::<Vec<_>>());

//     // first_poly
//     //     .into_iter()
//     //     .zip(second_poly)
//     //     .map(|(a, b)| a + b)
//     //     .collect::<Vec<_>>()
//     vec![]
// }
pub fn interpolate_cm31(
    zks: &[FieldElement<Degree2ExtensionField>],
    vks: &[FieldElement<Degree2ExtensionField>],
) -> Vec<M31> {
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
    assert_eq!(
        CM31::from_u32_unchecked(
            pol.coefficients[0].value()[0].representative(),
            pol.coefficients[0].value()[1].representative()
        ),
        CM31::from_u32_unchecked(
            pol.coefficients[pol.coefficients.len() - 1].value()[0].representative(),
            pol.coefficients[pol.coefficients.len() - 1].value()[1].representative()
        )
        .complex_conjugate()
    );
    assert_eq!(
        CM31::from_u32_unchecked(
            pol.coefficients[1].value()[0].representative(),
            pol.coefficients[1].value()[1].representative()
        ),
        CM31::from_u32_unchecked(
            pol.coefficients[pol.coefficients.len() - 2].value()[0].representative(),
            pol.coefficients[pol.coefficients.len() - 2].value()[1].representative()
        )
        .complex_conjugate()
    );
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
    println!("--------------------------------");
    println!("f_values = {:#?}", f_values);
    println!("--------------------------------");
    println!("adjusting");

    println!("N = {}", n);
    println!("{} evals", f_values.len());
    println!("packing");

    let re_col = f_values.iter().map(|x| x.value()[0]).collect::<Vec<_>>();

    let im_col = f_values.iter().map(|x| x.value()[1]).collect::<Vec<_>>();
    println!("fft");
    println!("re_col len = {}", re_col.len());
    println!("re_col log len = {}", re_col.len().ilog2());

    println!("im_col len = {}", im_col.len());
    println!("im_col log len = {}", im_col.len().ilog2());

    let circle_evaluations = CircleEvaluation::<CpuBackend, M31, _>::new_canonical_ordered(
        CanonicCoset::new(n.ilog2()),
        re_col
            .iter()
            .map(|eval| M31::from_u32_unchecked(*eval.value()))
            .collect(),
    );
    let some_coeffs = circle_evaluations
        .interpolate()
        .coeffs
        .iter()
        .map(|x| FieldElement::new(x.0))
        .collect::<Vec<_>>();

    println!("interpolating");
    let re_fft = interpolate_cfft(re_col.clone());
    println!("--------------------------------");
    println!("--------------------------------");
    println!("--------------------------------");
    println!("re_fft = {:#?}", re_fft);
    println!("--------------------------------");
    println!("--------------------------------");
    println!("--------------------------------");
    let re_evaluations = evaluate_cfft(re_fft.clone());
    assert_eq!(re_evaluations, re_col);
    let im_fft = interpolate_cfft(im_col);
    assert_eq!(some_coeffs, re_fft);
    println!("re_fft len = {}", re_fft.len());
    println!("im_fft len = {}", im_fft.len());

    re_fft
        .into_iter()
        .zip(im_fft)
        .flat_map(|(re, im)| {
            [
                M31::from_u32_unchecked(re.representative()),
                M31::from_u32_unchecked(im.representative()),
            ]
        })
        .collect::<Vec<_>>()
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

    use lambdaworks_math::field::fields::mersenne31::{
        extensions::Degree2ExtensionField, field::Mersenne31Field,
    };
    use num_traits::Pow;
    use rand::Rng;
    use stwo_prover::core::{
        fields::m31::BaseField,
        poly::circle::{CirclePoly, SecureCirclePoly},
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

        // let coset = Coset::get_coset_points(&Coset::new_standard(coeffs.len().ilog2() + 10))
        //     .iter()
        //     .skip(coeffs.len() * 2 + 1)
        //     .map(|point| FieldElement::<Degree2ExtensionField>::new([point.x, point.y]))
        //     .take(coeffs.len() + 1)
        //     .collect::<Vec<_>>();
        let circle_poly = CirclePoly::<CpuBackend>::new(vec![one; POLY_SIZE]);
        println!("circle_poly = {:#?}", circle_poly);
        let init_evals = evaluate_cfft(vec![FieldElement::<Mersenne31Field>::new(1); POLY_SIZE]);
        println!("init_evals = {:#?}", init_evals);

        // let poly = Polynomial::new(
        //     &coeffs
        //         .iter()
        //         .map(|(re, im)| {
        //             FieldElement::<Degree2ExtensionField>::new([
        //                 FieldElement::<Mersenne31Field>::new(re.0),
        //                 FieldElement::<Mersenne31Field>::new(im.0),
        //             ])
        //         })
        //         .collect::<Vec<_>>(),
        // );

        // let random_pos = coset;
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
        let interpolated = interpolate_cm31(&random_points, &random_evals);
        let interpolated = interpolated
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect::<Vec<_>>();

        // Compare results
        println!("interpolated.len(): {:?}", interpolated.len());
        println!("coeffs.len(): {:?}", coeffs.len());
        println!("coeffs: {:?}", &coeffs);
        println!();
        println!("interp: {:?}", &interpolated);

        for (i, ((bre, bim), (ire, iim))) in coeffs.iter().zip(interpolated.iter()).enumerate() {
            assert_eq!(bre, ire, "failed real part at {}", i);
            assert_eq!(bim, iim, "failed imaginary part at {}", i);
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

        let interpolated = interpolate_cm31(&random_points, &random_evals);
        println!("interpolated.len(): {:?}", interpolated.len());
        println!("coeffs.len(): {:?}", coeffs.len() / 2);
        println!("coeffs[..10]: {:?}", &coeffs[..10]);
        println!("interp[..10]: {:?}", &interpolated[..10]);
        assert_eq!(interpolated.len(), coeffs.len());
    }
    // #[test]
    // fn test_something() {
    //     let data = include_bytes!("../blob")[..].to_vec();
    //     let mut coeffs = frieda::utils::bytes_to_felt_le(&data);
    //     let next_power_of_2 = 1 << ((coeffs.len() as f64).log2().ceil() as u32).max(2);
    //     let next_power_of_2 = 2 * next_power_of_2;
    //     coeffs.resize(next_power_of_2, BaseField::from(0));
    //     let quarter_coeffs = &coeffs[..coeffs.len() / 4];
    //     let second_quarter_coeffs = &coeffs[coeffs.len() / 4..coeffs.len() / 2];
    //     let polys = SecureCirclePoly::<CpuBackend>([
    //         CirclePoly::<CpuBackend>::new(quarter_coeffs.to_vec()),
    //         CirclePoly::<CpuBackend>::new(second_quarter_coeffs.to_vec()),
    //         CirclePoly::<CpuBackend>::new(vec![BaseField::from(0); quarter_coeffs.len()]),
    //         CirclePoly::<CpuBackend>::new(vec![BaseField::from(0); quarter_coeffs.len()]),
    //     ]);
    //     // we should have polys.len().next_power_of_two() samples
    //     let samples_nb = (1 << polys.log_size().next_power_of_two()) / 20;
    //     println!("samples_nb: {:?}", samples_nb);
    //     let proofs_pos = (0..(samples_nb + 1))
    //         .into_par_iter()
    //         .map(|i| {
    //             let proof = generate_proof(&data, Some(i), PCS_CONFIG);
    //             let queries = get_queries_from_proof(proof.clone(), Some(i));
    //             (proof, queries)
    //         })
    //         .collect::<Vec<_>>();

    //     let domain = CircleDomain::new(SCoset::half_odds(proofs_pos[0].0.coset_log_size));
    //     let mut pos_set = HashSet::new();
    //     let mut pos_vec = Vec::new();
    //     let mut evals_vec = Vec::new();
    //     for (proof, (_log_size, pos)) in proofs_pos {
    //         for (i, p) in pos.iter().enumerate() {
    //             let point = domain.at(bit_reverse_index(*p, domain.log_size()));
    //             if pos_set.insert(point) {
    //                 pos_vec.push(point);
    //                 evals_vec.push(proof.evaluations[i]);
    //             }
    //         }
    //     }

    //     let pos = pos_vec;
    //     let evals = evals_vec;
    //     let evals_nb = (((samples_nb * 20) / 2).next_power_of_two() + 1) as usize;
    //     // Save the pos and evals to a file
    //     let pos_file = File::create("pos.json").unwrap();
    //     let evals_file = File::create("evals.json").unwrap();
    //     serde_json::to_writer(
    //         pos_file,
    //         &pos[..evals_nb]
    //             .iter()
    //             .map(|p| (p.x, p.y))
    //             .collect::<Vec<_>>(),
    //     )
    //     .unwrap();
    //     serde_json::to_writer(evals_file, &evals[..evals_nb]).unwrap();

    //     // Load the pos and evals from a file

    //     let pos_file = File::open("pos.json").unwrap();
    //     let evals_file = File::open("evals.json").unwrap();
    //     let pos: Vec<(M31, M31)> = serde_json::from_reader(pos_file).unwrap();
    //     let pos = pos
    //         .into_iter()
    //         .map(|(x, y)| CirclePoint { x, y })
    //         .collect::<Vec<_>>();
    //     let evals: Vec<QM31> = serde_json::from_reader(evals_file).unwrap();
    //     let interpolated = polynomial_from_felts(interpolate_qm31(&pos, &evals));
    //     // let evals = CirclePoly::<CpuBackend>::new(vec![CM31::from_u32_unchecked(1, 1)])
    //     //     .evaluate(CircleDomain::new(CanonicCoset::new(2).half_coset()));
    //     println!("interpolated.len(): {:?}", interpolated.0[0].coeffs.len());
    //     let mut coeff_init = polys.0[0].coeffs.clone();
    //     coeff_init.reverse();
    //     coeff_init = coeff_init.into_iter().skip_while(|v| v.is_zero()).collect();
    //     coeff_init.reverse();
    //     println!("polys.len(): {:?}", coeff_init.len());
    //     // assert_eq!(interpolated.0[0].coeffs, polys.0[0].coeffs);
    //     // assert_eq!(interpolated.0[1].coeffs, polys.0[1].coeffs);
    //     // assert_eq!(interpolated.0[2].coeffs, polys.0[2].coeffs);
    //     // assert_eq!(interpolated.0[3].coeffs, polys.0[3].coeffs);
    //    }
}
