use lambdaworks_math::circle::cosets::Coset;

use lambdaworks_math::circle::polynomial::interpolate_cfft;
use lambdaworks_math::fft::cpu::ops;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::extensions::Degree2ExtensionField;

use lambdaworks_math::polynomial::Polynomial;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use stwo_prover::core::circle::CirclePoint;

use stwo_prover::core::fields::{cm31::CM31, m31::M31, qm31::QM31};

pub fn interpolate_qm31(pks: &[CirclePoint<M31>], extvks: &[QM31]) -> Vec<M31> {
    println!("pks: {:?}", pks.len());
    println!("extvks: {:?}", extvks.len());
    let _zks = &pks
        .iter()
        .map(|zk| CM31::from_m31(zk.x, zk.y))
        .collect::<Vec<_>>();
    // let first_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.0).collect::<Vec<_>>());
    // let second_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.1).collect::<Vec<_>>());

    // first_poly
    //     .into_iter()
    //     .zip(second_poly)
    //     .map(|(a, b)| a + b)
    //     .collect::<Vec<_>>()
    vec![]
}
pub fn interpolate_cm31(
    zks: &[FieldElement<Degree2ExtensionField>],
    vks: &[FieldElement<Degree2ExtensionField>],
) -> Vec<M31> {
    println!("zks len = {}", zks.len());
    assert_eq!(zks.len(), vks.len());
    assert!((zks.len() - 1).is_power_of_two());
    println!("zks = {:#?}", zks);
    println!("vks = {:#?}", vks);

    let n = zks.len() - 1;
    println!("computing vprimes");
    let vprimes = vks
        .into_par_iter()
        .zip_eq(zks)
        .map(|(vk, zk)| vk.clone() * zk.pow(n as u128 / 2))
        .collect::<Vec<_>>();
    println!("vprimes = {:#?}", vprimes);
    println!("computing zks");
    let pol = fast_interpolation(zks, &vprimes);

    println!("pol = {:#?}", pol);
    assert_eq!(
        zks.iter()
            .map(|z| { ((pol.evaluate(z)) / z.pow(n as u128 / 2)).unwrap() })
            .collect::<Vec<_>>(),
        vks
    );
    assert_eq!(
        zks.iter().map(|z| pol.evaluate(z)).collect::<Vec<_>>(),
        vprimes
    );

    // let last_term = pol.coefficients.pop().unwrap();
    // println!("last term = {:?}", last_term);
    // println!("computing domain");
    // let domain = Radix2EvaluationDomain::<ComplexCircleFp>::new(pol.coeffs.len()).unwrap();
    println!("poly len = {}", pol.coefficients.len());
    let coset = Coset::new_standard(n.ilog2());
    let points = Coset::get_coset_points(&coset);
    assert_eq!(points.len(), n);
    println!("points = {:#?}", points);
    println!("evaluating");
    // assert_eq!(pol.coefficients.len(), n);
    // Evaluate p(z) (or p(z) - c_N * z^N, with c_N added back as last_term) on the domain.
    // let evals: Vec<ComplexCircleFp> = domain.fft(&pol.coeffs);
    let f_values = points
        .iter()
        .map(|z| {
            let z = FieldElement::<Degree2ExtensionField>::new([z.x, z.y]);
            ((pol.evaluate(&z)) / z.pow(n as u128 / 2)).unwrap()
        })
        .collect::<Vec<_>>();
    println!("f_values = {:#?}", f_values);
    println!("adjusting");

    println!("N = {}", n);
    println!("{} evals", f_values.len());
    println!("packing");

    let re_col = f_values.iter().map(|x| x.value()[0]).collect::<Vec<_>>();

    let im_col = f_values.iter().map(|x| x.value()[1]).collect::<Vec<_>>();
    println!("fft");
    println!("re_col len = {}", re_col.len());
    println!("re_col log len = {}", re_col.len().ilog2());
    println!(
        "re col len trailing zeros = {}",
        re_col.len().trailing_zeros()
    );
    println!("im_col len = {}", im_col.len());
    println!("im_col log len = {}", im_col.len().ilog2());
    println!(
        "im_col len trailing zeros = {}",
        im_col.len().trailing_zeros()
    );
    println!("interpolating");
    let re_fft = interpolate_cfft(re_col);
    let im_fft = interpolate_cfft(im_col);
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
    let vanishing_poly = tree.last().unwrap()[0].clone();

    let lambda = (v0 - poly.evaluate(z0)) * vanishing_poly.evaluate(z0).inv().unwrap();

    assert_eq!(
        &lambda * vanishing_poly.evaluate(z0) + poly.evaluate(z0),
        *v0
    );

    poly + vanishing_poly * lambda
}

pub fn build_subproduct_tree(
    u: &[FieldElement<Degree2ExtensionField>],
) -> Vec<Vec<Polynomial<FieldElement<Degree2ExtensionField>>>> {
    // Level 0: polynomials of the form (x - u[i])
    let k = u.len().ilog2() as usize;
    let mut tree: Vec<Vec<Polynomial<FieldElement<Degree2ExtensionField>>>> = Vec::with_capacity(k);

    tree.push(
        u.iter()
            .map(|ui| Polynomial::new(&[-ui.clone(), FieldElement::<Degree2ExtensionField>::one()]))
            .collect(),
    );

    for i in 1..=k {
        tree.push(Vec::with_capacity(1 << (k - i)));
        for j in 0..(1 << (k - i)) {
            let val = tree[i - 1][2 * j].mul_with_ref(&tree[i - 1][2 * j + 1]);
            tree[i].push(val);
        }
    }
    tree
}

pub fn eval_tree(
    f: &Polynomial<FieldElement<Degree2ExtensionField>>,
    tree: &Vec<Vec<Polynomial<FieldElement<Degree2ExtensionField>>>>,
    points: &[FieldElement<Degree2ExtensionField>],
    index: usize,
) -> Vec<FieldElement<Degree2ExtensionField>> {
    if points.len() == 1 {
        return vec![
            f.coefficients()
                .first()
                .unwrap_or(&FieldElement::<Degree2ExtensionField>::zero())
                .clone(),
        ];
    }
    let k = points.len().ilog2() as usize;

    let left_poly = &tree[k - 1][2 * index];
    let right_poly = &tree[k - 1][2 * index + 1];
    // Compute remainders of f modulo the children.
    let ((_, r_left), (_, r_right)) = rayon::join(
        || f.clone().long_division_with_remainder(left_poly),
        || f.clone().long_division_with_remainder(right_poly),
    );

    // Split the points into two halves.
    let mid = points.len() / 2;
    let left_points = &points[..mid];
    let right_points = &points[mid..];
    let (left_vals, right_vals) = rayon::join(
        || eval_tree(&r_left, tree, left_points, 2 * index),
        || eval_tree(&r_right, tree, right_points, 2 * index + 1),
    );
    let mut result = left_vals;
    result.extend(right_vals);
    result
}

pub fn linear_combination(
    u: &[FieldElement<Degree2ExtensionField>],
    c: &[FieldElement<Degree2ExtensionField>],
    subproduct: &Vec<Vec<Polynomial<FieldElement<Degree2ExtensionField>>>>,
    level: usize,
    index: usize,
) -> Polynomial<FieldElement<Degree2ExtensionField>> {
    if c.len() == 1 {
        // Base case: return the constant polynomial equal to c[0].
        return Polynomial::new(c);
    }
    let child_level = level - 1;
    let mid = u.len() / 2;

    let (r0, r1) = rayon::join(
        || linear_combination(&u[..mid], &c[..mid], subproduct, child_level, 2 * index),
        || linear_combination(&u[mid..], &c[mid..], subproduct, child_level, 2 * index + 1),
    );

    let left_poly = &subproduct[child_level][2 * index]; // corresponds to left half
    let right_poly = &subproduct[child_level][2 * index + 1]; // corresponds to right half
    let (term1, term2) = rayon::join(
        || right_poly.mul_with_ref(&r0),
        || left_poly.mul_with_ref(&r1),
    );
    term1 + term2
}

#[cfg(test)]
mod tests {
    use frieda::{
        api::generate_proof, proof::get_queries_from_proof, utils::polynomial_from_felts,
    };
    use lambdaworks_math::field::fields::mersenne31::{
        extensions::Degree2ExtensionField, field::Mersenne31Field,
    };
    use num_traits::Zero;
    use rand::Rng;
    use std::{collections::HashSet, fs::File};
    use stwo_prover::core::{
        backend::CpuBackend,
        circle::Coset as SCoset,
        fields::m31::BaseField,
        fri::FriConfig,
        pcs::PcsConfig,
        poly::circle::{CircleDomain, CirclePoly, SecureCirclePoly},
        utils::bit_reverse_index,
    };

    use super::*;
    const PCS_CONFIG: PcsConfig = PcsConfig {
        fri_config: FriConfig {
            log_blowup_factor: 4,
            log_last_layer_degree_bound: 1,
            n_queries: 20,
        },
        pow_bits: 20,
    };

    #[test]
    fn test_simple_interpolation() {
        // Create a simple polynomial with known coefficients
        // Each coefficient is a pair of field elements (real, imaginary)
        let coeffs = vec![
            (BaseField::from(1), BaseField::from(0)),
            (BaseField::from(1), BaseField::from(0)),
            (BaseField::from(1), BaseField::from(0)),
            (BaseField::from(1), BaseField::from(0)),
        ];

        println!("coeffs.len(): {:?}", coeffs.len());

        let coset = Coset::get_coset_points(&Coset::conjugate(Coset::new_standard(
            coeffs.len().ilog2() + 1,
        )));

        // First randomly select indices
        let mut selected_indices = Vec::new();
        let mut used_indices = HashSet::new();
        let mut index = 0;
        while selected_indices.len() <= coeffs.len() {
            if used_indices.insert(index) {
                selected_indices.push(index);
            }
            index += 1;
        }

        let poly = Polynomial::new(
            &coeffs
                .iter()
                .map(|(re, im)| {
                    FieldElement::<Degree2ExtensionField>::new([
                        FieldElement::<Mersenne31Field>::new(re.0),
                        FieldElement::<Mersenne31Field>::new(im.0),
                    ])
                })
                .collect::<Vec<_>>(),
        );
        let random_pos = selected_indices
            .into_iter()
            .map(|position| {
                FieldElement::<Degree2ExtensionField>::new([coset[position].x, coset[position].y])
            })
            .collect::<Vec<_>>();
        let random_evals = random_pos
            .par_iter()
            .map(|position| poly.evaluate(position))
            .collect::<Vec<_>>();

        // Interpolate back
        let interpolated = interpolate_cm31(&random_pos, &random_evals);
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
        let data = include_bytes!("../blob")[..(1024. * 3.75) as usize].to_vec();
        let mut coeffs = frieda::utils::bytes_to_felt_le(&data);
        let next_power_of_2 = 1 << ((coeffs.len() as f64).log2().ceil() as u32).max(2);

        coeffs.resize(next_power_of_2, BaseField::from(1));
        // evaluate the polynomial at the even coefficients

        let coset = Coset::get_coset_points(&Coset::new_standard(coeffs.len().ilog2() + 3));
        let mut rng = rand::rng();

        // First randomly select indices
        let mut selected_indices = Vec::new();
        let mut used_indices = HashSet::new();

        while selected_indices.len() < coeffs.len() / 2 + 1 {
            let index = rng.random_range(0..(1 << (coeffs.len().ilog2() + 3)));
            if used_indices.insert(index) {
                selected_indices.push(index);
            }
        }

        selected_indices.sort();
        let poly = Polynomial::new(
            &coeffs
                .chunks(2)
                .map(|chunk| {
                    FieldElement::<Degree2ExtensionField>::new([
                        FieldElement::<Mersenne31Field>::new(chunk[0].0),
                        FieldElement::<Mersenne31Field>::new(chunk[1].0),
                    ])
                })
                .collect::<Vec<_>>(),
        );
        let random_pos = selected_indices
            .into_iter()
            .map(|position| {
                FieldElement::<Degree2ExtensionField>::new([coset[position].x, coset[position].y])
            })
            .collect::<Vec<_>>();
        let random_evals = random_pos
            .par_iter()
            .map(|position| poly.evaluate(position))
            .collect::<Vec<_>>();
        // Get positions for selected indices

        // pick coeffs.len() + 1 randomevaluations

        // let pos_file = File::create("pos.json").unwrap();
        // let evals_file = File::create("evals.json").unwrap();
        // serde_json::to_writer(
        //     pos_file,
        //     &random_pos.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>(),
        // )
        // .unwrap();
        // serde_json::to_writer(
        //     evals_file,
        //     &random_evals
        //         .iter()
        //         .map(|eval| (eval.0.0, eval.1.0))
        //         .collect::<Vec<_>>(),
        // )
        // .unwrap();
        // let pos: Vec<(M31, M31)> =
        //     serde_json::from_reader(File::open("pos.json").unwrap()).unwrap();
        // let evals: Vec<(M31, M31)> =
        //     serde_json::from_reader(File::open("evals.json").unwrap()).unwrap();

        let interpolated = interpolate_cm31(&random_pos, &random_evals);
        println!("interpolated.len(): {:?}", interpolated.len());
        println!("coeffs.len(): {:?}", coeffs.len() / 2);
        println!("coeffs[..10]: {:?}", &coeffs[..10]);
        println!("interp[..10]: {:?}", &interpolated[..10]);
        assert_eq!(interpolated.len(), coeffs.len());
    }
    #[test]
    fn test_something() {
        let data = include_bytes!("../blob")[..].to_vec();
        let mut coeffs = frieda::utils::bytes_to_felt_le(&data);
        let next_power_of_2 = 1 << ((coeffs.len() as f64).log2().ceil() as u32).max(2);
        let next_power_of_2 = 2 * next_power_of_2;
        coeffs.resize(next_power_of_2, BaseField::from(0));
        let quarter_coeffs = &coeffs[..coeffs.len() / 4];
        let second_quarter_coeffs = &coeffs[coeffs.len() / 4..coeffs.len() / 2];
        let polys = SecureCirclePoly::<CpuBackend>([
            CirclePoly::<CpuBackend>::new(quarter_coeffs.to_vec()),
            CirclePoly::<CpuBackend>::new(second_quarter_coeffs.to_vec()),
            CirclePoly::<CpuBackend>::new(vec![BaseField::from(0); quarter_coeffs.len()]),
            CirclePoly::<CpuBackend>::new(vec![BaseField::from(0); quarter_coeffs.len()]),
        ]);
        // we should have polys.len().next_power_of_two() samples
        let samples_nb = (1 << polys.log_size().next_power_of_two()) / 20;
        println!("samples_nb: {:?}", samples_nb);
        let proofs_pos = (0..(samples_nb + 1))
            .into_par_iter()
            .map(|i| {
                let proof = generate_proof(&data, Some(i), PCS_CONFIG);
                let queries = get_queries_from_proof(proof.clone(), Some(i));
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
        let evals_nb = (((samples_nb * 20) / 2).next_power_of_two() + 1) as usize;
        // Save the pos and evals to a file
        let pos_file = File::create("pos.json").unwrap();
        let evals_file = File::create("evals.json").unwrap();
        serde_json::to_writer(
            pos_file,
            &pos[..evals_nb]
                .iter()
                .map(|p| (p.x, p.y))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        serde_json::to_writer(evals_file, &evals[..evals_nb]).unwrap();

        // Load the pos and evals from a file

        let pos_file = File::open("pos.json").unwrap();
        let evals_file = File::open("evals.json").unwrap();
        let pos: Vec<(M31, M31)> = serde_json::from_reader(pos_file).unwrap();
        let pos = pos
            .into_iter()
            .map(|(x, y)| CirclePoint { x, y })
            .collect::<Vec<_>>();
        let evals: Vec<QM31> = serde_json::from_reader(evals_file).unwrap();
        let interpolated = polynomial_from_felts(interpolate_qm31(&pos, &evals));
        // let evals = CirclePoly::<CpuBackend>::new(vec![CM31::from_u32_unchecked(1, 1)])
        //     .evaluate(CircleDomain::new(CanonicCoset::new(2).half_coset()));
        println!("interpolated.len(): {:?}", interpolated.0[0].coeffs.len());
        let mut coeff_init = polys.0[0].coeffs.clone();
        coeff_init.reverse();
        coeff_init = coeff_init.into_iter().skip_while(|v| v.is_zero()).collect();
        coeff_init.reverse();
        println!("polys.len(): {:?}", coeff_init.len());
        // assert_eq!(interpolated.0[0].coeffs, polys.0[0].coeffs);
        // assert_eq!(interpolated.0[1].coeffs, polys.0[1].coeffs);
        // assert_eq!(interpolated.0[2].coeffs, polys.0[2].coeffs);
        // assert_eq!(interpolated.0[3].coeffs, polys.0[3].coeffs);
    }
}
