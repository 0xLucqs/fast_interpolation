use std::mem::transmute;

use ark_ff::{Field, One, Zero};
use ark_ff_optimized::ccfp31::ComplexCircleFp;
use ark_ff_optimized::fp31::Fp;
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
use ark_poly::{DenseUVPolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain};
use lambdaworks_math::circle::cfft::{icfft, order_icfft_input_naive};
use lambdaworks_math::circle::cosets::Coset;

use lambdaworks_math::circle::twiddles::{TwiddlesConfig, get_twiddles};
use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use stwo_prover::core::backend::CpuBackend;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::simd::bit_reverse::bit_reverse_m31;
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::fft::ifft::ifft;
use stwo_prover::core::backend::simd::fft::rfft::get_twiddle_dbls;
use stwo_prover::core::backend::simd::m31::PackedBaseField;
use stwo_prover::core::circle::CirclePoint;
use stwo_prover::core::circle::Coset as SCoset;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::{FieldExpOps, cm31::CM31, m31::M31, qm31::QM31};
use stwo_prover::core::poly::circle::{CanonicCoset, CircleDomain, CirclePoly, PolyOps};
use stwo_prover::core::poly::utils::domain_line_twiddles_from_tree;
use stwo_prover::core::utils::bit_reverse_coset_to_circle_domain_order;

pub fn interpolate_qm31(pks: &[CirclePoint<M31>], extvks: &[QM31]) -> Vec<M31> {
    println!("pks: {:?}", pks.len());
    println!("extvks: {:?}", extvks.len());
    let zks = &pks
        .iter()
        .map(|zk| CM31::from_m31(zk.x, zk.y))
        .collect::<Vec<_>>();
    let first_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.0).collect::<Vec<_>>());
    let second_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.1).collect::<Vec<_>>());

    first_poly
        .into_iter()
        .zip(second_poly)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>()
}
pub fn interpolate_cm31(zks: &[CM31], vks: &[CM31]) -> Vec<M31> {
    println!("zks len = {}", zks.len());
    assert_eq!(zks.len(), vks.len());
    assert!((zks.len() - 1).is_power_of_two());

    let n = zks.len() - 1;
    println!("computing vprimes");
    let vprimes = vks
        .into_par_iter()
        .zip_eq(zks)
        .map(|(vk, zk)| {
            let res = *vk * zk.pow(n as u128 / 2);
            ComplexCircleFp(Fp(res.0.0), Fp(res.1.0))
        })
        .collect::<Vec<_>>();
    println!("computing zks");
    let zks = zks
        .iter()
        .map(|zk| ComplexCircleFp(Fp(zk.0.0), Fp(zk.1.0)))
        .collect::<Vec<_>>();
    let mut pol = fast_interpolation(&zks, &vprimes);
    // println!("computing last term");
    let last_term = pol.coeffs.pop().unwrap();
    // println!("computing domain");
    // let domain = Radix2EvaluationDomain::<ComplexCircleFp>::new(pol.coeffs.len()).unwrap();
    println!("poly len = {}", pol.coeffs.len());
    let coset = CanonicCoset::new(pol.coeffs.len().ilog2() - 1).coset();
    let domain = CircleDomain::new(coset);
    println!("computing points");
    // Get the evaluation points (the elements of the domain).
    let points = domain.iter().collect::<Vec<_>>();
    println!("evaluating");
    // Evaluate p(z) (or p(z) - c_N * z^N, with c_N added back as last_term) on the domain.
    // let evals: Vec<ComplexCircleFp> = domain.fft(&pol.coeffs);
    let evals = points
        .par_iter()
        .map(|z| pol.evaluate(&ComplexCircleFp(Fp(z.x.0), Fp(z.y.0))) - last_term)
        .collect::<Vec<_>>();
    println!("adjusting");
    // Now adjust each evaluation by dividing by z^(N/2) pointwise.
    let f_values: Vec<ComplexCircleFp> = evals
        .into_par_iter()
        .zip(points.into_par_iter())
        .map(|(eval, z)| {
            // Compute z^(N/2). Here N is pol.coeffs.len(), so N/2 is pol.coeffs.len()/2.
            // Make sure pol.coeffs.len() is a power-of-two.
            let exponent = n / 2;
            let z_n_over_2 = ComplexCircleFp(Fp(z.x.0), Fp(z.y.0)).pow([exponent as u64]);
            // Divide the evaluated value (with last_term added) by z^(N/2)
            eval / z_n_over_2
        })
        .collect();
    println!("N = {}", n);
    println!("{} evals", f_values.len());
    println!("packing");
    let mut re_col = f_values
        .iter()
        .map(|x| FieldElement::<Mersenne31Field>::new(x.0.0))
        .collect::<Vec<_>>();

    let mut im_col = f_values
        .iter()
        .map(|x| FieldElement::<Mersenne31Field>::new(x.1.0))
        .collect::<Vec<_>>();
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

    re_fft
        .into_iter()
        .zip(im_fft)
        .flat_map(|(re, im)| {
            [
                M31::from_u32_unchecked(re.to_raw()),
                M31::from_u32_unchecked(im.to_raw()),
            ]
        })
        .collect::<Vec<_>>()
}

pub fn bricolfft(mut values: BaseColumn) -> Vec<BaseField> {
    let coset = CanonicCoset::new(values.data.len().ilog2() - 1).coset();
    let twiddles = SimdBackend::precompute_twiddles(coset);

    let twiddles = domain_line_twiddles_from_tree(CircleDomain::new(coset), &twiddles.itwiddles);

    // Safe because [PackedBaseField] is aligned on 64 bytes.
    unsafe {
        ifft(
            transmute::<*mut PackedBaseField, *mut u32>(values.data.as_mut_ptr()),
            &twiddles,
            values.data.len().ilog2() as usize,
        );
    }

    // TODO(alont): Cache this inversion.
    let inv =
        PackedBaseField::broadcast(BaseField::from(CircleDomain::new(coset).size()).inverse());
    values.data.iter_mut().for_each(|x| *x *= inv);
    values.into_cpu_vec()
}

pub fn fast_interpolation(
    u: &[ComplexCircleFp],
    v: &[ComplexCircleFp],
) -> DensePolynomial<ComplexCircleFp> {
    let z0 = u[0];
    let v0 = v[0];
    let u = &u[1..];
    let v = &v[1..];
    assert_eq!(u.len(), v.len());
    assert!(u.len().is_power_of_two());

    let tree = build_subproduct_tree(u);

    let k = u.len().ilog2() as usize;
    let m = &tree[k][0];

    let m_deriv = differentiate(m);
    let m_deriv_vals = eval_tree(&m_deriv, &tree, u, 0);

    let c: Vec<ComplexCircleFp> = v
        .par_iter()
        .zip(m_deriv_vals.par_iter())
        .map(|(vi, m_prime_ui)| vi * &m_prime_ui.inverse().unwrap())
        .collect();

    let poly = linear_combination(u, &c, &tree, k, 0);
    let vanishing_poly = tree.last().unwrap()[0].clone();

    let lambda = (v0 - poly.evaluate(&z0)) * vanishing_poly.evaluate(&z0).inverse().unwrap();

    assert_eq!(
        lambda * vanishing_poly.evaluate(&z0) + poly.evaluate(&z0),
        v0
    );

    poly + vanishing_poly * lambda
}

fn interpolate_cfft(
    eval: Vec<FieldElement<Mersenne31Field>>,
) -> Vec<FieldElement<Mersenne31Field>> {
    let mut eval = eval;

    if eval.is_empty() {
        let poly: Vec<FieldElement<Mersenne31Field>> = Vec::new();
        return poly;
    }

    // We get the twiddles for the interpolation.
    let domain_log_2_size: u32 = eval.len().trailing_zeros();
    let coset = CanonicCoset::new(domain_log_2_size).coset();
    let twiddles = get_twiddle_dbls(coset)
        .into_iter()
        .map(|x| {
            x.into_iter()
                .map(FieldElement::<Mersenne31Field>::new)
                .collect()
        })
        .collect();

    // For our algorithm to work, we must give as input the evaluations ordered in a certain way.
    let mut eval_ordered = order_icfft_input_naive(&mut eval);
    icfft(&mut eval_ordered, twiddles);

    // The icfft returns the polynomial coefficients in bit reverse order. So we premute it to get the natural order.
    in_place_bit_reverse_permute::<FieldElement<Mersenne31Field>>(&mut eval_ordered);

    // The icfft returns all the coefficients multiplied by 2^n, the length of the evaluations.
    // So we multiply every element that outputs the icfft by the inverse of 2^n to get the actual coefficients.
    // Note that this `unwrap` will never panic because eval.len() != 0.
    let factor = (FieldElement::<Mersenne31Field>::from(eval.len() as u64))
        .inv()
        .unwrap();
    eval_ordered.iter().map(|coef| coef * factor).collect()
}

pub fn build_subproduct_tree(u: &[ComplexCircleFp]) -> Vec<Vec<DensePolynomial<ComplexCircleFp>>> {
    // Level 0: polynomials of the form (x - u[i])
    let k = u.len().ilog2() as usize;
    let mut tree: Vec<Vec<DensePolynomial<ComplexCircleFp>>> = Vec::with_capacity(k);

    tree.push(
        u.iter()
            .map(|ui| DensePolynomial::from_coefficients_slice(&[-*ui, ComplexCircleFp::one()]))
            .collect(),
    );

    for i in 1..=k {
        tree.push(Vec::with_capacity(1 << (k - i)));
        for j in 0..(1 << (k - i)) {
            let val = tree[i - 1][2 * j].naive_mul(&tree[i - 1][2 * j + 1]);
            tree[i].push(val);
        }
    }
    tree
}

pub fn eval_tree(
    f: &DensePolynomial<ComplexCircleFp>,
    tree: &Vec<Vec<DensePolynomial<ComplexCircleFp>>>,
    points: &[ComplexCircleFp],
    index: usize,
) -> Vec<ComplexCircleFp> {
    if points.len() == 1 {
        return vec![*f.coeffs.first().unwrap_or(&ComplexCircleFp::zero())];
    }
    let k = points.len().ilog2() as usize;

    let left_poly = &tree[k - 1][2 * index];
    let right_poly = &tree[k - 1][2 * index + 1];
    // Compute remainders of f modulo the children.
    let (_q_left, r_left) = DenseOrSparsePolynomial::from(f)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(left_poly))
        .unwrap();
    let (_q_right, r_right) = DenseOrSparsePolynomial::from(f)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(right_poly))
        .unwrap();
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
    u: &[ComplexCircleFp],
    c: &[ComplexCircleFp],
    subproduct: &Vec<Vec<DensePolynomial<ComplexCircleFp>>>,
    level: usize,
    index: usize,
) -> DensePolynomial<ComplexCircleFp> {
    if c.len() == 1 {
        // Base case: return the constant polynomial equal to c[0].
        return DensePolynomial::from_coefficients_slice(c);
    }
    let child_level = level - 1;
    let mid = u.len() / 2;

    let (r0, r1) = rayon::join(
        || linear_combination(&u[..mid], &c[..mid], subproduct, child_level, 2 * index),
        || linear_combination(&u[mid..], &c[mid..], subproduct, child_level, 2 * index + 1),
    );

    let left_poly = &subproduct[child_level][2 * index]; // corresponds to left half
    let right_poly = &subproduct[child_level][2 * index + 1]; // corresponds to right half
    let term1 = right_poly.naive_mul(&r0);
    let term2 = left_poly.naive_mul(&r1);
    term1 + term2
}

pub fn differentiate(poly: &DensePolynomial<ComplexCircleFp>) -> DensePolynomial<ComplexCircleFp> {
    if poly.coeffs.len() <= 1 {
        return DensePolynomial::from_coefficients_vec(vec![ComplexCircleFp::zero()]);
    }
    let deriv_coeffs: Vec<ComplexCircleFp> = poly
        .coeffs
        .par_iter()
        .enumerate()
        .skip(1)
        .map(|(i, coeff)| *coeff * ComplexCircleFp(Fp(i as u32), Fp(0)))
        .collect();
    DensePolynomial::from_coefficients_vec(deriv_coeffs)
}
#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs::File};

    use frieda::{
        api::generate_proof, proof::get_queries_from_proof, utils::polynomial_from_felts,
    };
    use rand::Rng;
    use stwo_prover::core::{
        circle::Coset as SCoset,
        fri::FriConfig,
        pcs::PcsConfig,
        poly::circle::{CircleDomain, SecureCirclePoly},
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
    fn test_interpolate_cm31() {
        let data = include_bytes!("../blob")[..].to_vec();
        let mut coeffs = frieda::utils::bytes_to_felt_le(&data);
        let next_power_of_2 = 1 << ((coeffs.len() as f64).log2().ceil() as u32).max(2);

        coeffs.resize(next_power_of_2, BaseField::from(1));
        let complex_coeffs = coeffs
            .chunks(2)
            .map(|chunk| ComplexCircleFp(Fp(chunk[0].0), Fp(chunk[1].0)))
            .collect::<Vec<_>>();
        let poly = DensePolynomial::from_coefficients_vec(complex_coeffs);
        println!("poly degree: {:?}", poly.degree());
        let circle_domain = CircleDomain::new(SCoset::half_odds(coeffs.len().ilog2() + 3));

        let points = circle_domain.iter().collect::<Vec<_>>();
        let evals = points
            .par_iter()
            .map(|point| poly.evaluate(&ComplexCircleFp(Fp(point.x.0), Fp(point.y.0))))
            .collect::<Vec<_>>();
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

        let mut rng = rand::rng();
        let mut random_evals = Vec::new();
        let mut random_pos = Vec::new();
        let mut used_indices = HashSet::new();
        while random_evals.len() < poly.coeffs.len() + 1 {
            let index = rng.random_range(0..evals.len());
            if used_indices.insert(index) {
                random_evals.push(evals[index]);
                random_pos.push(circle_domain.at(index));
            }
        }
        let interpolated = interpolate_cm31(
            &random_pos
                .iter()
                .map(|pos| CM31::from_m31(pos.x, pos.y))
                .collect::<Vec<_>>(),
            &random_evals
                .iter()
                .map(|eval| CM31::from_u32_unchecked(eval.0.0, eval.1.0))
                .collect::<Vec<_>>(),
        );
        println!("interpolated.len(): {:?}", interpolated.len());
        println!("coeffs.len(): {:?}", coeffs.len());
        println!("coeffs[..30]: {:?}", &coeffs[..30]);
        println!("interpolated[..30]: {:?}", &interpolated[..30]);
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
