use lambdaworks_math::fft::polynomial::evaluate_fft_cpu;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::mersenne31::{
    extensions::Degree2ExtensionField, field::Mersenne31Field,
};
use lambdaworks_math::polynomial::Polynomial;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use stwo_prover::core::circle::CirclePoint;
use stwo_prover::core::fields::{FieldExpOps, cm31::CM31, m31::M31, qm31::QM31};
type LambdaCM31 = FieldElement<Degree2ExtensionField>;
type LambdaM31 = FieldElement<Mersenne31Field>;

pub fn interpolate_qm31(pks: &[CirclePoint<M31>], extvks: &[QM31]) -> Vec<QM31> {
    println!("pks: {:?}", pks.len());
    println!("extvks: {:?}", extvks.len());
    let zks = &pks
        .iter()
        .map(|zk| CM31::from_m31(zk.x, zk.y))
        .collect::<Vec<_>>();
    let first_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.0).collect::<Vec<_>>());
    let second_poly = interpolate_cm31(zks, &extvks.iter().map(|vk| vk.1).collect::<Vec<_>>());

    first_poly
        .coefficients
        .into_iter()
        .zip(second_poly.coefficients)
        .map(|(a, b)| {
            QM31::from_u32_unchecked(
                a.clone().to_raw()[0].to_raw(),
                a.clone().to_raw()[1].to_raw(),
                b.clone().to_raw()[0].to_raw(),
                b.clone().to_raw()[1].to_raw(),
            )
        })
        .collect::<Vec<_>>()
}
pub fn interpolate_cm31(zks: &[CM31], vks: &[CM31]) -> Polynomial<LambdaCM31> {
    // assert_eq!(zks.len(), vks.len());
    // assert!((zks.len() - 1).is_power_of_two());
    let n = zks.len() - 1;
    let vprimes = vks
        .into_par_iter()
        .zip_eq(zks)
        .map(|(vk, zk)| *vk * zk.pow(n as u128 / 2))
        .collect::<Vec<_>>();
    let vprimes = vprimes
        .iter()
        .map(|vp| LambdaCM31::new([LambdaM31::new(vp.0.0), LambdaM31::new(vp.1.0)]))
        .collect::<Vec<_>>();
    let zks = zks
        .iter()
        .map(|zk| LambdaCM31::new([LambdaM31::new(zk.0.0), LambdaM31::new(zk.1.0)]))
        .collect::<Vec<_>>();
    fast_interpolation(&zks, &vprimes)
}

pub fn fast_interpolation(u: &[LambdaCM31], v: &[LambdaCM31]) -> Polynomial<LambdaCM31> {
    assert_eq!(u.len(), v.len());
    assert!(u.len().is_power_of_two());

    let tree = build_subproduct_tree(u);

    let k = u.len().ilog2() as usize;
    let m = &tree[k][0]; // m(x) = ∏ (x-u_i)

    let m_deriv = m.differentiate();
    let m_deriv_vals = eval_tree(&m_deriv, &tree, u, 0);

    let c: Vec<LambdaCM31> = v
        .par_iter()
        .zip(m_deriv_vals.par_iter())
        .map(|(vi, m_prime_ui)| vi * m_prime_ui.inv().unwrap())
        .collect();

    let pol = linear_combination(u, &c, &tree, k, 0);
    // let pol = pol - c.last().unwrap() * u.last().unwrap();
    let evals =
        evaluate_fft_cpu::<Mersenne31Field, Degree2ExtensionField>(&pol.coefficients).unwrap();
    println!("evals: {:?}", evals.len());
    pol
}

pub fn build_subproduct_tree(u: &[LambdaCM31]) -> Vec<Vec<Polynomial<LambdaCM31>>> {
    // Level 0: polynomials of the form (x - u[i])
    let k = u.len().ilog2() as usize;
    let mut tree: Vec<Vec<Polynomial<LambdaCM31>>> = Vec::with_capacity(k);

    tree.push(
        u.iter()
            .map(|ui| Polynomial::new(&[-ui, LambdaCM31::one()]))
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
    f: &Polynomial<LambdaCM31>,
    tree: &Vec<Vec<Polynomial<LambdaCM31>>>,
    points: &[LambdaCM31],
    index: usize,
) -> Vec<LambdaCM31> {
    if points.len() == 1 {
        return vec![f.coefficients[0].clone()];
    }
    let k = points.len().ilog2() as usize;

    let left_poly = &tree[k - 1][2 * index];
    let right_poly = &tree[k - 1][2 * index + 1];
    // Compute remainders of f modulo the children.
    let (_q_left, r_left) = f.clone().long_division_with_remainder(left_poly);
    let (_q_right, r_right) = f.clone().long_division_with_remainder(right_poly);
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
    u: &[LambdaCM31],
    c: &[LambdaCM31],
    subproduct: &Vec<Vec<Polynomial<LambdaCM31>>>,
    level: usize,
    index: usize,
) -> Polynomial<LambdaCM31> {
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
    let term1 = right_poly.mul_with_ref(&r0);
    let term2 = left_poly.mul_with_ref(&r1);
    term1 + term2
}

pub fn poly_derivative(poly: &Polynomial<LambdaCM31>) -> Polynomial<LambdaCM31> {
    if poly.coefficients.len() <= 1 {
        return Polynomial::new(&[LambdaCM31::zero()]);
    }
    let deriv_coeffs = poly
        .coefficients
        .par_iter()
        .enumerate()
        .skip(1)
        .map(|(i, coeff)| LambdaCM31::from(i as u64) * coeff)
        .collect::<Vec<_>>();
    Polynomial::new(&deriv_coeffs)
}
#[cfg(test)]
mod tests {
    use frieda::{api::generate_proof, proof::get_queries_from_proof};
    use stwo_prover::core::{
        circle::{CirclePoint, Coset},
        fri::FriConfig,
        pcs::PcsConfig,
        poly::circle::CircleDomain,
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
    fn test_interpolate_qm31() {
        let data = include_bytes!("../blob")[..1000].to_vec();
        let polys = frieda::utils::polynomial_from_bytes(&data);
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
        let domain = CircleDomain::new(Coset::half_odds(proofs_pos[0].0.coset_log_size));
        let pos: Vec<CirclePoint<M31>> = proofs_pos
            .iter()
            .flat_map(|(_, (_log_size, pos))| {
                pos.iter()
                    .map(move |p| domain.at(bit_reverse_index(*p, domain.log_size())))
            })
            .collect::<Vec<_>>();
        let evals = proofs_pos
            .into_iter()
            .flat_map(|(proof, _)| proof.evaluations)
            .collect::<Vec<_>>();
        let evals_nb = (((samples_nb * 20) / 2).next_power_of_two()) as usize;
        let _coeffs = interpolate_qm31(&pos[..evals_nb], &evals[..evals_nb]);
    }
}
