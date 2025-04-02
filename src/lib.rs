use ark_ff::fields::Field;
use ark_ff_optimized::fp31::Fp;
use ark_poly::DenseUVPolynomial;
use ark_poly::polynomial::univariate::DensePolynomial;
use ark_poly::univariate::DenseOrSparsePolynomial;
use num_traits::{One, Zero};
use rayon::prelude::*;
pub mod complex_interpolation;

pub fn fast_interpolation(u: &[Fp], v: &[Fp]) -> DensePolynomial<Fp> {
    assert_eq!(u.len(), v.len());
    assert!(u.len().is_power_of_two());

    let tree = build_subproduct_tree(u);

    let k = u.len().ilog2() as usize;
    let m = &tree[k][0]; // m(x) = ∏ (x-u_i)

    let m_deriv = poly_derivative(m);
    let m_deriv_vals = eval_tree(&m_deriv, &tree, u, 0);

    let c: Vec<Fp> = v
        .par_iter()
        .zip(m_deriv_vals.par_iter())
        .map(|(&vi, &m_prime_ui)| vi * m_prime_ui.inverse().expect("Nonzero derivative"))
        .collect();

    linear_combination(u, &c, &tree, k, 0)
}

pub fn build_subproduct_tree(u: &[Fp]) -> Vec<Vec<DensePolynomial<Fp>>> {
    // Level 0: polynomials of the form (x - u[i])
    let k = u.len().ilog2() as usize;
    let mut tree: Vec<Vec<DensePolynomial<Fp>>> = Vec::with_capacity(k);
    tree.push(
        u.iter()
            .map(|&ui| DensePolynomial::from_coefficients_vec(vec![-ui, Fp::one()]))
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
    f: &DensePolynomial<Fp>,
    tree: &Vec<Vec<DensePolynomial<Fp>>>,
    points: &[Fp],
    index: usize,
) -> Vec<Fp> {
    if points.len() == 1 {
        return vec![f.coeffs[0]];
    }
    let k = points.len().ilog2() as usize;

    let left_poly = &tree[k - 1][2 * index];
    let right_poly = &tree[k - 1][2 * index + 1];
    // Compute remainders of f modulo the children.
    let (_q_left, r_left) = DenseOrSparsePolynomial::from(f)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(left_poly))
        .expect("Division failed");
    let (_q_right, r_right) = DenseOrSparsePolynomial::from(f)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(right_poly))
        .expect("Division failed");
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
    u: &[Fp],
    c: &[Fp],
    subproduct: &Vec<Vec<DensePolynomial<Fp>>>,
    level: usize,
    index: usize,
) -> DensePolynomial<Fp> {
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
    &term1 + &term2
}

pub fn poly_derivative(poly: &DensePolynomial<Fp>) -> DensePolynomial<Fp> {
    if poly.coeffs.len() <= 1 {
        return DensePolynomial::from_coefficients_vec(vec![Fp::zero()]);
    }
    let deriv_coeffs: Vec<Fp> = poly
        .coeffs
        .par_iter()
        .enumerate()
        .skip(1)
        .map(|(i, coeff)| Fp::from(i as u64) * *coeff)
        .collect();
    DensePolynomial::from_coefficients_vec(deriv_coeffs)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ark_poly::Polynomial;
    use bitvec::{field::BitField, order::Lsb0, vec::BitVec};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use super::*;

    #[test]
    pub fn test_regular_interpolation() {
        let mut bytes = include_bytes!("../blob").to_vec();
        // bytes.extend_from_slice(include_bytes!("../blob"));
        // let bytes2 = bytes.clone();
        // bytes.extend_from_slice(&bytes2);
        // let bytes3 = bytes.clone();
        // bytes.extend_from_slice(&bytes3);

        let bitvec = BitVec::<u8, Lsb0>::from_slice(&bytes);
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
        println!("poly degree: {:?}", poly.degree());

        println!("computing xs and ys");
        let xs_ys = (0..poly.degree() + 1)
            .into_par_iter()
            .map(|i| {
                let x = Fp::from(i as u32);
                let y = poly.evaluate(&x);
                (x, y)
            })
            .collect::<Vec<_>>();

        let xs = xs_ys.iter().map(|x| x.0).collect::<Vec<_>>();
        let ys = xs_ys.iter().map(|y| y.1).collect::<Vec<_>>();
        // save xs and ys to file
        // let mut xs_file = File::create("xs.json").unwrap();
        // serde_json::to_writer(&mut xs_file, &xs.iter().map(|x| x.0).collect::<Vec<_>>()).unwrap();
        // let mut ys_file = File::create("ys.json").unwrap();
        // serde_json::to_writer(&mut ys_file, &ys.iter().map(|y| y.0).collect::<Vec<_>>()).unwrap();
        // let xs = serde_json::from_slice::<Vec<u32>>(include_bytes!("../xs.json"))
        //     .unwrap()
        //     .into_par_iter()
        //     .map(Fp)
        //     .collect::<Vec<_>>();
        // let ys = serde_json::from_slice::<Vec<u32>>(include_bytes!("../ys.json"))
        //     .unwrap()
        //     .into_par_iter()
        //     .map(Fp)
        //     .collect::<Vec<_>>();
        println!("Done");
        let start = Instant::now();
        let f = fast_interpolation(&xs, &ys);
        let duration = start.elapsed();
        println!("fast interpolation time: {:?}", duration);
        assert_eq!(f.degree(), poly.degree());
        assert_eq!(f, poly);
    }

    #[test]
    pub fn test_derivative() {
        let poly =
            DensePolynomial::from_coefficients_vec(vec![Fp::from(1), Fp::from(1), Fp::from(1)]);
        let deriv = poly_derivative(&poly);
        assert_eq!(deriv.coeffs, vec![Fp::from(1), Fp::from(2)]);
    }
}
