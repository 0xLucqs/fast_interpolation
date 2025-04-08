use lambdaworks_math::{
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field, traits::IsField},
    polynomial::Polynomial,
};
use num_traits::{One, Zero};
use rayon::prelude::*;
pub mod complex_interpolation;

pub fn fast_interpolation(
    u: &[FieldElement<Mersenne31Field>],
    v: &[FieldElement<Mersenne31Field>],
) -> Polynomial<FieldElement<Mersenne31Field>> {
    assert_eq!(u.len(), v.len());
    assert!(u.len().is_power_of_two());

    let tree = build_subproduct_tree(u);

    let k = u.len().ilog2() as usize;
    let m = &tree[k][0]; // m(x) = ∏ (x-u_i)

    let m_deriv = m.differentiate();
    let m_deriv_vals = eval_tree(&m_deriv, &tree, u, 0);

    let c: Vec<FieldElement<Mersenne31Field>> = v
        .par_iter()
        .zip(m_deriv_vals.par_iter())
        .map(|(&vi, &m_prime_ui)| vi * m_prime_ui.inv().expect("Nonzero derivative"))
        .collect();

    linear_combination(u, &c, &tree, k, 0)
}

pub fn build_subproduct_tree<F: IsField>(
    u: &[FieldElement<F>],
) -> Vec<Vec<Polynomial<FieldElement<F>>>> {
    // Level 0: polynomials of the form (x - u[i])
    let k = u.len().ilog2() as usize;
    let mut tree: Vec<Vec<Polynomial<FieldElement<F>>>> = Vec::with_capacity(k);
    tree.push(
        u.iter()
            .map(|ui| Polynomial::new(&[-ui, FieldElement::<F>::one()]))
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

pub fn eval_tree<F: IsField>(
    f: &Polynomial<FieldElement<F>>,
    tree: &Vec<Vec<Polynomial<FieldElement<F>>>>,
    points: &[FieldElement<F>],
    index: usize,
) -> Vec<FieldElement<F>> {
    if points.len() == 1 {
        return vec![
            f.coefficients
                .last()
                .unwrap_or(&FieldElement::<F>::zero())
                .clone(),
        ];
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
    let left_vals = eval_tree(&r_left, tree, left_points, 2 * index);
    let right_vals = eval_tree(&r_right, tree, right_points, 2 * index + 1);
    let mut result = left_vals;
    result.extend(right_vals);
    result
}

pub fn linear_combination<F: IsField>(
    u: &[FieldElement<F>],
    c: &[FieldElement<F>],
    subproduct: &Vec<Vec<Polynomial<FieldElement<F>>>>,
    level: usize,
    index: usize,
) -> Polynomial<FieldElement<F>> {
    if c.len() == 1 {
        // Base case: return the constant polynomial equal to c[0].
        return Polynomial::new(c);
    }
    let child_level = level - 1;
    let mid = u.len() / 2;

    let r0 = linear_combination(&u[..mid], &c[..mid], subproduct, child_level, 2 * index);
    let r1 = linear_combination(&u[mid..], &c[mid..], subproduct, child_level, 2 * index + 1);

    let left_poly = &subproduct[child_level][2 * index]; // corresponds to left half
    let right_poly = &subproduct[child_level][2 * index + 1]; // corresponds to right half
    let term1 = right_poly.mul_with_ref(&r0);
    let term2 = left_poly.mul_with_ref(&r1);
    &term1 + &term2
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use bitvec::{field::BitField, order::Lsb0, vec::BitVec};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use super::*;

    #[test]
    pub fn test_regular_interpolation() {
        let bytes = include_bytes!("../blob")[..(2048. * 3.75) as usize].to_vec();
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
                FieldElement::<Mersenne31Field>::new(value)
            })
            .collect::<Vec<_>>();
        let length = coeffs.len().next_power_of_two();

        coeffs.resize(length, FieldElement::<Mersenne31Field>::one());

        let poly = Polynomial::new(&coeffs);
        println!("poly degree: {:?}", poly.degree());

        println!("computing xs and ys");
        let xs_ys = (0..poly.degree() + 1)
            .into_par_iter()
            .map(|i| {
                let x = FieldElement::<Mersenne31Field>::new(i as u32);
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
        //     .map(FieldElement<Mersenne31Field>)
        //     .collect::<Vec<_>>();
        // let ys = serde_json::from_slice::<Vec<u32>>(include_bytes!("../ys.json"))
        //     .unwrap()
        //     .into_par_iter()
        //     .map(FieldElement<Mersenne31Field>)
        //     .collect::<Vec<_>>();
        println!("Done");
        let start = Instant::now();
        let f = fast_interpolation(&xs, &ys);
        let duration = start.elapsed();
        println!("fast interpolation time: {:?}", duration);
        assert_eq!(f.degree(), poly.degree());
        assert_eq!(f, poly);
    }
}
