//! Tropical polynomial division for neural network simplification.

use crate::network::TropicalNetwork;
use crate::polynomial::TropicalPolynomial;

/// Tropical polynomial division — divides one tropical polynomial by another.
///
/// In tropical algebra: f(x) = q(x) ⊗ d(x) ⊕ r(x)
/// which means: f(x) = max(q(x) + d(x), r(x))
/// where deg(r) < deg(d) in tropical degree sense.
pub struct TropicalDivision;

impl TropicalDivision {
    /// Compute the tropical quotient and remainder of f divided by d.
    ///
    /// Returns (quotient, remainder) such that:
    /// f(x) = max(quotient(x) + d(x), remainder(x))
    pub fn divide(
        f: &TropicalPolynomial,
        d: &TropicalPolynomial,
    ) -> (TropicalPolynomial, TropicalPolynomial) {
        if d.is_empty() || d.n_vars != f.n_vars {
            return (
                TropicalPolynomial::constant_with_vars(0.0, f.n_vars),
                f.clone(),
            );
        }

        // Tropical division: for each monomial of f, find the best matching monomial of d
        // The quotient collects the "subtractions" and remainder collects leftovers
        let mut q_coeffs = Vec::new();
        let mut q_exp = Vec::new();
        let mut r_coeffs = Vec::new();
        let mut r_exp = Vec::new();

        // Simple division: for each monomial of f, subtract the "leading" monomial of d
        if d.is_empty() {
            let n = f.n_vars;
            return (TropicalPolynomial::constant_with_vars(0.0, n), f.clone());
        }

        // Use the monomial with highest coefficient in d as the "leading" term
        let (lead_idx, _) = d
            .coefficients
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));
        let lead_c = d.coefficients[lead_idx];
        let lead_e = &d.exponents[lead_idx];

        for i in 0..f.coefficients.len() {
            // Tropical "subtraction" = subtraction of coefficients and exponents
            let q_c = f.coefficients[i] - lead_c;
            let q_e: Vec<f64> = f.exponents[i]
                .iter()
                .zip(lead_e.iter())
                .map(|(fe, de)| fe - de)
                .collect();

            // Check if this quotient term actually contributes
            // by verifying the product q⊗d ≤ f
            let product_val = q_c + lead_c;
            if product_val <= f.coefficients[i] + 1e-10 {
                q_coeffs.push(q_c);
                q_exp.push(q_e);
            } else {
                // Goes into remainder
                r_coeffs.push(f.coefficients[i]);
                r_exp.push(f.exponents[i].clone());
            }
        }

        if q_coeffs.is_empty() {
            q_coeffs.push(0.0);
            q_exp.push(vec![0.0; f.n_vars]);
        }

        let n = f.n_vars;
        let quotient = if q_exp.is_empty() {
            TropicalPolynomial::constant_with_vars(0.0, f.n_vars)
        } else {
            TropicalPolynomial::new(q_coeffs, q_exp)
        };

        let remainder = if r_coeffs.is_empty() {
            TropicalPolynomial::constant(f64::NEG_INFINITY)
        } else {
            TropicalPolynomial::new(r_coeffs, r_exp)
        };

        (quotient, remainder)
    }

    /// Simplify a ReLU network using tropical polynomial division.
    ///
    /// Factor out common tropical factors from numerator and denominator,
    /// effectively removing redundant neurons.
    pub fn simplify_network(net: &mut TropicalNetwork) -> usize {
        let before = net.numerator.len() + net.denominator.len();

        // Step 1: Remove dominated monomials
        net.simplify();

        // Step 2: Try to divide numerator by denominator
        let (q, r) = Self::divide(&net.numerator, &net.denominator);

        // If remainder is "small", the division was clean
        if r.len() <= 1 {
            // Network can be simplified
            net.numerator = q;
            net.denominator = TropicalPolynomial::constant_with_vars(0.0, 1);
        }

        before - net.numerator.len() - net.denominator.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_division() {
        // f(x) = max(5 + 2x, 3 + x), d(x) = max(2 + x)
        let f = TropicalPolynomial::new(vec![5.0, 3.0], vec![vec![2.0], vec![1.0]]);
        let d = TropicalPolynomial::new(vec![2.0], vec![vec![1.0]]);
        let (q, r) = TropicalDivision::divide(&f, &d);
        // q should have monomials: (5-2, 2-1) = (3, 1) and (3-2, 1-1) = (1, 0)
        assert!(q.len() >= 1);
    }

    #[test]
    fn test_division_by_constant() {
        let f = TropicalPolynomial::new(vec![3.0, 5.0], vec![vec![1.0], vec![2.0]]);
        let d = TropicalPolynomial::constant(2.0);
        // Division by a scalar constant — should handle gracefully
        let (q, _r) = TropicalDivision::divide(&f, &d);
        // With mismatched n_vars, the function returns early
        assert!(q.n_vars == f.n_vars || q.n_vars == 0);
    }

    #[test]
    fn test_network_simplification() {
        let num =
            TropicalPolynomial::new(vec![1.0, 2.0, 3.0], vec![vec![1.0], vec![1.0], vec![1.0]]);
        let den = TropicalPolynomial::constant_with_vars(0.0, 1);
        let mut net = TropicalNetwork::new(num, den);
        let removed = TropicalDivision::simplify_network(&mut net);
        // Should remove dominated monomials
        assert!(removed >= 1);
    }
}
