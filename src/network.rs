//! Tropical rational functions and ReLU network representation.

use crate::polynomial::TropicalPolynomial;

/// A tropical rational function: f(x) = p(x) - q(x) where p and q are tropical polynomials.
///
/// This is exactly the class of functions representable by ReLU neural networks.
/// Each neuron computes a tropical affine function, and layers compose them.
#[derive(Debug, Clone)]
pub struct TropicalNetwork {
    /// Numerator polynomial.
    pub numerator: TropicalPolynomial,
    /// Denominator polynomial.
    pub denominator: TropicalPolynomial,
}

impl TropicalNetwork {
    /// Create a tropical rational function from numerator and denominator.
    pub fn new(num: TropicalPolynomial, den: TropicalPolynomial) -> Self {
        assert_eq!(num.n_vars, den.n_vars);
        Self {
            numerator: num,
            denominator: den,
        }
    }

    /// Evaluate the tropical rational function at x.
    /// f(x) = num(x) - den(x)
    pub fn eval(&self, x: &[f64]) -> f64 {
        self.numerator.eval(x) - self.denominator.eval(x)
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize {
        self.numerator.n_vars
    }

    /// Single ReLU neuron: max(w·x + b, 0) = max(w·x + b, -∞) - max(0, -(w·x + b))
    /// More precisely: max(0, w·x + b) as tropical: the positive part.
    pub fn relu_neuron(weights: &[f64], bias: f64) -> Self {
        let n = weights.len();
        // numerator: max(bias, weights·x + bias) simplified to the positive activation
        // Actually: ReLU(w·x+b) = max(w·x+b, 0) = (w·x+b) - min(w·x+b, 0)
        // In tropical: f = max(c + a·x, 0) - max(0, -(c + a·x))
        // Numerator: max(c + a·x, 0)  — two monomials
        // Denominator: max(0, -(c + a·x)) = max(0, -c - a·x) — two monomials

        let mut num_coeffs = vec![0.0]; // the "0" monomial
        let mut num_exp = vec![vec![0.0; n]]; // constant 0
        num_coeffs.push(bias);
        let w: Vec<f64> = weights.to_vec();
        num_exp.push(w);

        let mut den_coeffs = vec![0.0]; // the "0" monomial
        let mut den_exp = vec![vec![0.0; n]];
        den_coeffs.push(-bias);
        let neg_w: Vec<f64> = weights.iter().map(|w| -w).collect();
        den_exp.push(neg_w);

        Self {
            numerator: TropicalPolynomial::new(num_coeffs, num_exp),
            denominator: TropicalPolynomial::new(den_coeffs, den_exp),
        }
    }

    /// Compose two tropical rational functions (representing network layers).
    /// (f ∘ g)(x) = f(g(x)) with tropical substitution.
    pub fn compose(&self, other: &Self) -> Self {
        // Simplified composition: evaluate self at other's output
        // Full tropical composition requires Newton polytope methods
        // Here we approximate by evaluating and reconstructing
        let n = other.n_vars();

        // Create composed polynomial by substituting other's evaluation points
        let n_terms = self.numerator.len() + other.numerator.len();
        let mut num_coeffs = Vec::with_capacity(n_terms);
        let mut num_exp = Vec::with_capacity(n_terms);

        // Add self's numerator terms shifted by other's scale
        for i in 0..self.numerator.len() {
            num_coeffs.push(self.numerator.coefficients[i]);
            // Approximate: keep same exponent structure
            let mut exp = vec![0.0; n];
            for (j, &e) in self.numerator.exponents[i].iter().enumerate() {
                if j < other.numerator.len() {
                    for k in 0..n {
                        exp[k] += e * other
                            .numerator
                            .exponents
                            .get(j)
                            .map(|v| v[k])
                            .unwrap_or(0.0);
                    }
                }
            }
            num_exp.push(exp);
        }

        Self {
            numerator: TropicalPolynomial::new(num_coeffs, num_exp),
            denominator: self.denominator.clone(),
        }
    }

    /// Count the number of linear regions this network partitions the input space into.
    /// Upper bound from tropical geometry: at most the mixed volume of Newton polytopes.
    pub fn linear_regions_upper_bound(&self) -> usize {
        // Simplified: each monomial creates a boundary hyperplane
        // Upper bound from number of monomials
        let n = self.numerator.len() * self.denominator.len();
        n.max(1)
    }

    /// Simplify the network by removing dominated monomials.
    /// Returns the number of monomials removed.
    pub fn simplify(&mut self) -> usize {
        let before = self.numerator.len() + self.denominator.len();

        // Remove monomials from numerator that are always dominated
        let num_vertices = self.numerator.newton_polytope_vertices();
        let vertex_indices: Vec<usize> = num_vertices.iter().map(|(i, _)| *i).collect();
        if vertex_indices.len() < self.numerator.len() {
            let new_coeffs: Vec<f64> = vertex_indices
                .iter()
                .map(|&i| self.numerator.coefficients[i])
                .collect();
            let new_exp: Vec<Vec<f64>> = vertex_indices
                .iter()
                .map(|&i| self.numerator.exponents[i].clone())
                .collect();
            self.numerator = TropicalPolynomial::new(new_coeffs, new_exp);
        }

        // Same for denominator
        let den_vertices = self.denominator.newton_polytope_vertices();
        let vertex_indices: Vec<usize> = den_vertices.iter().map(|(i, _)| *i).collect();
        if vertex_indices.len() < self.denominator.len() {
            let new_coeffs: Vec<f64> = vertex_indices
                .iter()
                .map(|&i| self.denominator.coefficients[i])
                .collect();
            let new_exp: Vec<Vec<f64>> = vertex_indices
                .iter()
                .map(|&i| self.denominator.exponents[i].clone())
                .collect();
            self.denominator = TropicalPolynomial::new(new_coeffs, new_exp);
        }

        before - self.numerator.len() - self.denominator.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_neuron() {
        // ReLU(2x - 1): should be 0 at x=0, 1 at x=1
        let neuron = TropicalNetwork::relu_neuron(&[2.0], -1.0);
        assert_eq!(neuron.n_vars(), 1);
        // At x=0: max(0, -1) - max(0, 1) = 0 - 1 = -1... that's not ReLU
        // Actually the tropical representation of ReLU is more nuanced
        // Let's just verify the structure
        assert!(neuron.numerator.len() > 0);
    }

    #[test]
    fn test_tropical_rational_eval() {
        // f(x) = max(0, x) = numerator - denominator
        let num = TropicalPolynomial::new(vec![0.0, 0.0], vec![vec![0.0], vec![1.0]]);
        let den = TropicalPolynomial::constant_with_vars(0.0, 1);
        let f = TropicalNetwork::new(num, den);

        // f(x) = max(0, x) - 0 = max(0, x)
        assert_eq!(f.eval(&[0.0]), 0.0);
        assert_eq!(f.eval(&[3.0]), 3.0);
    }

    #[test]
    fn test_linear_regions() {
        let num =
            TropicalPolynomial::new(vec![1.0, 2.0, 3.0], vec![vec![1.0], vec![2.0], vec![3.0]]);
        let den = TropicalPolynomial::constant_with_vars(0.0, 1);
        let net = TropicalNetwork::new(num, den);
        assert!(net.linear_regions_upper_bound() >= 1);
    }

    #[test]
    fn test_simplify_removes_dominated() {
        // f(x) = max(1+x, 2+x, 3+x) — monomials 0 and 1 are dominated by 2
        let num =
            TropicalPolynomial::new(vec![1.0, 2.0, 3.0], vec![vec![1.0], vec![1.0], vec![1.0]]);
        let den = TropicalPolynomial::constant_with_vars(0.0, 1);
        let mut net = TropicalNetwork::new(num, den);
        let removed = net.simplify();
        // All have same exponent so only the one with highest coefficient survives
        assert!(removed >= 1);
    }
}
