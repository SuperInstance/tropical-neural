//! Tropical polynomials: max-norm piecewise-linear functions.

use crate::algebra::TropicalSemiring;

/// A tropical polynomial in n variables.
///
/// Represents: f(x) = maxᵢ(cᵢ + ⟨aᵢ, x⟩)
/// where cᵢ are coefficients and aᵢ are exponent vectors (monomials in tropical sense).
#[derive(Debug, Clone)]
pub struct TropicalPolynomial {
    /// Coefficients (the constant term in each monomial).
    pub coefficients: Vec<f64>,
    /// Exponent vectors: a[i] are the tropical powers for each variable in monomial i.
    pub exponents: Vec<Vec<f64>>,
    /// Number of variables.
    pub n_vars: usize,
}

impl TropicalPolynomial {
    /// Create a new tropical polynomial.
    pub fn new(coefficients: Vec<f64>, exponents: Vec<Vec<f64>>) -> Self {
        assert_eq!(coefficients.len(), exponents.len());
        let n_vars = exponents.first().map(|e| e.len()).unwrap_or(0);
        for e in &exponents {
            assert_eq!(e.len(), n_vars);
        }
        Self {
            coefficients,
            exponents,
            n_vars,
        }
    }

    /// A constant tropical polynomial in n_vars dimensions.
    pub fn constant_with_vars(c: f64, n_vars: usize) -> Self {
        Self {
            coefficients: vec![c],
            exponents: vec![vec![0.0; n_vars]],
            n_vars,
        }
    }

    /// A truly scalar constant (no variables).
    pub fn constant(c: f64) -> Self {
        Self {
            coefficients: vec![c],
            exponents: vec![vec![]],
            n_vars: 0,
        }
    }

    /// A single-variable tropical monomial: c + a*x.
    pub fn monomial(c: f64, power: f64) -> Self {
        Self {
            coefficients: vec![c],
            exponents: vec![vec![power]],
            n_vars: 1,
        }
    }

    /// Evaluate the polynomial at point x.
    pub fn eval(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.n_vars);
        TropicalSemiring::eval_tropical_poly(&self.coefficients, &self.exponents, x).value()
    }

    /// Tropical addition of two polynomials (max of corresponding evaluations).
    /// The result is the union of all monomials.
    pub fn tropical_add(&self, other: &Self) -> Self {
        assert_eq!(self.n_vars, other.n_vars);
        let mut coefficients = self.coefficients.clone();
        let mut exponents = self.exponents.clone();
        coefficients.extend_from_slice(&other.coefficients);
        exponents.extend_from_slice(&other.exponents);
        Self {
            coefficients,
            exponents,
            n_vars: self.n_vars,
        }
    }

    /// Tropical multiplication: convolution of monomials.
    /// (f ⊗ g)(x) = f(x) + g(x), so each monomial of f is added to each of g.
    pub fn tropical_mul(&self, other: &Self) -> Self {
        assert_eq!(self.n_vars, other.n_vars);
        let mut coefficients = Vec::new();
        let mut exponents = Vec::new();
        for i in 0..self.coefficients.len() {
            for j in 0..other.coefficients.len() {
                coefficients.push(self.coefficients[i] + other.coefficients[j]);
                let exp: Vec<f64> = self.exponents[i]
                    .iter()
                    .zip(&other.exponents[j])
                    .map(|(a, b)| a + b)
                    .collect();
                exponents.push(exp);
            }
        }
        Self {
            coefficients,
            exponents,
            n_vars: self.n_vars,
        }
    }

    /// Number of monomials.
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Whether this polynomial has no monomials.
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The domain regions (sectors) where each monomial is active.
    /// Returns a list of (monomial_index, vertex) pairs describing the Newton polytope.
    pub fn newton_polytope_vertices(&self) -> Vec<(usize, Vec<f64>)> {
        // A monomial is a vertex of the Newton polytope if it is NOT dominated
        // by any other monomial for all inputs. Simplified check:
        let mut vertices = Vec::new();
        for i in 0..self.coefficients.len() {
            let mut is_vertex = true;
            for j in 0..self.coefficients.len() {
                if i == j {
                    continue;
                }
                // Monomial j dominates i if c_j >= c_i AND a_j >= a_i (component-wise)
                let dominates = self.coefficients[j] >= self.coefficients[i]
                    && self.exponents[j]
                        .iter()
                        .zip(&self.exponents[i])
                        .all(|(ej, ei)| ej >= ei);
                if dominates {
                    is_vertex = false;
                    break;
                }
            }
            if is_vertex {
                vertices.push((i, self.exponents[i].clone()));
            }
        }
        vertices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_single_var() {
        // f(x) = max(1 + 2x, 3 + x) = max(1+2x, 3+x)
        let poly = TropicalPolynomial::new(vec![1.0, 3.0], vec![vec![2.0], vec![1.0]]);
        // At x=0: max(1, 3) = 3
        assert_eq!(poly.eval(&[0.0]), 3.0);
        // At x=2: max(5, 5) = 5
        assert_eq!(poly.eval(&[2.0]), 5.0);
        // At x=5: max(11, 8) = 11
        assert_eq!(poly.eval(&[5.0]), 11.0);
    }

    #[test]
    fn test_tropical_mul() {
        // f(x) = max(0, x), g(x) = max(0, -x)
        // f*g = max(0+0, 0+(-x), x+0, x+(-x)) = max(0, -x, x, 0) = max(x, -x, 0)
        let f = TropicalPolynomial::new(vec![0.0, 0.0], vec![vec![0.0], vec![1.0]]);
        let g = TropicalPolynomial::new(vec![0.0, 0.0], vec![vec![0.0], vec![-1.0]]);
        let h = f.tropical_mul(&g);
        assert_eq!(h.eval(&[1.0]), 1.0); // max(0, -1, 1, 0) = 1
        assert_eq!(h.eval(&[-1.0]), 1.0); // max(0, 1, -1, 0) = 1
        assert_eq!(h.eval(&[0.0]), 0.0); // max(0, 0, 0, 0) = 0
    }

    #[test]
    fn test_tropical_add() {
        let f = TropicalPolynomial::new(vec![1.0], vec![vec![1.0]]);
        let g = TropicalPolynomial::new(vec![2.0], vec![vec![0.5]]);
        let h = f.tropical_add(&g);
        assert_eq!(h.len(), 2);
        // max(1+x, 2+0.5x)
        assert_eq!(h.eval(&[0.0]), 2.0); // max(1, 2) = 2
        assert_eq!(h.eval(&[2.0]), 3.0); // max(3, 3) = 3
    }

    #[test]
    fn test_constant() {
        let c = TropicalPolynomial::constant(5.0);
        assert_eq!(c.eval(&[]), 5.0);
    }

    #[test]
    fn test_newton_polytope() {
        // f(x) = max(1+x, 3+0.5x, 0+2x) — monomial 0 dominated by monomial 2 when x > 1
        let poly =
            TropicalPolynomial::new(vec![1.0, 3.0, 0.0], vec![vec![1.0], vec![0.5], vec![2.0]]);
        let vertices = poly.newton_polytope_vertices();
        // Monomial 2 (c=0, e=2) is never dominated by 0 (c=1, e=1) since e2 > e0
        // Monomial 0 (c=1, e=1) is dominated by monomial 1 (c=3, e=0.5)? No: 3 > 1 but 0.5 < 1
        assert!(vertices.len() >= 2);
    }
}
