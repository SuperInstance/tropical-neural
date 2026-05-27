//! Tropical semiring and basic algebra.

use std::fmt;

/// A number in the tropical semiring (max-plus algebra).
///
/// Representation: finite values are real numbers, -∞ is the additive identity (zero element).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TropicalNumber {
    val: Option<f64>,
}

impl TropicalNumber {
    /// Tropical zero (-∞), the additive identity.
    pub fn neg_inf() -> Self {
        Self { val: None }
    }

    /// A finite tropical value.
    pub fn new(v: f64) -> Self {
        Self { val: Some(v) }
    }

    /// Tropical addition: max(a, b).
    pub fn tropical_add(self, other: Self) -> Self {
        match (self.val, other.val) {
            (None, _) => other,
            (_, None) => self,
            (Some(a), Some(b)) => Self::new(a.max(b)),
        }
    }

    /// Tropical multiplication: a + b.
    pub fn tropical_mul(self, other: Self) -> Self {
        match (self.val, other.val) {
            (None, _) | (_, None) => Self::neg_inf(),
            (Some(a), Some(b)) => Self::new(a + b),
        }
    }

    /// Get the underlying value, or -∞.
    pub fn value(&self) -> f64 {
        self.val.unwrap_or(f64::NEG_INFINITY)
    }

    /// Whether this is -∞.
    pub fn is_neg_inf(&self) -> bool {
        self.val.is_none()
    }
}

impl fmt::Display for TropicalNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.val {
            Some(v) => write!(f, "{}", v),
            None => write!(f, "-∞"),
        }
    }
}

/// The tropical semiring (R ∪ {-∞}, ⊕, ⊗).
///
/// Provides vector-space operations in tropical geometry.
pub struct TropicalSemiring;

impl TropicalSemiring {
    /// Tropical dot product: max of element-wise sums.
    /// ⊕ᵢ (aᵢ ⊗ bᵢ) = maxᵢ(aᵢ + bᵢ)
    pub fn dot(a: &[f64], b: &[f64]) -> TropicalNumber {
        assert_eq!(a.len(), b.len(), "vectors must have same length");
        if a.is_empty() {
            return TropicalNumber::neg_inf();
        }
        let mut best = f64::NEG_INFINITY;
        for i in 0..a.len() {
            let prod = a[i] + b[i];
            if prod > best {
                best = prod;
            }
        }
        TropicalNumber::new(best)
    }

    /// Tropical matrix-vector product.
    /// (A ⊗ x)ᵢ = ⊕ⱼ (Aᵢⱼ ⊗ xⱼ) = maxⱼ(Aᵢⱼ + xⱼ)
    pub fn mat_vec(mat: &[Vec<f64>], v: &[f64]) -> Vec<TropicalNumber> {
        mat.iter().map(|row| Self::dot(row, v)).collect()
    }

    /// Tropical polynomial evaluation: max over monomials.
    /// Given coefficients c and exponents e for each variable,
    /// evaluate: maxᵢ(cᵢ + Σⱼ eᵢⱼ * xⱼ)
    pub fn eval_tropical_poly(coeffs: &[f64], exponents: &[Vec<f64>], x: &[f64]) -> TropicalNumber {
        assert_eq!(coeffs.len(), exponents.len());
        let mut best = TropicalNumber::neg_inf();
        for i in 0..coeffs.len() {
            let mut sum = coeffs[i];
            for j in 0..x.len() {
                sum += exponents[i][j] * x[j];
            }
            best = best.tropical_add(TropicalNumber::new(sum));
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_add() {
        let a = TropicalNumber::new(3.0);
        let b = TropicalNumber::new(5.0);
        let c = a.tropical_add(b);
        assert_eq!(c.value(), 5.0);
    }

    #[test]
    fn test_tropical_add_identity() {
        let a = TropicalNumber::new(3.0);
        let zero = TropicalNumber::neg_inf();
        assert_eq!(a.tropical_add(zero).value(), 3.0);
        assert_eq!(zero.tropical_add(a).value(), 3.0);
    }

    #[test]
    fn test_tropical_mul() {
        let a = TropicalNumber::new(3.0);
        let b = TropicalNumber::new(5.0);
        let c = a.tropical_mul(b);
        assert_eq!(c.value(), 8.0);
    }

    #[test]
    fn test_tropical_mul_zero() {
        let a = TropicalNumber::new(3.0);
        let zero = TropicalNumber::neg_inf();
        assert!(a.tropical_mul(zero).is_neg_inf());
    }

    #[test]
    fn test_tropical_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // max(1+4, 2+5, 3+6) = max(5, 7, 9) = 9
        let result = TropicalSemiring::dot(&a, &b);
        assert_eq!(result.value(), 9.0);
    }

    #[test]
    fn test_tropical_mat_vec() {
        let mat = vec![vec![1.0, 2.0], vec![3.0, 1.0]];
        let v = vec![1.0, 1.0];
        // row 0: max(1+1, 2+1) = 3
        // row 1: max(3+1, 1+1) = 4
        let result = TropicalSemiring::mat_vec(&mat, &v);
        assert_eq!(result[0].value(), 3.0);
        assert_eq!(result[1].value(), 4.0);
    }
}
