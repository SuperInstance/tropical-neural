//! # Tropical Neural
//!
//! Tropical geometry for neural network analysis. Represents ReLU networks as tropical
//! rational maps, implements tropical polynomial division for network simplification,
//! and provides tropical attention mechanisms.
//!
//! # Key Concepts
//!
//! - **Tropical semiring**: (R ∪ {-∞}, ⊕, ⊗) where a ⊕ b = max(a,b) and a ⊗ b = a + b
//! - **Tropical polynomials**: max-norm piecewise-linear functions
//! - **Tropical rational maps**: difference of two tropical polynomials — exactly the
//!   class of functions representable by ReLU networks
//! - **Tropical polynomial division**: provably simplifies networks by removing redundant neurons

mod algebra;
mod attention;
mod division;
mod network;
mod polynomial;

pub use algebra::{TropicalNumber, TropicalSemiring};
pub use attention::TropicalAttention;
pub use division::TropicalDivision;
pub use network::TropicalNetwork;
pub use polynomial::TropicalPolynomial;
