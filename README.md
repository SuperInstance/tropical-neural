# tropical-neural

Tropical geometry for neural networks — max-plus semiring operations, tropical polynomials with Newton polytopes, tropical rational maps (ReLU network representation), and tropical attention mechanisms.

## Usage

```rust
use tropical_neural::{TropicalSemiring, TropicalPolynomial};

let ts = TropicalSemiring::new();

// Tropical addition: max(a, b)
let sum = ts.add(3.0, 5.0); // 5.0

// Tropical multiplication: a + b
let prod = ts.mul(3.0, 5.0); // 8.0

// Tropical polynomial evaluation
let poly = TropicalPolynomial::from_coefficients(vec![1.0, 2.0, 3.0]);
let val = poly.evaluate(2.0);
```

## Features

- **Tropical semiring** (max-plus algebra): `a ⊕ b = max(a,b)`, `a ⊗ b = a + b`
- **Tropical polynomials** with Newton polytope computation
- **Tropical rational maps** representing ReLU neural networks
- **Tropical attention** mechanism for transformer-like architectures
- **Tropical division** for polynomial decomposition

## Tests

22 tests, all passing. `cargo test` to run.

## License

MIT
