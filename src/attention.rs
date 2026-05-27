//! Tropical attention mechanism for neural algorithmic reasoning.

use crate::algebra::TropicalSemiring;

/// Tropical attention — an attention mechanism based on tropical (max-plus) algebra
/// instead of softmax. Preserves the polyhedral structure of combinatorial value functions.
///
/// Standard attention: softmax(QK^T / √d) V
/// Tropical attention: max-plus(QK^T) V where max-plus replaces the exp/sum/softmax
pub struct TropicalAttention {
    /// Temperature parameter for scaling.
    pub temperature: f64,
}

impl TropicalAttention {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Compute tropical attention scores.
    ///
    /// For queries Q (n×d) and keys K (m×d):
    /// score[i][j] = max_k(Q[i][k] + K[j][k]) / temperature
    pub fn scores(&self, queries: &[Vec<f64>], keys: &[Vec<f64>]) -> Vec<Vec<f64>> {
        queries
            .iter()
            .map(|q| {
                keys.iter()
                    .map(|k| TropicalSemiring::dot(q, k).value() / self.temperature)
                    .collect()
            })
            .collect()
    }

    /// Apply tropical attention: for each query, find the key with maximum tropical score
    /// and return its value scaled by the tropical score.
    ///
    /// Unlike softmax attention which creates a weighted average,
    /// tropical attention selects the maximum — preserving polyhedral structure.
    pub fn attend(
        &self,
        queries: &[Vec<f64>],
        keys: &[Vec<f64>],
        values: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        assert_eq!(keys.len(), values.len());
        let scores = self.scores(queries, keys);

        scores
            .iter()
            .map(|row| {
                // Find the argmax key
                let best_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                // Return the value scaled by the tropical score
                let best_score = row[best_idx];
                values[best_idx].iter().map(|v| v + best_score).collect()
            })
            .collect()
    }

    /// Soft tropical attention — a smooth approximation using LogSumExp
    /// instead of hard max. Blends tropical and classical attention.
    pub fn soft_attend(
        &self,
        queries: &[Vec<f64>],
        keys: &[Vec<f64>],
        values: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        assert_eq!(keys.len(), values.len());
        let d = values[0].len();
        let scores = self.scores(queries, keys);

        scores
            .iter()
            .map(|row| {
                // LogSumExp for smooth max
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = row.iter().map(|s| (s - max_val).exp()).sum();
                let log_sum_exp = max_val + sum_exp.ln();

                // Weighted combination using softmax-like weights from tropical scores
                let mut result = vec![0.0; d];
                for (j, &s) in row.iter().enumerate() {
                    let weight = (s - log_sum_exp).exp();
                    for k in 0..d {
                        result[k] += weight * values[j][k];
                    }
                }
                result
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_scores() {
        let q = vec![vec![1.0, 0.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let attn = TropicalAttention::new(1.0);
        let scores = attn.scores(&q, &k);
        // max(1+1, 0+0) = 2, max(1+0, 0+1) = 1
        assert_eq!(scores[0][0], 2.0);
        assert_eq!(scores[0][1], 1.0);
    }

    #[test]
    fn test_tropical_attend() {
        let q = vec![vec![1.0, 0.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![10.0], vec![20.0]];
        let attn = TropicalAttention::new(1.0);
        let result = attn.attend(&q, &k, &v);
        // Best match is key 0 (score 2.0), so value = 10.0 + 2.0 = 12.0
        assert_eq!(result[0][0], 12.0);
    }

    #[test]
    fn test_soft_attend() {
        let q = vec![vec![1.0, 0.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![10.0], vec![20.0]];
        let attn = TropicalAttention::new(1.0);
        let result = attn.soft_attend(&q, &k, &v);
        // Should be a weighted combination, closer to v[0] since it has higher score
        assert!(result[0][0] > 10.0);
    }

    #[test]
    fn test_temperature_scaling() {
        let q = vec![vec![1.0]];
        let k = vec![vec![2.0]];
        let attn = TropicalAttention::new(2.0);
        let scores = attn.scores(&q, &k);
        // (1+2)/2 = 1.5
        assert_eq!(scores[0][0], 1.5);
    }
}
