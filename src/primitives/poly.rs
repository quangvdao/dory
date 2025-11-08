//! Polynomial trait for multilinear polynomials

use crate::error::DoryError;
use crate::setup::ProverSetup;

use super::arithmetic::{DoryRoutines, Field, Group, PairingCurve};

/// Dory commitment containing both tier-1 and tier-2 commitments
///
/// The Dory commitment scheme uses a two-tier structure:
/// - Tier-1: Row commitments in G1 (one per row of the coefficient matrix)
/// - Tier-2: Final commitment in GT (combining row commitments via pairing)
#[derive(Debug, Clone)]
pub struct DoryCommitment<G1, GT> {
    /// Tier-2 commitment: Final commitment in GT group
    pub tier_2: GT,
    /// Tier-1 commitments: Row commitments in G1 group
    pub tier_1: Vec<G1>,
}

impl<G1, GT> DoryCommitment<G1, GT> {
    /// Create a new Dory commitment from tier-1 and tier-2 components
    pub fn new(tier_2: GT, tier_1: Vec<G1>) -> Self {
        Self { tier_2, tier_1 }
    }
}

/// Compute a standardized (nu, sigma) split for a given total variable count.
///
/// Standardization rules:
/// - sigma <= nu
/// - nu + sigma = total_vars
/// - |nu - sigma| <= 1 (as close as possible)
///
/// This chooses nu = ceil(total_vars/2) and sigma = floor(total_vars/2).
/// The intent is to make the matrix as square as possible while ensuring
/// the row dimension (2^nu) dominates and is the one covered by the
/// reduce-and-fold rounds.
pub fn standardize_nu_sigma(total_vars: usize) -> (usize, usize) {
    let sigma = total_vars / 2; // floor
    let nu = total_vars - sigma; // ceil
    debug_assert!(nu >= sigma);
    debug_assert_eq!(nu + sigma, total_vars);
    debug_assert!(nu.saturating_sub(sigma) <= 1);
    (nu, sigma)
}

/// Trait for multilinear Lagrange polynomial operations
pub trait MultilinearLagrange<F: Field>: Polynomial<F> {
    /// Compute multilinear Lagrange basis evaluations at a point
    ///
    /// For variables (r₀, r₁, ..., r_{n-1}), computes all 2^n basis polynomial evaluations.
    /// The i-th basis polynomial evaluates to 1 at the i-th hypercube vertex and 0 elsewhere.
    fn lagrange_basis(&self, output: &mut [F], point: &[F]) {
        multilinear_lagrange_basis(output, point)
    }

    /// Compute vector-matrix product: v = L^T * M
    ///
    /// Treats coefficients as a 2^nu × 2^sigma matrix.
    /// For each column j: v[j] = Σ_i left_vec[i] * coefficients[i][j]
    fn vector_matrix_product(&self, left_vec: &[F], nu: usize, sigma: usize) -> Vec<F> {
        compute_v_vec(self.coefficients(), left_vec, nu, sigma)
    }

    /// Compute left and right vectors from evaluation point
    ///
    /// Given a point arranged for matrix evaluation, computes L and R such that:
    /// polynomial_evaluation(point) = L^T × M × R
    fn compute_evaluation_vectors(&self, point: &[F], nu: usize, sigma: usize) -> (Vec<F>, Vec<F>) {
        compute_left_right_vectors(point, nu, sigma)
    }
}

/// Trait for multilinear polynomials
///
/// Represents a polynomial in evaluation form (coefficients at hypercube points).
pub trait Polynomial<F: Field> {
    /// Number of variables
    fn num_vars(&self) -> usize;

    /// Total number of coefficients (2^num_vars)
    fn len(&self) -> usize {
        1 << self.num_vars()
    }

    /// Check if polynomial is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluate polynomial at a point
    ///
    /// # Parameters
    /// - `point`: Evaluation point (length must equal num_vars)
    ///
    /// # Returns
    /// Polynomial evaluation result
    fn evaluate(&self, point: &[F]) -> F;

    /// Get reference to coefficients
    fn coefficients(&self) -> &[F];

    /// Commit to polynomial using Dory's 2-tier (AFGHO) homomorphic commitment
    ///
    /// The polynomial coefficients are arranged as a 2D matrix with 2^nu rows and 2^sigma columns.
    ///
    /// # Tier 1 (Row Commitments)
    /// For each row i: `row_commit[i] = MSM(g1_generators[0..2^sigma], row_coefficients[i])`
    ///
    /// # Tier 2 (Final Commitment)
    /// `commitment = Σ e(row_commit[i], g2_generators[i])` for i in 0..2^nu
    ///
    /// # Parameters
    /// - `nu`: Log₂ of number of rows
    /// - `sigma`: Log₂ of number of columns
    /// - `setup`: Prover setup containing generators
    ///
    /// # Returns
    /// `(commitment, row_commitments)` where:
    /// - `commitment`: Final commitment in GT
    /// - `row_commitments`: Intermediate row commitments in G1 (used in opening proof)
    ///
    /// # Errors
    /// Returns error if coefficient length doesn't match 2^(nu + sigma) or if setup is insufficient.
    fn commit<E, M1>(
        &self,
        nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>), DoryError>
    where
        E: PairingCurve,
        M1: DoryRoutines<E::G1>,
        E::G1: Group<Scalar = F>;
}

/// Compute multilinear Lagrange basis evaluations at a point
///
/// For variables (r₀, r₁, ..., r_{n-1}), computes all 2^n basis polynomial evaluations.
/// The i-th basis polynomial evaluates to 1 at the i-th hypercube vertex and 0 elsewhere.
///
/// Uses an iterative doubling approach:
/// - Start with [1-r₀, r₀]
/// - For each variable rᵢ, split each value v into [v*(1-rᵢ), v*rᵢ]
pub(crate) fn multilinear_lagrange_basis<F: Field>(output: &mut [F], point: &[F]) {
    assert!(
        output.len() <= (1 << point.len()),
        "Output length must be at most 2^point.len()"
    );

    if point.is_empty() || output.is_empty() {
        output.fill(F::one());
        return;
    }

    // Initialize for first variable: [1-r₀, r₀]
    let one_minus_p0 = F::one() - point[0];
    output[0] = one_minus_p0;
    if output.len() > 1 {
        output[1] = point[0];
    }

    // For each subsequent variable, double the active portion
    for (level, p) in point[1..].iter().enumerate() {
        let mid = 1 << (level + 1);
        let one_minus_p = F::one() - p;

        if mid >= output.len() {
            // No split possible, just multiply all by (1-p)
            for val in output.iter_mut() {
                *val = val.mul(&one_minus_p);
            }
        } else {
            // Split: left *= (1-p), right = left * p
            let (left, right) = output.split_at_mut(mid);
            let k = left.len().min(right.len());

            for (l, r) in left[..k].iter_mut().zip(right[..k].iter_mut()) {
                let l_val = *l;
                *r = l_val.mul(p);
                *l = l_val.mul(&one_minus_p);
            }

            // Handle remaining left elements if any
            for l in left[k..].iter_mut() {
                *l = l.mul(&one_minus_p);
            }
        }
    }
}

/// Compute left and right vectors from evaluation point
///
/// Given a point arranged for matrix evaluation, computes L and R such that:
/// polynomial_evaluation(point) = L^T × M × R
///
/// Standardized split: first `nu` coordinates produce the left vector L (size 2^nu),
/// next `sigma` coordinates produce the right vector R (size 2^sigma).
pub fn compute_left_right_vectors<F: Field>(
    point: &[F],
    nu: usize,
    sigma: usize,
) -> (Vec<F>, Vec<F>) {
    let mut left_vec = vec![F::zero(); 1 << nu];
    let mut right_vec = vec![F::zero(); 1 << sigma];
    // Left vector from first nu coordinates
    let left_coords = &point[..nu];
    multilinear_lagrange_basis(&mut left_vec, left_coords);
    // Right vector from next sigma coordinates
    let right_coords = &point[nu..nu + sigma];
    multilinear_lagrange_basis(&mut right_vec, right_coords);
    (left_vec, right_vec)
}

/// Compute vector-matrix product: v = L^T * M
///
/// Treats coefficients as a 2^nu × 2^sigma matrix.
/// For each column j: v[j] = Σ_i left_vec[i] * coefficients[i][j]
fn compute_v_vec<F: Field>(coefficients: &[F], left_vec: &[F], nu: usize, sigma: usize) -> Vec<F> {
    let num_cols = 1 << sigma;
    let num_rows = 1 << nu;
    let mut v_vec = vec![F::zero(); num_cols];

    for (j, v) in v_vec.iter_mut().enumerate() {
        let mut sum = F::zero();
        for (i, left_val) in left_vec.iter().enumerate().take(num_rows) {
            let coeff_idx = i * num_cols + j;
            if coeff_idx < coefficients.len() {
                sum = sum + left_val.mul(&coefficients[coeff_idx]);
            }
        }
        *v = sum;
    }

    v_vec
}

/// Compute matrix-vector product by columns: u = M × R
///
/// Treats coefficients as a 2^nu × 2^sigma matrix.
/// For each row i: u[i] = Σ_j coefficients[i][j] * right_vec[j]
pub fn matrix_vector_product_rows<F: Field>(
    coefficients: &[F],
    right_vec: &[F],
    nu: usize,
    sigma: usize,
) -> Vec<F> {
    let num_cols = 1 << sigma;
    let num_rows = 1 << nu;
    let mut u_vec = vec![F::zero(); num_rows];

    for i in 0..num_rows {
        let mut sum = F::zero();
        let row_start = i * num_cols;
        let row_slice = &coefficients[row_start..row_start + num_cols];
        for (coeff, r) in row_slice.iter().zip(right_vec.iter()) {
            sum = sum + coeff.mul(r);
        }
        u_vec[i] = sum;
    }

    u_vec
}
