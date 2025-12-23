#!/usr/bin/env python3
"""
Sparse linear solver for graph-based SLAM.

This module provides efficient sparse matrix operations suitable for
real-time SLAM on embedded hardware like the Jetson Nano.

Uses scipy.sparse for memory-efficient storage and fast solves.
Falls back to dense operations if scipy.sparse is unavailable.
"""

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse import linalg as splinalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SparseSolver:
    """
    Sparse linear system solver for SLAM optimization.
    
    Solves systems of the form: H @ dx = -b
    where H is the information matrix (J^T @ Sigma^-1 @ J)
    and b is the gradient (J^T @ Sigma^-1 @ r)
    """
    
    def __init__(self, use_sparse: bool = True, regularization: float = 1e-6):
        """
        Args:
            use_sparse: Use sparse matrices if scipy is available
            regularization: Tikhonov regularization for numerical stability
        """
        self.use_sparse = use_sparse and HAS_SCIPY
        self.regularization = regularization
    
    def solve(self, H: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve H @ dx = -b for dx.
        
        Args:
            H: Information matrix (symmetric positive semi-definite)
            b: Gradient vector
            
        Returns:
            dx: Solution vector
        """
        n = H.shape[0]
        
        # Add Tikhonov regularization for numerical stability
        if self.regularization > 0:
            H = H + self.regularization * np.eye(n)
        
        if self.use_sparse:
            return self._solve_sparse(H, b)
        else:
            return self._solve_dense(H, b)
    
    def _solve_sparse(self, H: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using sparse Cholesky or conjugate gradient."""
        H_sparse = sparse.csc_matrix(H)
        
        try:
            # Try sparse Cholesky first (fastest for SPD matrices)
            # Use sparse LU as scipy doesn't have sparse Cholesky built-in
            dx = splinalg.spsolve(H_sparse, -b)
            return dx
        except Exception:
            # Fall back to conjugate gradient (iterative)
            dx, info = splinalg.cg(H_sparse, -b, tol=1e-8, maxiter=500)
            if info != 0:
                # CG didn't converge, fall back to dense
                return self._solve_dense(H, b)
            return dx
    
    def _solve_dense(self, H: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using dense Cholesky decomposition."""
        try:
            # Try Cholesky (fastest for SPD)
            L = np.linalg.cholesky(H)
            # Solve L @ y = -b, then L^T @ dx = y
            y = np.linalg.solve(L, -b)
            dx = np.linalg.solve(L.T, y)
            return dx
        except np.linalg.LinAlgError:
            # Matrix not positive definite, use general solver
            try:
                dx = np.linalg.solve(H, -b)
                return dx
            except np.linalg.LinAlgError:
                # Singular matrix, use pseudo-inverse
                dx = np.linalg.lstsq(H, -b, rcond=None)[0]
                return dx


class IncrementalSolver:
    """
    Incremental solver that maintains a running factorization.
    
    This is more efficient for online SLAM where we add one factor
    at a time, similar to iSAM's incremental updates.
    """
    
    def __init__(self, initial_size: int = 3):
        """
        Args:
            initial_size: Initial state dimension (typically 3 for pose)
        """
        self.n = initial_size
        self.H = np.zeros((initial_size, initial_size))
        self.b = np.zeros(initial_size)
        self.solver = SparseSolver()
        
        # Track which variables need relinearization
        self.dirty_vars = set()
        self.relinearize_threshold = 0.1
    
    def resize(self, new_size: int):
        """Expand the system to accommodate new variables."""
        if new_size <= self.n:
            return
        
        H_new = np.zeros((new_size, new_size))
        b_new = np.zeros(new_size)
        
        H_new[:self.n, :self.n] = self.H
        b_new[:self.n] = self.b
        
        self.H = H_new
        self.b = b_new
        self.n = new_size
    
    def add_factor_contribution(self, 
                                 J: np.ndarray, 
                                 r: np.ndarray, 
                                 Sigma_inv: np.ndarray,
                                 var_indices: list):
        """
        Add a factor's contribution to H and b.
        
        Args:
            J: Jacobian of the factor (m x k where k is local var dimension)
            r: Residual vector (m,)
            Sigma_inv: Inverse covariance (m x m)
            var_indices: List of global indices for each variable block
        """
        # J^T @ Sigma^-1 @ J contribution to H
        # J^T @ Sigma^-1 @ r contribution to b
        
        JtS = J.T @ Sigma_inv  # (k, m)
        H_local = JtS @ J      # (k, k)
        b_local = JtS @ r      # (k,)
        
        # Scatter into global H and b
        k = 0
        for i, idx_i in enumerate(var_indices):
            dim_i = 3 if idx_i < 3 else 2  # poses are 3D, landmarks are 2D
            # Adjust for variable block sizes
            if i == 0:
                start_i = idx_i
                end_i = start_i + dim_i
                local_start_i = 0
                local_end_i = dim_i
            else:
                start_i = idx_i
                end_i = start_i + 2  # landmarks are 2D
                local_start_i = k
                local_end_i = k + 2
            
            self.b[start_i:end_i] += b_local[local_start_i:local_end_i]
            
            k2 = 0
            for j, idx_j in enumerate(var_indices):
                dim_j = 3 if idx_j < 3 else 2
                if j == 0:
                    start_j = idx_j
                    end_j = start_j + dim_j
                    local_start_j = 0
                    local_end_j = dim_j
                else:
                    start_j = idx_j
                    end_j = start_j + 2
                    local_start_j = k2
                    local_end_j = k2 + 2
                
                self.H[start_i:end_i, start_j:end_j] += H_local[local_start_i:local_end_i, local_start_j:local_end_j]
                k2 += dim_j
            k += dim_i
    
    def solve(self) -> np.ndarray:
        """Solve the current system."""
        return self.solver.solve(self.H, self.b)
    
    def reset(self):
        """Clear the accumulated H and b for re-linearization."""
        self.H = np.zeros((self.n, self.n))
        self.b = np.zeros(self.n)
