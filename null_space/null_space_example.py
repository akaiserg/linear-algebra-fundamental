"""
Null Space, Nullity, and Kernel Example

The null space (also called kernel) of a matrix A is the set of all vectors x
such that Ax = 0. The nullity is the dimension of the null space.
"""

import numpy as np
from scipy.linalg import null_space


def demonstrate_null_space():
    """Demonstrate null space, nullity, and kernel operations"""
    
    print("=" * 60)
    print("Null Space, Nullity, and Kernel Examples")
    print("=" * 60)
    
    # Example 1: Basic null space
    print("\n1. Finding Null Space of a Matrix:")
    A = np.array([[1, 2, 3],
                  [2, 4, 6]])  # Second row is 2× first row (linearly dependent)
    print(f"Matrix A:\n{A}")
    print(f"Shape: {A.shape}")
    
    # Find null space using scipy
    null_space_vectors = null_space(A)
    print(f"\nNull space vectors (basis):\n{null_space_vectors}")
    print(f"Nullity (dimension of null space): {null_space_vectors.shape[1]}")
    
    # Verify: A @ null_space_vectors should be close to zero
    verification = A @ null_space_vectors
    print(f"\nVerification (A @ null_space_vectors):\n{verification}")
    print(f"All close to zero? {np.allclose(verification, 0)}")
    
    # Example 2: Matrix with no null space (full rank)
    print("\n2. Matrix with No Null Space (Full Rank):")
    B = np.array([[1, 0],
                  [0, 1]])  # Identity matrix
    print(f"Matrix B (Identity):\n{B}")
    null_space_B = null_space(B)
    print(f"Null space vectors:\n{null_space_B}")
    print(f"Nullity: {null_space_B.shape[1]}")
    print("Note: Full rank matrices have nullity = 0 (only zero vector)")
    
    # Example 3: Rank-Nullity Theorem
    print("\n3. Rank-Nullity Theorem:")
    print("   rank(A) + nullity(A) = number of columns")
    A = np.array([[1, 2, 3],
                  [0, 1, 2],
                  [0, 0, 0]])  # Third row is zero
    print(f"\nMatrix A:\n{A}")
    
    rank_A = np.linalg.matrix_rank(A)
    null_space_A = null_space(A)
    nullity_A = null_space_A.shape[1]
    num_columns = A.shape[1]
    
    print(f"Rank: {rank_A}")
    print(f"Nullity: {nullity_A}")
    print(f"Number of columns: {num_columns}")
    print(f"Rank + Nullity = {rank_A} + {nullity_A} = {rank_A + nullity_A}")
    print(f"Equals number of columns? {rank_A + nullity_A == num_columns}")
    
    # Example 4: Manual null space calculation
    print("\n4. Understanding Null Space:")
    print("   Null space consists of all vectors x such that Ax = 0")
    A = np.array([[1, 1],
                  [2, 2]])  # Linearly dependent rows
    print(f"\nMatrix A:\n{A}")
    
    # Solve Ax = 0
    # This gives: x1 + x2 = 0, so x2 = -x1
    # Null space is spanned by [1, -1]
    print("Solving Ax = 0:")
    print("  x1 + x2 = 0")
    print("  2x1 + 2x2 = 0")
    print("  Solution: x2 = -x1")
    print("  Null space basis: [1, -1]")
    
    null_space_manual = np.array([[1], [-1]])
    verification_manual = A @ null_space_manual
    print(f"\nVerification: A @ [1, -1]^T = {verification_manual.flatten()}")
    
    # Example 5: Null space and linear independence
    print("\n5. Null Space and Linear Independence:")
    print("   If null space contains only zero vector, columns are linearly independent")
    A_independent = np.array([[1, 0],
                              [0, 1]])
    A_dependent = np.array([[1, 2],
                            [2, 4]])
    
    null_independent = null_space(A_independent)
    null_dependent = null_space(A_dependent)
    
    print(f"\nMatrix with independent columns:\n{A_independent}")
    print(f"Nullity: {null_independent.shape[1]} (only zero vector)")
    
    print(f"\nMatrix with dependent columns:\n{A_dependent}")
    print(f"Nullity: {null_dependent.shape[1]} (non-trivial null space)")
    
    # Example 6: Using SVD to find null space
    print("\n6. Finding Null Space Using SVD:")
    A = np.array([[1, 2, 3],
                  [2, 4, 6]])
    print(f"Matrix A:\n{A}")
    
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    print(f"\nSVD decomposition:")
    print(f"Singular values: {s}")
    print(f"Number of zero singular values: {np.sum(np.isclose(s, 0))}")
    print(f"Nullity (from SVD): {np.sum(np.isclose(s, 0))}")
    
    # Null space is spanned by columns of V corresponding to zero singular values
    # In this case, last column(s) of V
    V = Vt.T
    print(f"\nRight singular vectors V:\n{V}")
    print("Columns of V corresponding to zero singular values form null space basis")


if __name__ == "__main__":
    demonstrate_null_space()

