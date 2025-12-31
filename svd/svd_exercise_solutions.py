"""
Singular Value Decomposition (SVD) Exercise - SOLUTIONS

This file contains the solutions to the SVD exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic SVD Decomposition"""
    A = np.array([[1, 2],
                  [3, 4]])
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Note: np.linalg.svd returns V^T (V transpose), not V
    # If you need V, use V = Vt.T
    # For consistency, we'll return Vt as V (which is the standard convention)
    
    return (U, S, Vt)


def exercise_2_solution():
    """Solution to Exercise 2: Reconstruct Matrix from SVD"""
    A = np.array([[1, 2],
                  [3, 4]])
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Reconstruct: A = U @ diag(S) @ V^T
    # Since np.linalg.svd returns Vt (which is V^T), we can use it directly
    S_matrix = np.diag(S)
    reconstructed = U @ S_matrix @ Vt
    
    return reconstructed


def exercise_3_solution():
    """Solution to Exercise 3: Singular Values"""
    A = np.array([[2, 0],
                  [0, 3]])
    
    # Get singular values
    S = np.linalg.svd(A, compute_uv=False)
    
    return S


def exercise_4_solution():
    """Solution to Exercise 4: Rank of Matrix Using SVD"""
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
    # Get singular values
    S = np.linalg.svd(A, compute_uv=False)
    
    # Count non-zero singular values (using tolerance for floating point)
    tolerance = 1e-10
    rank = np.sum(S > tolerance)
    
    return rank


def exercise_5_solution():
    """Solution to Exercise 5: Low-Rank Approximation"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    k = 1
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only first k singular values and corresponding vectors
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct rank-k approximation
    S_k_matrix = np.diag(S_k)
    A_k = U_k @ S_k_matrix @ Vt_k
    
    return A_k


def exercise_6_solution():
    """Solution to Exercise 6: SVD Property - Orthogonal Matrices"""
    A = np.array([[1, 2],
                  [3, 4]])
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Check if U^T U = I
    U_orthogonal = np.allclose(U.T @ U, np.eye(U.shape[1]))
    
    # Check if V^T V = I (Vt is V^T, so V = Vt.T)
    # We need to check V^T V = Vt @ Vt.T = I
    V_orthogonal = np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]))
    
    result = U_orthogonal and V_orthogonal
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: SVD of Transpose"""
    A = np.array([[1, 2],
                  [3, 4]])
    
    # Get singular values of A
    S_A = np.linalg.svd(A, compute_uv=False)
    
    # Get singular values of A^T
    S_AT = np.linalg.svd(A.T, compute_uv=False)
    
    # Compare (they should be the same)
    result = np.allclose(S_A, S_AT)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: SVD for Square Matrix"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    # Get singular values
    S = np.linalg.svd(A, compute_uv=False)
    
    # Return number of singular values
    result = len(S)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: SVD for Rectangular Matrix"""
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Get shapes
    U_shape = U.shape
    S_length = len(S)
    V_shape = Vt.shape  # Vt is k×n where k=min(m,n) and n=4
    
    # Note: With full_matrices=False, Vt is k×n (2×4)
    # With full_matrices=True, Vt would be n×n (4×4)
    
    return (U_shape, S_length, V_shape)


def exercise_10_solution():
    """Solution to Exercise 10: Compression Using SVD"""
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    k = 2
    
    # Original size
    m, n = A.shape
    original_size = m * n
    
    # Compressed size: U (m×k) + S (k) + V^T (k×n)
    # Note: With full_matrices=False, we store:
    # - U: m×k
    # - S: k values
    # - Vt: k×n
    compressed_size = m * k + k + k * n
    
    # Compression ratio
    ratio = original_size / compressed_size
    
    return ratio


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:")
    U, S, Vt = exercise_1_solution()
    print(f"  U shape: {U.shape}, S: {S}, Vt shape: {Vt.shape}")
    
    print("\nExercise 2 Result:")
    print(exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:")
    print(exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

