"""
Null Space, Nullity, and Kernel Exercise - SOLUTIONS

This file contains the solutions to the null space exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np
from scipy.linalg import null_space


def exercise_1_solution():
    """Solution to Exercise 1: Find Null Space"""
    A = np.array([[1, 2],
                  [2, 4]])
    
    # Use scipy.linalg.null_space() to find null space
    result = null_space(A)
    
    # Alternative: Solve Ax = 0 manually
    # x1 + 2x2 = 0, so x1 = -2x2
    # Basis vector: [-2, 1] or [1, -0.5]
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Calculate Nullity"""
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
    # Nullity is the dimension of null space
    null_space_vectors = null_space(A)
    result = null_space_vectors.shape[1]
    
    # Alternative: nullity = number of columns - rank
    # result = A.shape[1] - np.linalg.matrix_rank(A)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Verify Null Space Property"""
    A = np.array([[1, 1, 1],
                  [2, 2, 2]])
    x = np.array([1, -1, 0])  # This should be in the null space
    
    # Check if Ax is close to zero
    Ax = A @ x
    result = np.allclose(Ax, 0)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Rank-Nullity Theorem"""
    A = np.array([[1, 2, 3, 4],
                  [0, 1, 2, 3],
                  [0, 0, 0, 0]])
    
    # Rank-Nullity Theorem: rank(A) + nullity(A) = number of columns
    rank_A = np.linalg.matrix_rank(A)
    null_space_vectors = null_space(A)
    nullity_A = null_space_vectors.shape[1]
    num_columns = A.shape[1]
    
    # Check if rank + nullity equals number of columns
    result = (rank_A + nullity_A) == num_columns
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Null Space of Full Rank Matrix"""
    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])  # Identity matrix (full rank)
    
    # Find nullity
    null_space_vectors = null_space(A)
    nullity = null_space_vectors.shape[1]
    
    # Full rank matrix has nullity = 0
    result = (nullity == 0)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Null Space and Linear Dependence"""
    A = np.array([[1, 2, 3],
                  [2, 4, 6]])  # Third column is 1.5× second column
    
    # Find nullity
    null_space_vectors = null_space(A)
    nullity = null_space_vectors.shape[1]
    
    # If nullity > 0, columns are linearly dependent
    result = (nullity > 0)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Null Space Using SVD"""
    A = np.array([[1, 2],
                  [2, 4],
                  [3, 6]])
    
    # Perform SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    
    # Nullity equals number of zero (or near-zero) singular values
    # Use a small tolerance for floating point comparison
    zero_singular_values = np.sum(np.isclose(s, 0, atol=1e-10))
    result = zero_singular_values
    
    # Alternative: nullity = number of columns - number of non-zero singular values
    # non_zero_singular = np.sum(~np.isclose(s, 0, atol=1e-10))
    # result = A.shape[1] - non_zero_singular
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Null Space of Transpose"""
    A = np.array([[1, 2, 3],
                  [2, 4, 6]])
    
    # Find nullity of A
    null_space_A = null_space(A)
    nullity_A = null_space_A.shape[1]
    
    # Find nullity of A^T
    null_space_AT = null_space(A.T)
    nullity_AT = null_space_AT.shape[1]
    
    result = (nullity_A, nullity_AT)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Null Space of Product"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[1, 1],
                  [-1, -1]])  # Has null space
    
    # Find nullity of B
    null_space_B = null_space(B)
    nullity_B = null_space_B.shape[1]
    
    # Find nullity of AB
    AB = A @ B
    null_space_AB = null_space(AB)
    nullity_AB = null_space_AB.shape[1]
    
    # Check if nullity(AB) >= nullity(B)
    result = (nullity_AB >= nullity_B)
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Manual Null Space Calculation"""
    A = np.array([[1, 1],
                  [2, 2]])
    
    # Solve Ax = 0 manually
    # x1 + x2 = 0
    # 2x1 + 2x2 = 0 (same equation)
    # So x2 = -x1
    # A basis vector is [1, -1] or any scalar multiple
    
    result = np.array([1, -1])
    
    # Verify: A @ result should be close to zero
    # verification = A @ result  # Should be [0, 0]
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:")
    print(exercise_1_solution())
    
    print("\nExercise 2 Result:", exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

