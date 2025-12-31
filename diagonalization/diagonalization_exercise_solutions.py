"""
Matrix Diagonalization Exercise - SOLUTIONS

This file contains the solutions to the diagonalization exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Find Eigenvalues and Eigenvectors"""
    A = np.array([[4, 1],
                  [2, 3]])
    
    # Use np.linalg.eig() to find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Note: eigenvectors are returned as columns
    return eigenvalues, eigenvectors


def exercise_2_solution():
    """Solution to Exercise 2: Create Diagonal Matrix from Eigenvalues"""
    A = np.array([[5, 2],
                  [2, 5]])
    
    # Find eigenvalues
    eigenvalues, _ = np.linalg.eig(A)
    
    # Create diagonal matrix using np.diag()
    D = np.diag(eigenvalues)
    
    return D


def exercise_3_solution():
    """Solution to Exercise 3: Verify Diagonalization A = P D P^(-1)"""
    A = np.array([[4, 1],
                  [2, 3]])
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create diagonal matrix D
    D = np.diag(eigenvalues)
    
    # Matrix P is the eigenvectors (as columns)
    P = eigenvectors
    
    # Compute P^(-1)
    P_inv = np.linalg.inv(P)
    
    # Verify: A = P D P^(-1)
    reconstructed_A = P @ D @ P_inv
    
    # Use np.allclose() for floating point comparison
    result = np.allclose(A, reconstructed_A)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Compute A^k Using Diagonalization"""
    A = np.array([[2, 1],
                  [1, 2]])
    k = 3
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create diagonal matrix D
    D = np.diag(eigenvalues)
    
    # Matrix P is the eigenvectors
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    
    # Compute D^k (raising diagonal matrix to power k)
    D_power = np.linalg.matrix_power(D, k)
    # Alternative: D_power = np.diag(eigenvalues ** k)
    
    # Compute A^k = P D^k P^(-1)
    result = P @ D_power @ P_inv
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Determinant from Eigenvalues"""
    A = np.array([[3, 1],
                  [1, 3]])
    
    # Compute determinant directly
    det_A = np.linalg.det(A)
    
    # Find eigenvalues
    eigenvalues, _ = np.linalg.eig(A)
    
    # Product of eigenvalues
    product_eigenvalues = np.prod(eigenvalues)
    
    # Compare
    result = np.isclose(det_A, product_eigenvalues)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Trace from Eigenvalues"""
    A = np.array([[5, 2],
                  [2, 5]])
    
    # Compute trace directly
    trace_A = np.trace(A)
    
    # Find eigenvalues
    eigenvalues, _ = np.linalg.eig(A)
    
    # Sum of eigenvalues
    sum_eigenvalues = np.sum(eigenvalues)
    
    # Compare
    result = np.isclose(trace_A, sum_eigenvalues)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Check if Matrix is Diagonalizable"""
    A = np.array([[1, 1],
                  [0, 1]])
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # A matrix is diagonalizable if it has n linearly independent eigenvectors
    # where n is the dimension of the matrix
    # Check the rank of the eigenvector matrix
    rank = np.linalg.matrix_rank(eigenvectors)
    n = A.shape[0]
    
    # If rank equals dimension, we have enough linearly independent eigenvectors
    result = rank == n
    
    # Note: This matrix is NOT diagonalizable (defective matrix)
    # It has only one linearly independent eigenvector
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Diagonalize a Symmetric Matrix"""
    A = np.array([[1, 2],
                  [2, 1]])
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # For symmetric matrices, eigenvectors are orthogonal
    # Check if eigenvectors are orthogonal (dot product = 0)
    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    
    dot_product = np.dot(v1, v2)
    
    # Check if dot product is close to 0 (orthogonal)
    result = np.isclose(dot_product, 0.0)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Reconstruct Matrix from Diagonalization"""
    eigenvalues = np.array([6, 1])
    eigenvectors = np.array([[1, 1],
                             [1, -1]])
    
    # Create diagonal matrix D from eigenvalues
    D = np.diag(eigenvalues)
    
    # Matrix P is the eigenvectors (as columns)
    P = eigenvectors
    
    # Compute P^(-1)
    P_inv = np.linalg.inv(P)
    
    # Reconstruct A = P D P^(-1)
    A = P @ D @ P_inv
    
    return A


def exercise_10_solution():
    """Solution to Exercise 10: Power of Matrix Using Diagonalization"""
    A = np.array([[2, 0, 0],
                  [0, 3, 0],
                  [0, 0, 4]])
    k = 5
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create diagonal matrix D
    D = np.diag(eigenvalues)
    
    # Matrix P is the eigenvectors
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    
    # Compute D^k
    D_power = np.linalg.matrix_power(D, k)
    # Alternative: D_power = np.diag(eigenvalues ** k)
    
    # Compute A^k = P D^k P^(-1)
    result = P @ D_power @ P_inv
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:")
    eigvals, eigvecs = exercise_1_solution()
    print(f"  Eigenvalues: {eigvals}")
    print(f"  Eigenvectors shape: {eigvecs.shape}")
    
    print("\nExercise 2 Result:")
    print(exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:")
    print(exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:")
    print(exercise_9_solution())
    
    print("\nExercise 10 Result:")
    print(exercise_10_solution())

