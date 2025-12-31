"""
Eigenvalues and Eigenvectors Exercise - SOLUTIONS

This file contains the solutions to the eigenvalue and eigenvector exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Eigenvalue and Eigenvector Computation"""
    A = np.array([[3, 1],
                  [0, 2]])
    
    # Use np.linalg.eig() to compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Note: eigenvectors are returned as columns of the matrix
    return eigenvalues, eigenvectors


def exercise_2_solution():
    """Solution to Exercise 2: Verify Eigenvalue Property (Av = λv)"""
    A = np.array([[4, 2],
                  [1, 3]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Verify Av = λv for each eigenvalue/eigenvector pair
    all_verified = True
    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]  # Get i-th eigenvector (as column)
        Av = A @ v_i
        lambda_v = lambda_i * v_i
        if not np.allclose(Av, lambda_v):
            all_verified = False
            break
    
    return all_verified


def exercise_3_solution():
    """Solution to Exercise 3: Eigenvalues of Diagonal Matrix"""
    D = np.array([[5, 0, 0],
                  [0, 3, 0],
                  [0, 0, 7]])
    
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(D)
    
    # Get diagonal elements
    diagonal_elements = np.diag(D)
    
    # Compare (need to sort since order might differ)
    result = np.allclose(np.sort(eigenvalues), np.sort(diagonal_elements))
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Sum of Eigenvalues Equals Trace"""
    A = np.array([[2, 1],
                  [1, 2]])
    
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(A)
    
    # Compute trace (sum of diagonal elements)
    trace_A = np.trace(A)
    
    # Compare sum of eigenvalues with trace
    result = np.isclose(np.sum(eigenvalues), trace_A)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Product of Eigenvalues Equals Determinant"""
    A = np.array([[3, 1],
                  [2, 4]])
    
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(A)
    
    # Compute determinant
    det_A = np.linalg.det(A)
    
    # Compare product of eigenvalues with determinant
    result = np.isclose(np.prod(eigenvalues), det_A)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Eigenvalues of Transpose"""
    A = np.array([[1, 3],
                  [2, 4]])
    
    # Compute eigenvalues of A
    eigenvals_A, _ = np.linalg.eig(A)
    
    # Compute eigenvalues of A^T
    eigenvals_AT, _ = np.linalg.eig(A.T)
    
    # Compare (need to sort since order might differ)
    result = np.allclose(np.sort(eigenvals_A), np.sort(eigenvals_AT))
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Eigenvalue Decomposition"""
    A = np.array([[4, 1],
                  [2, 3]])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create diagonal matrix of eigenvalues
    Lambda = np.diag(eigenvalues)
    
    # P is the matrix of eigenvectors (as columns)
    P = eigenvectors
    
    # Compute P inverse
    P_inv = np.linalg.inv(P)
    
    # Reconstruct A: A = PΛP⁻¹
    A_reconstructed = P @ Lambda @ P_inv
    
    # Check if reconstruction matches original
    result = np.allclose(A, A_reconstructed)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Eigenvalues of Identity Matrix"""
    # Create 3x3 identity matrix
    I = np.eye(3)
    
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(I)
    
    # Check if all eigenvalues equal 1
    result = np.allclose(eigenvalues, 1.0)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Symmetric Matrix Has Real Eigenvalues"""
    S = np.array([[1, 2],
                  [2, 1]])
    
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(S)
    
    # Check if all eigenvalues are real (imaginary part is zero)
    # Method 1: Check if all are real using np.isreal()
    result = np.all(np.isreal(eigenvalues))
    
    # Method 2: Check if imaginary parts are close to zero
    # result = np.allclose(eigenvalues.imag, 0)
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Power of Matrix Using Eigenvalues"""
    A = np.array([[2, 1],
                  [1, 2]])
    
    # Compute eigenvalues of A
    eigenvals_A, _ = np.linalg.eig(A)
    
    # Compute A^2
    A_squared = A @ A
    
    # Compute eigenvalues of A^2
    eigenvals_A2, _ = np.linalg.eig(A_squared)
    
    # Verify: if λ is eigenvalue of A, then λ^2 is eigenvalue of A^2
    # Square the eigenvalues of A
    eigenvals_A_squared = eigenvals_A ** 2
    
    # Compare (need to sort since order might differ)
    result = np.allclose(np.sort(eigenvals_A_squared), np.sort(eigenvals_A2))
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:")
    eigenvals, eigenvecs = exercise_1_solution()
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    print("\nExercise 2 Result:", exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

