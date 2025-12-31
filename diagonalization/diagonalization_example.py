"""
Matrix Diagonalization Example

Diagonalization is the process of finding a diagonal matrix D and an invertible 
matrix P such that A = P D P^(-1), where D contains the eigenvalues of A on 
the diagonal, and P contains the corresponding eigenvectors as columns.
"""

import numpy as np


def demonstrate_diagonalization():
    """Demonstrate matrix diagonalization"""
    
    print("=" * 60)
    print("Matrix Diagonalization Examples")
    print("=" * 60)
    
    # Example 1: Basic diagonalization
    print("\n1. Basic Diagonalization:")
    A = np.array([[4, 1],
                  [2, 3]])
    print(f"Matrix A:\n{A}")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors (as columns):\n{eigenvectors}")
    
    # Create diagonal matrix D
    D = np.diag(eigenvalues)
    print(f"\nDiagonal matrix D:\n{D}")
    
    # Create matrix P (eigenvectors as columns)
    P = eigenvectors
    print(f"Matrix P (eigenvectors):\n{P}")
    
    # Verify: A = P D P^(-1)
    P_inv = np.linalg.inv(P)
    reconstructed_A = P @ D @ P_inv
    print(f"\nReconstructed A = P D P^(-1):\n{reconstructed_A}")
    print(f"Are they equal? {np.allclose(A, reconstructed_A)}")
    
    # Example 2: Diagonalization of a 3x3 matrix
    print("\n2. Diagonalization of a 3x3 Matrix:")
    B = np.array([[2, 0, 0],
                  [0, 3, 0],
                  [0, 0, 4]])
    print(f"Matrix B (already diagonal):\n{B}")
    
    eigenvalues_B, eigenvectors_B = np.linalg.eig(B)
    print(f"Eigenvalues: {eigenvalues_B}")
    print(f"Eigenvectors:\n{eigenvectors_B}")
    
    # Example 3: Non-diagonalizable matrix (defective)
    print("\n3. Checking if a Matrix is Diagonalizable:")
    C = np.array([[1, 1],
                  [0, 1]])
    print(f"Matrix C:\n{C}")
    
    eigenvalues_C, eigenvectors_C = np.linalg.eig(C)
    print(f"Eigenvalues: {eigenvalues_C}")
    print(f"Eigenvectors:\n{eigenvectors_C}")
    
    # Check if we have enough linearly independent eigenvectors
    # A matrix is diagonalizable if it has n linearly independent eigenvectors
    # where n is the dimension
    rank = np.linalg.matrix_rank(eigenvectors_C)
    is_diagonalizable = rank == C.shape[0]
    print(f"Number of linearly independent eigenvectors: {rank}")
    print(f"Is diagonalizable? {is_diagonalizable}")
    
    # Example 4: Properties of diagonalization
    print("\n4. Properties of Diagonalization:")
    A = np.array([[5, 2],
                  [2, 5]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    
    print(f"Original matrix A:\n{A}")
    print(f"\nProperty 1: A = P D P^(-1)")
    print(f"  Verified: {np.allclose(A, P @ D @ P_inv)}")
    
    print(f"\nProperty 2: A^k = P D^k P^(-1)")
    k = 3
    A_power = np.linalg.matrix_power(A, k)
    D_power = np.linalg.matrix_power(D, k)
    reconstructed_power = P @ D_power @ P_inv
    print(f"  A^3:\n{A_power}")
    print(f"  P D^3 P^(-1):\n{reconstructed_power}")
    print(f"  Verified: {np.allclose(A_power, reconstructed_power)}")
    
    # Example 5: Determinant and trace from eigenvalues
    print("\n5. Determinant and Trace from Eigenvalues:")
    A = np.array([[3, 1],
                  [1, 3]])
    eigenvalues, _ = np.linalg.eig(A)
    print(f"Matrix A:\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    
    det_from_eigenvalues = np.prod(eigenvalues)
    det_direct = np.linalg.det(A)
    print(f"\nDeterminant from eigenvalues (product): {det_from_eigenvalues}")
    print(f"Determinant directly: {det_direct}")
    print(f"Equal? {np.isclose(det_from_eigenvalues, det_direct)}")
    
    trace_from_eigenvalues = np.sum(eigenvalues)
    trace_direct = np.trace(A)
    print(f"\nTrace from eigenvalues (sum): {trace_from_eigenvalues}")
    print(f"Trace directly: {trace_direct}")
    print(f"Equal? {np.isclose(trace_from_eigenvalues, trace_direct)}")
    
    # Example 6: Symmetric matrix (always diagonalizable)
    print("\n6. Symmetric Matrix (Always Diagonalizable):")
    S = np.array([[1, 2],
                  [2, 1]])
    print(f"Symmetric matrix S:\n{S}")
    print(f"Is symmetric? {np.allclose(S, S.T)}")
    
    eigenvalues_S, eigenvectors_S = np.linalg.eig(S)
    print(f"Eigenvalues: {eigenvalues_S}")
    print(f"Eigenvectors:\n{eigenvectors_S}")
    
    # For symmetric matrices, eigenvectors are orthogonal
    print(f"\nEigenvectors are orthogonal:")
    for i in range(eigenvectors_S.shape[1]):
        for j in range(i+1, eigenvectors_S.shape[1]):
            dot_product = np.dot(eigenvectors_S[:, i], eigenvectors_S[:, j])
            print(f"  v{i+1} · v{j+1} = {dot_product:.6f}")


if __name__ == "__main__":
    demonstrate_diagonalization()

