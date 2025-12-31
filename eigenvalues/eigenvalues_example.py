"""
Eigenvalues and Eigenvectors Example

Eigenvalues and eigenvectors are fundamental concepts in linear algebra.
An eigenvector of a matrix A is a nonzero vector v such that Av = λv,
where λ is the corresponding eigenvalue.
"""

import numpy as np


def demonstrate_eigenvalues():
    """Demonstrate eigenvalues and eigenvectors operations"""
    
    print("=" * 60)
    print("Eigenvalues and Eigenvectors Examples")
    print("=" * 60)
    
    # Example 1: Basic eigenvalue/eigenvector computation
    print("\n1. Basic Eigenvalue and Eigenvector Computation:")
    A = np.array([[4, 1],
                  [2, 3]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Matrix A:\n{A}")
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors (as columns):\n{eigenvectors}")
    
    # Verify: Av = λv for each eigenvector
    print("\nVerification (Av = λv):")
    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]
        Av = A @ v_i
        lambda_v = lambda_i * v_i
        print(f"  Eigenvalue {i+1}: λ = {lambda_i:.4f}")
        print(f"    Av = {Av}")
        print(f"    λv = {lambda_v}")
        print(f"    Are they equal? {np.allclose(Av, lambda_v)}")
    
    # Example 2: Diagonal matrix (eigenvalues are diagonal elements)
    print("\n2. Diagonal Matrix (Eigenvalues = Diagonal Elements):")
    D = np.array([[3, 0, 0],
                  [0, 5, 0],
                  [0, 0, 2]])
    eigenvals, eigenvecs = np.linalg.eig(D)
    print(f"Diagonal matrix D:\n{D}")
    print(f"Eigenvalues: {eigenvals}")
    print("Note: For diagonal matrices, eigenvalues are the diagonal elements")
    
    # Example 3: Identity matrix (all eigenvalues = 1)
    print("\n3. Identity Matrix:")
    I = np.eye(3)
    eigenvals, eigenvecs = np.linalg.eig(I)
    print(f"Identity matrix I (3x3):\n{I}")
    print(f"Eigenvalues: {eigenvals}")
    print("Note: All eigenvalues of identity matrix are 1")
    
    # Example 4: Properties of eigenvalues
    print("\n4. Properties of Eigenvalues:")
    A = np.array([[2, 1],
                  [1, 2]])
    eigenvals, eigenvecs = np.linalg.eig(A)
    print(f"Matrix A:\n{A}")
    print(f"Eigenvalues: {eigenvals}")
    
    print(f"\nProperty 1: Sum of eigenvalues = Trace of matrix")
    print(f"  Sum of eigenvalues: {np.sum(eigenvals):.4f}")
    print(f"  Trace of A: {np.trace(A):.4f}")
    print(f"  Equal? {np.isclose(np.sum(eigenvals), np.trace(A))}")
    
    print(f"\nProperty 2: Product of eigenvalues = Determinant")
    print(f"  Product of eigenvalues: {np.prod(eigenvals):.4f}")
    print(f"  Determinant of A: {np.linalg.det(A):.4f}")
    print(f"  Equal? {np.isclose(np.prod(eigenvals), np.linalg.det(A))}")
    
    # Example 5: Eigenvalues of transpose
    print("\n5. Eigenvalues of Transpose:")
    A = np.array([[1, 2],
                  [3, 4]])
    eigenvals_A, _ = np.linalg.eig(A)
    eigenvals_AT, _ = np.linalg.eig(A.T)
    print(f"Matrix A:\n{A}")
    print(f"Eigenvalues of A: {eigenvals_A}")
    print(f"Eigenvalues of A^T: {eigenvals_AT}")
    print("Note: A and A^T have the same eigenvalues")
    print(f"  Equal? {np.allclose(eigenvals_A, eigenvals_AT)}")
    
    # Example 6: Real vs Complex eigenvalues
    print("\n6. Real vs Complex Eigenvalues:")
    # Rotation matrix (has complex eigenvalues)
    theta = np.pi / 4  # 45 degrees
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    eigenvals, eigenvecs = np.linalg.eig(R)
    print(f"Rotation matrix (45°):\n{R}")
    print(f"Eigenvalues: {eigenvals}")
    print(f"Are they complex? {np.any(np.iscomplex(eigenvals))}")
    
    # Symmetric matrix (always has real eigenvalues)
    S = np.array([[1, 2],
                  [2, 1]])
    eigenvals_S, _ = np.linalg.eig(S)
    print(f"\nSymmetric matrix S:\n{S}")
    print(f"Eigenvalues: {eigenvals_S}")
    print("Note: Symmetric matrices always have real eigenvalues")
    
    # Example 7: Eigenvalue decomposition
    print("\n7. Eigenvalue Decomposition (A = PΛP⁻¹):")
    A = np.array([[4, 1],
                  [2, 3]])
    eigenvals, eigenvecs = np.linalg.eig(A)
    
    # Create diagonal matrix of eigenvalues
    Lambda = np.diag(eigenvals)
    P = eigenvecs  # Matrix of eigenvectors
    P_inv = np.linalg.inv(P)
    
    # Reconstruct A
    A_reconstructed = P @ Lambda @ P_inv
    
    print(f"Original matrix A:\n{A}")
    print(f"Eigenvalues (diagonal):\n{Lambda}")
    print(f"Eigenvectors (columns):\n{P}")
    print(f"Reconstructed A (PΛP⁻¹):\n{A_reconstructed}")
    print(f"Are they equal? {np.allclose(A, A_reconstructed)}")


if __name__ == "__main__":
    demonstrate_eigenvalues()

