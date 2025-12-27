"""
Matrix Determinant Example

The determinant is a scalar value that can be computed from a square matrix.
It provides important information about the matrix properties.
"""

import numpy as np


def demonstrate_determinant():
    """Demonstrate matrix determinant operations"""
    
    print("=" * 60)
    print("Matrix Determinant Examples")
    print("=" * 60)
    
    # Example 1: 2x2 matrix
    print("\n1. Determinant of a 2x2 matrix:")
    A = np.array([[3, 1],
                  [2, 4]])
    det_A = np.linalg.det(A)
    print(f"Matrix A:\n{A}")
    print(f"Determinant: {det_A}")
    print(f"Manual calculation: 3*4 - 1*2 = {3*4 - 1*2}")
    
    # Example 2: 3x3 matrix
    print("\n2. Determinant of a 3x3 matrix:")
    B = np.array([[1, 2, 3],
                  [0, 1, 4],
                  [5, 6, 0]])
    det_B = np.linalg.det(B)
    print(f"Matrix B:\n{B}")
    print(f"Determinant: {det_B}")
    
    # Example 3: Identity matrix
    print("\n3. Determinant of identity matrix:")
    I = np.eye(3)
    det_I = np.linalg.det(I)
    print(f"Identity matrix (3x3):\n{I}")
    print(f"Determinant: {det_I}")
    print("Note: Determinant of identity matrix is always 1")
    
    # Example 4: Singular matrix (zero determinant)
    print("\n4. Singular matrix (zero determinant):")
    C = np.array([[1, 2],
                  [2, 4]])  # Second row is 2× first row
    det_C = np.linalg.det(C)
    print(f"Matrix C:\n{C}")
    print(f"Determinant: {det_C}")
    print("Note: Zero determinant means the matrix is singular (not invertible)")
    print("This happens when rows/columns are linearly dependent")
    
    # Example 5: Determinant properties
    print("\n5. Determinant properties:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"\nProperty 1: det(A) = det(A^T)")
    print(f"  det(A) = {np.linalg.det(A)}")
    print(f"  det(A^T) = {np.linalg.det(A.T)}")
    print(f"  Equal? {np.isclose(np.linalg.det(A), np.linalg.det(A.T))}")
    
    print(f"\nProperty 2: det(AB) = det(A) * det(B)")
    print(f"  det(AB) = {np.linalg.det(A @ B)}")
    print(f"  det(A) * det(B) = {np.linalg.det(A) * np.linalg.det(B)}")
    print(f"  Equal? {np.isclose(np.linalg.det(A @ B), np.linalg.det(A) * np.linalg.det(B))}")
    
    print(f"\nProperty 3: det(kA) = k^n * det(A) for n×n matrix")
    k = 2
    n = A.shape[0]
    print(f"  k = {k}, n = {n}")
    print(f"  det({k}A) = {np.linalg.det(k * A)}")
    print(f"  {k}^{n} * det(A) = {k**n} * {np.linalg.det(A)} = {k**n * np.linalg.det(A)}")
    print(f"  Equal? {np.isclose(np.linalg.det(k * A), k**n * np.linalg.det(A))}")
    
    # Example 6: Effect of row operations
    print("\n6. Effect of row operations on determinant:")
    D = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    det_D = np.linalg.det(D)
    print(f"Original matrix D:\n{D}")
    print(f"Determinant: {det_D}")
    
    # Swap two rows
    D_swapped = D.copy()
    D_swapped[[0, 1]] = D_swapped[[1, 0]]
    det_D_swapped = np.linalg.det(D_swapped)
    print(f"\nAfter swapping rows 0 and 1:\n{D_swapped}")
    print(f"Determinant: {det_D_swapped}")
    print(f"Note: Swapping rows multiplies determinant by -1")
    print(f"  {det_D_swapped} ≈ -1 * {det_D}? {np.isclose(det_D_swapped, -1 * det_D)}")


if __name__ == "__main__":
    demonstrate_determinant()

