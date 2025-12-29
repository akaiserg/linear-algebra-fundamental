"""
Matrix Inverse Example

The inverse of a matrix A, denoted as A^(-1), is a matrix such that:
A × A^(-1) = A^(-1) × A = I (identity matrix)
"""

import numpy as np


def demonstrate_inverse():
    """Demonstrate matrix inverse operations"""
    
    print("=" * 60)
    print("Matrix Inverse Examples")
    print("=" * 60)
    
    # Example 1: 2x2 matrix inverse
    print("\n1. Inverse of a 2x2 matrix:")
    A = np.array([[3, 1],
                  [2, 4]])
    print(f"Matrix A:\n{A}")
    
    try:
        A_inv = np.linalg.inv(A)
        print(f"Inverse A^(-1):\n{A_inv}")
        
        # Verify: A × A^(-1) = I
        identity_check = A @ A_inv
        print(f"\nVerification: A × A^(-1) =\n{identity_check}")
        print(f"Is it identity? {np.allclose(identity_check, np.eye(2))}")
    except np.linalg.LinAlgError:
        print("Matrix is singular (not invertible)")
    
    # Example 2: 3x3 matrix inverse
    print("\n2. Inverse of a 3x3 matrix:")
    B = np.array([[1, 2, 3],
                  [0, 1, 4],
                  [5, 6, 0]])
    print(f"Matrix B:\n{B}")
    
    try:
        B_inv = np.linalg.inv(B)
        print(f"Inverse B^(-1):\n{B_inv}")
        
        # Verify
        identity_check = B @ B_inv
        print(f"\nVerification: B × B^(-1) =\n{identity_check}")
        print(f"Is it identity? {np.allclose(identity_check, np.eye(3))}")
    except np.linalg.LinAlgError:
        print("Matrix is singular (not invertible)")
    
    # Example 3: Singular matrix (not invertible)
    print("\n3. Singular matrix (not invertible):")
    C = np.array([[1, 2],
                  [2, 4]])  # Second row is 2× first row (linearly dependent)
    print(f"Matrix C:\n{C}")
    print(f"Determinant: {np.linalg.det(C)}")
    
    try:
        C_inv = np.linalg.inv(C)
        print(f"Inverse: {C_inv}")
    except np.linalg.LinAlgError as e:
        print(f"Cannot compute inverse: {e}")
        print("A matrix is singular (not invertible) if det(A) = 0")
    
    # Example 4: Properties of inverse
    print("\n4. Properties of matrix inverse:")
    A = np.array([[2, 1],
                  [1, 1]])
    
    print(f"Matrix A:\n{A}")
    
    A_inv = np.linalg.inv(A)
    print(f"A^(-1):\n{A_inv}")
    
    print(f"\nProperty 1: (A^(-1))^(-1) = A")
    A_inv_inv = np.linalg.inv(A_inv)
    print(f"  (A^(-1))^(-1) =\n{A_inv_inv}")
    print(f"  Equal to A? {np.allclose(A_inv_inv, A)}")
    
    print(f"\nProperty 2: (A^T)^(-1) = (A^(-1))^T")
    A_T_inv = np.linalg.inv(A.T)
    A_inv_T = A_inv.T
    print(f"  (A^T)^(-1) =\n{A_T_inv}")
    print(f"  (A^(-1))^T =\n{A_inv_T}")
    print(f"  Equal? {np.allclose(A_T_inv, A_inv_T)}")
    
    # Example 5: Inverse of product
    print("\n5. Inverse of product: (AB)^(-1) = B^(-1) A^(-1)")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    AB = A @ B
    AB_inv = np.linalg.inv(AB)
    B_inv_A_inv = np.linalg.inv(B) @ np.linalg.inv(A)
    
    print(f"\n  (AB)^(-1) =\n{AB_inv}")
    print(f"  B^(-1) A^(-1) =\n{B_inv_A_inv}")
    print(f"  Equal? {np.allclose(AB_inv, B_inv_A_inv)}")
    print("  Note: Order is reversed!")
    
    # Example 6: Solving linear system using inverse
    print("\n6. Solving linear system Ax = b using inverse:")
    A = np.array([[2, 1],
                  [1, 1]])
    b = np.array([5, 3])
    
    print(f"System: Ax = b")
    print(f"  A =\n{A}")
    print(f"  b = {b}")
    
    # Solution: x = A^(-1) b
    A_inv = np.linalg.inv(A)
    x = A_inv @ b
    
    print(f"\n  Solution: x = A^(-1) b = {x}")
    print(f"  Verification: Ax = {A @ x}")
    print(f"  Matches b? {np.allclose(A @ x, b)}")
    
    # Example 7: Identity matrix inverse
    print("\n7. Inverse of identity matrix:")
    I = np.eye(3)
    print(f"Identity matrix I:\n{I}")
    I_inv = np.linalg.inv(I)
    print(f"I^(-1) =\n{I_inv}")
    print("Note: I^(-1) = I (identity is its own inverse)")


if __name__ == "__main__":
    demonstrate_inverse()

