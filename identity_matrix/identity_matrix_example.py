"""
Identity Matrix Example

The identity matrix is a square matrix with 1s on the main diagonal and 0s elsewhere.
It acts as the multiplicative identity for matrices.
"""

import numpy as np


def demonstrate_identity_matrix():
    """Demonstrate identity matrix operations"""
    
    print("=" * 60)
    print("Identity Matrix Examples")
    print("=" * 60)
    
    # Example 1: Creating identity matrices
    print("\n1. Creating Identity Matrices:")
    I2 = np.eye(2)
    I3 = np.eye(3)
    I4 = np.identity(4)
    print(f"2x2 Identity (using np.eye(2)):\n{I2}")
    print(f"\n3x3 Identity (using np.eye(3)):\n{I3}")
    print(f"\n4x4 Identity (using np.identity(4)):\n{I4}")
    
    # Example 2: Identity as multiplicative identity
    print("\n2. Identity Matrix as Multiplicative Identity:")
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    I = np.eye(3)
    print(f"Matrix A:\n{A}")
    print(f"\nIdentity matrix I:\n{I}")
    print(f"\nA @ I (A times identity):\n{A @ I}")
    print(f"I @ A (identity times A):\n{I @ A}")
    print("Note: A @ I = I @ A = A (identity preserves the matrix)")
    
    # Example 3: Identity with vectors
    print("\n3. Identity Matrix with Vectors:")
    v = np.array([2, 3, 4])
    I = np.eye(3)
    result = I @ v
    print(f"Vector v: {v}")
    print(f"Identity matrix I:\n{I}")
    print(f"I @ v: {result}")
    print("Note: I @ v = v (identity preserves the vector)")
    
    # Example 4: Properties of identity matrix
    print("\n4. Properties of Identity Matrix:")
    I = np.eye(3)
    
    print(f"Property 1: I^T = I (transpose of identity is itself)")
    print(f"I:\n{I}")
    print(f"I^T:\n{I.T}")
    print(f"Are they equal? {np.array_equal(I, I.T)}")
    
    print(f"\nProperty 2: det(I) = 1 (determinant is always 1)")
    print(f"det(I) = {np.linalg.det(I)}")
    
    print(f"\nProperty 3: I^n = I (any power of identity is itself)")
    I_squared = I @ I
    I_cubed = I @ I @ I
    print(f"I^2:\n{I_squared}")
    print(f"I^3:\n{I_cubed}")
    print(f"All equal to I? {np.array_equal(I, I_squared) and np.array_equal(I, I_cubed)}")
    
    # Example 5: Identity in matrix equations
    print("\n5. Identity in Matrix Equations:")
    A = np.array([[2, 1],
                  [1, 3]])
    I = np.eye(2)
    
    print(f"Matrix A:\n{A}")
    print(f"Identity I:\n{I}")
    print(f"\nA - I:\n{A - I}")
    print(f"A + I:\n{A + I}")
    print(f"2I (scalar multiple):\n{2 * I}")
    
    # Example 6: Inverse relationship
    print("\n6. Identity and Inverse Relationship:")
    A = np.array([[2, 1],
                  [1, 3]])
    A_inv = np.linalg.inv(A)
    I = np.eye(2)
    
    print(f"Matrix A:\n{A}")
    print(f"Inverse A^(-1):\n{A_inv}")
    print(f"\nA @ A^(-1):\n{A @ A_inv}")
    print(f"A^(-1) @ A:\n{A_inv @ A}")
    print(f"Both equal to I? {np.allclose(A @ A_inv, I) and np.allclose(A_inv @ A, I)}")
    print("Note: A @ A^(-1) = A^(-1) @ A = I")
    
    # Example 7: Creating identity with specific dtype
    print("\n7. Identity Matrix with Different Data Types:")
    I_int = np.eye(3, dtype=int)
    I_float = np.eye(3, dtype=float)
    print(f"Integer identity:\n{I_int}")
    print(f"Float identity:\n{I_float}")


if __name__ == "__main__":
    demonstrate_identity_matrix()

