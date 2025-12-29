"""
Matrix Multiplication Example

Matrix multiplication is a fundamental operation in linear algebra.
It combines two matrices to produce a third matrix.
"""

import numpy as np


def demonstrate_matrix_multiplication():
    """Demonstrate matrix multiplication operations"""
    
    print("=" * 60)
    print("Matrix Multiplication Examples")
    print("=" * 60)
    
    # Example 1: Basic 2x2 matrix multiplication
    print("\n1. Basic 2x2 Matrix Multiplication:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    result = A @ B
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Result (A @ B):\n{result}")
    print("\nCalculation:")
    print(f"  [1, 2] · [5, 7] = {1*5 + 2*7}, [1, 2] · [6, 8] = {1*6 + 2*8}")
    print(f"  [3, 4] · [5, 7] = {3*5 + 4*7}, [3, 4] · [6, 8] = {3*6 + 4*8}")
    
    # Example 2: Different ways to compute matrix multiplication
    print("\n2. Different Ways to Compute Matrix Multiplication:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"  A @ B:\n{A @ B}")
    print(f"  np.dot(A, B):\n{np.dot(A, B)}")
    print(f"  np.matmul(A, B):\n{np.matmul(A, B)}")
    print("  All methods give the same result!")
    
    # Example 3: Matrix-vector multiplication
    print("\n3. Matrix-Vector Multiplication:")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    v = np.array([7, 8, 9])
    result = A @ v
    print(f"Matrix A (2x3):\n{A}")
    print(f"Vector v: {v}")
    print(f"Result (A @ v): {result}")
    print("Each row of A is dotted with v")
    
    # Example 4: Dimension requirements
    print("\n4. Dimension Requirements:")
    print("For A @ B to be valid:")
    print("  - Number of columns in A must equal number of rows in B")
    print("  - If A is m×n and B is n×p, then A @ B is m×p")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2×3
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])  # 3×2
    result = A @ B
    print(f"\nExample:")
    print(f"  A shape: {A.shape} (2×3)")
    print(f"  B shape: {B.shape} (3×2)")
    print(f"  Result shape: {result.shape} (2×2)")
    print(f"  Result:\n{result}")
    
    # Example 5: Non-commutative property
    print("\n5. Matrix Multiplication is NOT Commutative:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"\nA @ B:\n{A @ B}")
    print(f"B @ A:\n{B @ A}")
    print("Note: A @ B ≠ B @ A in general!")
    
    # Example 6: Associative property
    print("\n6. Matrix Multiplication is Associative:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    C = np.array([[9, 10],
                  [11, 12]])
    print(f"Matrices A, B, C")
    left = (A @ B) @ C
    right = A @ (B @ C)
    print(f"(A @ B) @ C:\n{left}")
    print(f"A @ (B @ C):\n{right}")
    print(f"Are they equal? {np.array_equal(left, right)}")
    print("Note: (A @ B) @ C = A @ (B @ C)")
    
    # Example 7: Distributive property
    print("\n7. Matrix Multiplication is Distributive:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    C = np.array([[9, 10],
                  [11, 12]])
    print(f"Matrices A, B, C")
    left = A @ (B + C)
    right = A @ B + A @ C
    print(f"A @ (B + C):\n{left}")
    print(f"A @ B + A @ C:\n{right}")
    print(f"Are they equal? {np.array_equal(left, right)}")
    print("Note: A @ (B + C) = A @ B + A @ C")
    
    # Example 8: Identity matrix
    print("\n8. Identity Matrix Multiplication:")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    I = np.eye(3)  # 3×3 identity
    I2 = np.eye(2)  # 2×2 identity
    print(f"Matrix A:\n{A}")
    print(f"\nA @ I (3×3 identity):\n{A @ I}")
    print(f"I2 @ A (2×2 identity):\n{I2 @ A}")
    print("Note: A @ I = A and I @ A = A (when dimensions match)")
    
    # Example 9: Scalar multiplication
    print("\n9. Scalar Multiplication with Matrix Multiplication:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    k = 3
    print(f"k = {k}")
    print(f"k(A @ B):\n{k * (A @ B)}")
    print(f"(kA) @ B:\n{(k * A) @ B}")
    print(f"A @ (kB):\n{A @ (k * B)}")
    print("Note: k(A @ B) = (kA) @ B = A @ (kB)")


if __name__ == "__main__":
    demonstrate_matrix_multiplication()

