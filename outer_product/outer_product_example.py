"""
Outer Product Example

The outer product (also called tensor product) of two vectors creates a matrix
where each element (i, j) is the product of the i-th element of the first vector
and the j-th element of the second vector.
"""

import numpy as np


def demonstrate_outer_product():
    """Demonstrate outer product operations"""
    
    print("=" * 60)
    print("Outer Product Examples")
    print("=" * 60)
    
    # Example 1: Basic outer product
    print("\n1. Basic Outer Product of Two Vectors:")
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    outer = np.outer(a, b)
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Outer product (a ⊗ b):\n{outer}")
    print(f"Shape: {outer.shape}")
    print("Each element (i, j) = a[i] * b[j]")
    
    # Example 2: Using np.outer() vs manual calculation
    print("\n2. Outer Product Calculation:")
    a = np.array([2, 3])
    b = np.array([5, 7, 9])
    outer = np.outer(a, b)
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Outer product:\n{outer}")
    print("\nManual calculation:")
    print(f"  [2*5, 2*7, 2*9]   [10, 14, 18]")
    print(f"  [3*5, 3*7, 3*9] = [15, 21, 27]")
    
    # Example 3: Outer product properties
    print("\n3. Outer Product Properties:")
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    
    print(f"Vectors: a={a}, b={b}, c={c}")
    
    print(f"\nProperty 1: Distributive - (a + b) ⊗ c = a ⊗ c + b ⊗ c")
    left = np.outer(a + b, c)
    right = np.outer(a, c) + np.outer(b, c)
    print(f"  (a + b) ⊗ c:\n{left}")
    print(f"  a ⊗ c + b ⊗ c:\n{right}")
    print(f"  Equal? {np.array_equal(left, right)}")
    
    print(f"\nProperty 2: Scalar Multiplication - (ka) ⊗ b = k(a ⊗ b) = a ⊗ (kb)")
    k = 2
    expr1 = np.outer(k * a, b)
    expr2 = k * np.outer(a, b)
    expr3 = np.outer(a, k * b)
    print(f"  (ka) ⊗ b:\n{expr1}")
    print(f"  k(a ⊗ b):\n{expr2}")
    print(f"  a ⊗ (kb):\n{expr3}")
    print(f"  All equal? {np.array_equal(expr1, expr2) and np.array_equal(expr2, expr3)}")
    
    # Example 4: Relationship to matrix multiplication
    print("\n4. Relationship to Matrix Multiplication:")
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    outer = np.outer(a, b)
    print(f"Vector a (column):\n{a.reshape(-1, 1)}")
    print(f"Vector b (row):\n{b.reshape(1, -1)}")
    print(f"Outer product = column vector × row vector:")
    print(f"{outer}")
    print(f"\nNote: outer(a, b) = a.reshape(-1, 1) @ b.reshape(1, -1)")
    manual = a.reshape(-1, 1) @ b.reshape(1, -1)
    print(f"Verification: {np.array_equal(outer, manual)}")
    
    # Example 5: Outer product vs dot product
    print("\n5. Outer Product vs Dot Product:")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    outer = np.outer(a, b)
    dot = np.dot(a, b)
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Outer product (matrix):\n{outer}")
    print(f"Shape: {outer.shape}")
    print(f"\nDot product (scalar): {dot}")
    print("Note: Outer product creates a matrix, dot product creates a scalar")
    
    # Example 6: Rank-1 matrix
    print("\n6. Rank-1 Matrix Property:")
    a = np.array([2, 3])
    b = np.array([5, 7])
    outer = np.outer(a, b)
    print(f"Outer product:\n{outer}")
    print(f"Matrix rank: {np.linalg.matrix_rank(outer)}")
    print("Note: Outer product of two vectors always produces a rank-1 matrix")
    print("(All rows are multiples of each other, all columns are multiples of each other)")
    
    # Example 7: Applications - creating basis matrices
    print("\n7. Creating Basis Matrices:")
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    basis_matrix = np.outer(e1, e1)
    print(f"Standard basis vectors: e1={e1}, e2={e2}")
    print(f"Outer product e1 ⊗ e1:\n{basis_matrix}")
    print("This creates a basis matrix for the standard basis")


if __name__ == "__main__":
    demonstrate_outer_product()

