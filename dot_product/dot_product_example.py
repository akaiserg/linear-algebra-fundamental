"""
Dot Product Example

The dot product (also called scalar product or inner product) is an operation
that takes two vectors and returns a scalar value.
"""

import numpy as np


def demonstrate_dot_product():
    """Demonstrate dot product operations"""
    
    print("=" * 60)
    print("Dot Product Examples")
    print("=" * 60)
    
    # Example 1: Basic dot product
    print("\n1. Basic Dot Product of Two Vectors:")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    dot_product = np.dot(a, b)
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Dot product: {dot_product}")
    print(f"Calculation: 1*4 + 2*5 + 3*6 = {1*4} + {2*5} + {3*6} = {dot_product}")
    
    # Example 2: Different ways to compute dot product
    print("\n2. Different Ways to Compute Dot Product:")
    a = np.array([2, 3])
    b = np.array([5, 7])
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"  np.dot(a, b): {np.dot(a, b)}")
    print(f"  a @ b: {a @ b}")
    print(f"  np.sum(a * b): {np.sum(a * b)}")
    print("  All methods give the same result!")
    
    # Example 3: Properties of dot product
    print("\n3. Properties of Dot Product:")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    k = 2
    
    print(f"Vectors: a={a}, b={b}, c={c}, k={k}")
    
    print(f"\nProperty 1: Commutative - a · b = b · a")
    print(f"  a · b = {np.dot(a, b)}")
    print(f"  b · a = {np.dot(b, a)}")
    print(f"  Equal? {np.isclose(np.dot(a, b), np.dot(b, a))}")
    
    print(f"\nProperty 2: Distributive - a · (b + c) = a · b + a · c")
    left = np.dot(a, b + c)
    right = np.dot(a, b) + np.dot(a, c)
    print(f"  a · (b + c) = {left}")
    print(f"  a · b + a · c = {right}")
    print(f"  Equal? {np.isclose(left, right)}")
    
    print(f"\nProperty 3: Scalar Multiplication - (ka) · b = k(a · b)")
    left = np.dot(k * a, b)
    right = k * np.dot(a, b)
    print(f"  (ka) · b = {left}")
    print(f"  k(a · b) = {right}")
    print(f"  Equal? {np.isclose(left, right)}")
    
    # Example 4: Orthogonal vectors
    print("\n4. Orthogonal Vectors (Perpendicular):")
    a = np.array([1, 0])
    b = np.array([0, 1])
    dot_product = np.dot(a, b)
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Dot product: {dot_product}")
    print("Two vectors are orthogonal if their dot product is 0")
    print(f"Are they orthogonal? {np.isclose(dot_product, 0)}")
    
    # Example 5: Matrix-vector dot product
    print("\n5. Matrix-Vector Dot Product:")
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    v = np.array([7, 8])
    result = A @ v
    print(f"Matrix A (3x2):\n{A}")
    print(f"Vector v: {v}")
    print(f"Result (A @ v): {result}")
    print("Each row of A is dotted with v")
    
    # Example 6: Matrix-matrix dot product
    print("\n6. Matrix-Matrix Dot Product:")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    result = A @ B
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Result (A @ B):\n{result}")
    print("Matrix multiplication is computed using dot products of rows and columns")
    
    # Example 7: Magnitude (length) using dot product
    print("\n7. Vector Magnitude (Length) Using Dot Product:")
    v = np.array([3, 4])
    magnitude = np.sqrt(np.dot(v, v))
    print(f"Vector v: {v}")
    print(f"Magnitude = sqrt(v · v) = sqrt({np.dot(v, v)}) = {magnitude}")
    print(f"Using np.linalg.norm(): {np.linalg.norm(v)}")
    print("Both methods give the same result!")
    
    # Example 8: Geometric interpretation
    print("\n8. Geometric Interpretation:")
    print("The dot product a · b can be written as:")
    print("  a · b = ||a|| ||b|| cos(θ)")
    print("where ||a|| and ||b|| are magnitudes, and θ is the angle between vectors")
    a = np.array([1, 0])
    b = np.array([np.sqrt(2)/2, np.sqrt(2)/2])  # Unit vector at 45 degrees
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    
    # Clamp cos_theta to [-1, 1] to avoid floating point precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    print(f"\nExample:")
    print(f"  a = {a}, ||a|| = {magnitude_a}")
    print(f"  b = {b}, ||b|| = {magnitude_b}")
    print(f"  a · b = {dot_product}")
    print(f"  cos(θ) = {cos_theta:.4f}")
    print(f"  θ = {theta_deg:.1f}°")


if __name__ == "__main__":
    demonstrate_dot_product()

