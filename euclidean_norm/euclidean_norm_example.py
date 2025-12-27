"""
Euclidean Norm Example

The Euclidean norm (also called L2 norm) measures the length or magnitude of a vector.
It's the square root of the sum of squares of the vector components.
"""

import numpy as np


def demonstrate_euclidean_norm():
    """Demonstrate Euclidean norm operations"""
    
    print("=" * 60)
    print("Euclidean Norm Examples")
    print("=" * 60)
    
    # Example 1: Basic Euclidean norm
    print("\n1. Basic Euclidean Norm:")
    v = np.array([3, 4])
    norm_v = np.linalg.norm(v)
    print(f"Vector v: {v}")
    print(f"Euclidean norm: {norm_v}")
    print(f"Calculation: sqrt(3² + 4²) = sqrt({3**2} + {4**2}) = sqrt({3**2 + 4**2}) = {norm_v}")
    
    # Example 2: Different ways to compute norm
    print("\n2. Different Ways to Compute Euclidean Norm:")
    v = np.array([1, 2, 2])
    print(f"Vector v: {v}")
    print(f"  np.linalg.norm(v): {np.linalg.norm(v)}")
    print(f"  sqrt(sum(v²)): {np.sqrt(np.sum(v**2))}")
    print(f"  sqrt(v · v): {np.sqrt(np.dot(v, v))}")
    print("  All methods give the same result!")
    
    # Example 3: Unit vectors (normalized vectors)
    print("\n3. Unit Vectors (Normalized Vectors):")
    v = np.array([3, 4])
    v_unit = v / np.linalg.norm(v)
    print(f"Original vector v: {v}, norm: {np.linalg.norm(v)}")
    print(f"Normalized vector: {v_unit}, norm: {np.linalg.norm(v_unit)}")
    print("A unit vector has norm = 1")
    
    # Example 4: Distance between points
    print("\n4. Distance Between Two Points:")
    a = np.array([1, 2])
    b = np.array([4, 6])
    distance = np.linalg.norm(a - b)
    print(f"Point a: {a}")
    print(f"Point b: {b}")
    print(f"Distance: ||a - b|| = {distance}")
    print(f"Calculation: ||[{a[0]}, {a[1]}] - [{b[0]}, {b[1]}]|| = ||[{a[0]-b[0]}, {a[1]-b[1]}]|| = {distance}")
    
    # Example 5: Properties of norm
    print("\n5. Properties of Euclidean Norm:")
    v = np.array([3, 4])
    k = 2
    
    print(f"Vector v: {v}, scalar k: {k}")
    print(f"\nProperty 1: ||kv|| = |k| × ||v||")
    print(f"  ||kv|| = ||{k} × {v}|| = ||{k*v}|| = {np.linalg.norm(k*v)}")
    print(f"  |k| × ||v|| = {abs(k)} × {np.linalg.norm(v)} = {abs(k) * np.linalg.norm(v)}")
    print(f"  Equal? {np.isclose(np.linalg.norm(k*v), abs(k) * np.linalg.norm(v))}")
    
    # Example 6: Triangle inequality
    print("\n6. Triangle Inequality:")
    a = np.array([1, 2])
    b = np.array([3, 4])
    left = np.linalg.norm(a + b)
    right = np.linalg.norm(a) + np.linalg.norm(b)
    print(f"Vectors: a={a}, b={b}")
    print(f"  ||a + b|| = ||{a+b}|| = {left}")
    print(f"  ||a|| + ||b|| = {np.linalg.norm(a)} + {np.linalg.norm(b)} = {right}")
    print(f"  Triangle inequality: ||a + b|| ≤ ||a|| + ||b||")
    print(f"  {left} ≤ {right}? {left <= right}")
    
    # Example 7: Zero vector
    print("\n7. Zero Vector:")
    v_zero = np.array([0, 0, 0])
    norm_zero = np.linalg.norm(v_zero)
    print(f"Zero vector: {v_zero}")
    print(f"Norm: {norm_zero}")
    print("The zero vector has norm 0")
    
    # Example 8: Matrix norm (Frobenius norm)
    print("\n8. Matrix Norm (Frobenius Norm):")
    A = np.array([[1, 2],
                  [3, 4]])
    frobenius_norm = np.linalg.norm(A)
    print(f"Matrix A:\n{A}")
    print(f"Frobenius norm: {frobenius_norm:.4f}")
    print(f"Calculation: sqrt(1² + 2² + 3² + 4²) = sqrt({1**2 + 2**2 + 3**2 + 4**2}) = {frobenius_norm:.4f}")
    print("The Frobenius norm treats the matrix as a vector and computes its Euclidean norm")
    
    # Example 9: Geometric interpretation
    print("\n9. Geometric Interpretation:")
    print("The Euclidean norm represents the length of the vector from origin to the point.")
    print("In 2D: ||[x, y]|| = sqrt(x² + y²) is the distance from (0,0) to (x,y)")
    print("In 3D: ||[x, y, z]|| = sqrt(x² + y² + z²) is the distance from (0,0,0) to (x,y,z)")
    v_2d = np.array([3, 4])
    print(f"\nExample: Vector {v_2d} has length {np.linalg.norm(v_2d)}")
    print("This is the distance from origin (0,0) to point (3,4)")


if __name__ == "__main__":
    demonstrate_euclidean_norm()

