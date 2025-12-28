"""
Linear Combination, Basis, Span, and Rank Example

These are fundamental concepts in linear algebra that describe how vectors
relate to each other and the spaces they create.
"""

import numpy as np


def demonstrate_concepts():
    """Demonstrate linear combination, basis, span, and rank"""
    
    print("=" * 60)
    print("Linear Combination, Basis, Span, and Rank Examples")
    print("=" * 60)
    
    # Example 1: Linear Combination
    print("\n1. Linear Combination:")
    print("A linear combination of vectors v₁, v₂, ..., vₙ is:")
    print("  c₁v₁ + c₂v₂ + ... + cₙvₙ")
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    v3 = np.array([1, 1])
    coefficients = [2, 3, -1]
    combination = coefficients[0]*v1 + coefficients[1]*v2 + coefficients[2]*v3
    print(f"\nExample: {coefficients[0]}*{v1} + {coefficients[1]}*{v2} + {coefficients[2]}*{v3}")
    print(f"Result: {combination}")
    
    # Example 2: Linear Independence
    print("\n2. Linear Independence:")
    print("Vectors are linearly independent if no vector can be written")
    print("as a linear combination of the others.")
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    A = np.column_stack([v1, v2])
    rank = np.linalg.matrix_rank(A)
    print(f"\nVectors: v₁={v1}, v₂={v2}")
    print(f"Matrix with vectors as columns:\n{A}")
    print(f"Rank: {rank}, Number of vectors: {A.shape[1]}")
    print(f"Linearly independent? {rank == A.shape[1]}")
    
    # Example 3: Linear Dependence
    print("\n3. Linear Dependence:")
    v1 = np.array([1, 2])
    v2 = np.array([2, 4])  # v₂ = 2*v₁
    A = np.column_stack([v1, v2])
    rank = np.linalg.matrix_rank(A)
    print(f"Vectors: v₁={v1}, v₂={v2}")
    print(f"Note: v₂ = 2*v₁ (linearly dependent)")
    print(f"Matrix:\n{A}")
    print(f"Rank: {rank}, Number of vectors: {A.shape[1]}")
    print(f"Linearly dependent? {rank < A.shape[1]}")
    
    # Example 4: Standard Basis
    print("\n4. Standard Basis:")
    print("The standard basis for ℝ² consists of:")
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    print(f"  e₁ = {e1}")
    print(f"  e₂ = {e2}")
    print("Any vector in ℝ² can be written as a linear combination:")
    v = np.array([5, 7])
    print(f"  {v} = {v[0]}*{e1} + {v[1]}*{e2}")
    
    # Example 5: Span
    print("\n5. Span:")
    print("The span of vectors {v₁, v₂, ..., vₙ} is the set of all")
    print("possible linear combinations of those vectors.")
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    A = np.column_stack([v1, v2])
    print(f"\nSpan{{v₁, v₂}} where v₁={v1}, v₂={v2}")
    print(f"These vectors span ℝ² (all 2D vectors)")
    w = np.array([3, 4])
    coefficients = np.linalg.solve(A, w)
    print(f"Vector {w} is in span{{v₁, v₂}}")
    print(f"  {w} = {coefficients[0]:.1f}*{v1} + {coefficients[1]:.1f}*{v2}")
    
    # Example 6: Rank
    print("\n6. Matrix Rank:")
    print("The rank of a matrix is:")
    print("  - The dimension of the column space (span of columns)")
    print("  - The number of linearly independent columns")
    print("  - The number of linearly independent rows")
    A = np.array([[1, 2, 3],
                  [0, 1, 2],
                  [0, 0, 1]])
    rank = np.linalg.matrix_rank(A)
    print(f"\nMatrix A:\n{A}")
    print(f"Rank: {rank}")
    print(f"All {A.shape[1]} columns are linearly independent")
    
    # Example 7: Rank with Dependent Columns
    print("\n7. Rank with Linearly Dependent Columns:")
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    rank = np.linalg.matrix_rank(A)
    print(f"Matrix A:\n{A}")
    print(f"Note: Column 2 = 2*Column 1, Column 3 = 3*Column 1")
    print(f"Rank: {rank}")
    print(f"Only {rank} column is linearly independent")
    
    # Example 8: Basis
    print("\n8. Basis:")
    print("A basis for a vector space is a set of vectors that:")
    print("  1. Are linearly independent")
    print("  2. Span the entire space")
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    A = np.column_stack([v1, v2])
    rank = np.linalg.matrix_rank(A)
    dimension = 2
    print(f"\nVectors: v₁={v1}, v₂={v2}")
    print(f"Form a basis for ℝ²? {rank == dimension and A.shape[1] == dimension}")
    print("These are the standard basis vectors for ℝ²")
    
    # Example 9: Finding Coefficients
    print("\n9. Finding Coefficients for Linear Combination:")
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    w = np.array([7, 10])
    A = np.column_stack([v1, v2])
    coefficients = np.linalg.solve(A, w)
    print(f"Given: v₁={v1}, v₂={v2}, w={w}")
    print(f"Find: w = c₁*v₁ + c₂*v₂")
    print(f"Solution: c₁={coefficients[0]:.1f}, c₂={coefficients[1]:.1f}")
    print(f"Verification: {coefficients[0]:.1f}*{v1} + {coefficients[1]:.1f}*{v2} = {coefficients[0]*v1 + coefficients[1]*v2}")
    
    # Example 10:   
    print("\n10. Dimension of Span:")
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])
    A = np.column_stack([v1, v2, v3])
    rank = np.linalg.matrix_rank(A)
    print(f"Vectors: v₁={v1}, v₂={v2}, v₃={v3}")
    print(f"Note: v₃ = v₁ + v₂ (linearly dependent)")
    print(f"Matrix:\n{A}")
    print(f"Rank: {rank}")
    print(f"Dimension of span{{v₁, v₂, v₃}} = {rank}")


if __name__ == "__main__":
    demonstrate_concepts()

