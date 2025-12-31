"""
Outer Product Exercise - SOLUTIONS

This file contains the solutions to the outer product exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Outer Product"""
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    
    # Use np.outer() function
    result = np.outer(a, b)
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Outer Product Using np.outer()"""
    a = np.array([2, 3])
    b = np.array([5, 7, 9])
    
    # Use np.outer() function
    result = np.outer(a, b)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Outer Product Using Matrix Multiplication"""
    a = np.array([1, 2])
    b = np.array([3, 4, 5])
    
    # Outer product = column vector × row vector
    result = a.reshape(-1, 1) @ b.reshape(1, -1)
    
    # Alternative: np.outer(a, b)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Outer Product Property - Distributive"""
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    
    # Compute (a + b) ⊗ c
    left_side = np.outer(a + b, c)
    
    # Compute a ⊗ c + b ⊗ c
    right_side = np.outer(a, c) + np.outer(b, c)
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Outer Product Property - Scalar Multiplication"""
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    k = 3
    
    # Compute (ka) ⊗ b
    expr1 = np.outer(k * a, b)
    
    # Compute k(a ⊗ b)
    expr2 = k * np.outer(a, b)
    
    # Compute a ⊗ (kb)
    expr3 = np.outer(a, k * b)
    
    # Check if all three are equal
    result = (np.array_equal(expr1, expr2) and 
             np.array_equal(expr2, expr3) and 
             np.array_equal(expr1, expr3))
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Outer Product vs Dot Product"""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # Calculate outer product (matrix)
    outer_product = np.outer(a, b)
    
    # Calculate dot product (scalar)
    dot_product = np.dot(a, b)
    
    result = (outer_product, dot_product)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Rank of Outer Product Matrix"""
    a = np.array([2, 3, 4])
    b = np.array([5, 6])
    
    # Calculate outer product
    outer = np.outer(a, b)
    
    # Calculate rank
    result = np.linalg.matrix_rank(outer)
    
    # Note: Outer product of two non-zero vectors always has rank 1
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Outer Product Dimensions"""
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7])
    
    # Calculate outer product and get its shape
    outer = np.outer(a, b)
    result = outer.shape
    
    # Alternative: (len(a), len(b))
    # result = (len(a), len(b))
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Outer Product with Standard Basis Vectors"""
    # Create standard basis vector e1
    e1 = np.array([1, 0])
    
    # Compute e1 ⊗ e1
    result = np.outer(e1, e1)
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Outer Product Transpose Property"""
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    
    # Compute (a ⊗ b)^T
    left_side = np.outer(a, b).T
    
    # Compute b ⊗ a
    right_side = np.outer(b, a)
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:")
    print(exercise_1_solution())
    
    print("\nExercise 2 Result:")
    print(exercise_2_solution())
    
    print("\nExercise 3 Result:")
    print(exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:")
    outer, dot = exercise_6_solution()
    print(f"  Outer product:\n{outer}")
    print(f"  Dot product: {dot}")
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:")
    print(exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

