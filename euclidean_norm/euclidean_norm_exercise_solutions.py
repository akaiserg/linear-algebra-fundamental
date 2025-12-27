"""
Euclidean Norm Exercise - SOLUTIONS

This file contains the solutions to the Euclidean norm exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Euclidean Norm"""
    v = np.array([3, 4])
    
    # Method 1: Using np.linalg.norm()
    result = np.linalg.norm(v)
    
    # Method 2: Manual calculation
    # result = np.sqrt(np.sum(v**2))
    
    # Method 3: Using dot product
    # result = np.sqrt(np.dot(v, v))
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Euclidean Norm Using np.linalg.norm()"""
    v = np.array([1, 2, 2])
    
    # Use np.linalg.norm() - default is Euclidean norm
    result = np.linalg.norm(v)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Euclidean Norm Using Dot Product"""
    v = np.array([5, 12])
    
    # Norm = sqrt(v · v)
    result = np.sqrt(np.dot(v, v))
    
    # Alternative: np.sqrt(v @ v)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Unit Vector (Normalized Vector)"""
    v = np.array([3, 4])
    
    # Normalize by dividing by norm
    norm_v = np.linalg.norm(v)
    result = v / norm_v
    
    # Alternative one-liner:
    # result = v / np.linalg.norm(v)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Verify Unit Vector Has Norm 1"""
    v = np.array([1, 1, 1])
    
    # Normalize the vector
    v_normalized = v / np.linalg.norm(v)
    
    # Check if norm equals 1
    norm_normalized = np.linalg.norm(v_normalized)
    result = np.isclose(norm_normalized, 1.0)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Distance Between Two Points"""
    a = np.array([1, 2])
    b = np.array([4, 6])
    
    # Distance = ||a - b||
    result = np.linalg.norm(a - b)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Norm Property - Scalar Multiplication"""
    v = np.array([3, 4])
    k = -2
    
    # Compute ||kv||
    left_side = np.linalg.norm(k * v)
    
    # Compute |k| × ||v||
    right_side = abs(k) * np.linalg.norm(v)
    
    # Check if they are equal
    result = np.isclose(left_side, right_side)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Triangle Inequality"""
    a = np.array([1, 2])
    b = np.array([3, 4])
    
    # Compute ||a + b||
    left_side = np.linalg.norm(a + b)
    
    # Compute ||a|| + ||b||
    right_side = np.linalg.norm(a) + np.linalg.norm(b)
    
    # Check if ||a + b|| ≤ ||a|| + ||b||
    result = left_side <= right_side
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Zero Vector Has Zero Norm"""
    v = np.array([0, 0, 0])
    
    # Compute norm
    norm_v = np.linalg.norm(v)
    
    # Check if norm is 0
    result = np.isclose(norm_v, 0.0)
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Norm of Matrix (Frobenius Norm)"""
    A = np.array([[1, 2],
                  [3, 4]])
    
    # Frobenius norm - default behavior of np.linalg.norm() for matrices
    result = np.linalg.norm(A)
    
    # Alternative: Treat matrix as vector and compute Euclidean norm
    # result = np.linalg.norm(A.flatten())
    
    # Manual calculation:
    # result = np.sqrt(np.sum(A**2))
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:", exercise_1_solution())
    
    print("\nExercise 2 Result:", exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

