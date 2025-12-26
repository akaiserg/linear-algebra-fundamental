"""
Matrix Transpose Exercise - SOLUTIONS

This file contains the solutions to the transpose exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Transpose"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    # Simply use .T attribute or np.transpose()
    result = A.T
    # Alternative: result = np.transpose(A)
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Transpose Property"""
    A = np.array([[2, 5, 8],
                  [1, 3, 7],
                  [9, 4, 6]])
    
    # Compute (A^T)^T
    double_transpose = (A.T).T
    
    # Check if it equals A
    result = np.array_equal(double_transpose, A)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Transpose of Sum"""
    A = np.array([[1, 3],
                  [2, 4]])
    B = np.array([[5, 7],
                  [6, 8]])
    
    # Compute (A + B)^T
    left_side = (A + B).T
    
    # Compute A^T + B^T
    right_side = A.T + B.T
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Transpose of Product"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # Compute (AB)^T
    left_side = (A @ B).T  # or np.dot(A, B).T
    
    # Compute B^T A^T (note the order is reversed!)
    right_side = B.T @ A.T  # or np.dot(B.T, A.T)
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Convert Row Vector to Column Vector"""
    vector = np.array([10, 20, 30, 40])
    
    # Method 1: Reshape to column vector
    result = vector.reshape(-1, 1)
    
    # Method 2: Reshape then transpose (but reshape is simpler)
    # result = vector.reshape(4, 1)
    
    # Note: For 1D arrays, .T doesn't change the shape, so we need reshape
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Check if Matrix is Symmetric"""
    A = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    
    # A matrix is symmetric if A = A^T
    result = np.array_equal(A, A.T)
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:")
    print(exercise_1_solution())
    
    print("\nExercise 2 Result:", exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:")
    print(exercise_5_solution())
    print("Shape:", exercise_5_solution().shape)
    
    print("\nExercise 6 Result:", exercise_6_solution())

