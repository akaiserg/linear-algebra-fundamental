"""
Matrix Multiplication Exercise - SOLUTIONS

This file contains the solutions to the matrix multiplication exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic 2x2 Matrix Multiplication"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # Use @ operator for matrix multiplication
    result = A @ B
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Matrix Multiplication Using np.dot()"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    # Use np.dot() function
    result = np.dot(A, B)
    
    # Alternative: result = A @ B
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Matrix-Vector Multiplication"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    v = np.array([10, 11, 12])
    
    # Matrix-vector multiplication
    result = A @ v  # or np.dot(A, v)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Matrix Multiplication Property - Associative"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    C = np.array([[9, 10],
                  [11, 12]])
    
    # Compute (A @ B) @ C
    left_side = (A @ B) @ C
    
    # Compute A @ (B @ C)
    right_side = A @ (B @ C)
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Matrix Multiplication Property - Distributive"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    C = np.array([[9, 10],
                  [11, 12]])
    
    # Compute A @ (B + C)
    left_side = A @ (B + C)
    
    # Compute A @ B + A @ C
    right_side = A @ B + A @ C
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Matrix Multiplication Property - Scalar Multiplication"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    k = 2
    
    # Compute k(A @ B)
    expr1 = k * (A @ B)
    
    # Compute (kA) @ B
    expr2 = (k * A) @ B
    
    # Compute A @ (kB)
    expr3 = A @ (k * B)
    
    # Check if all three are equal
    result = (np.array_equal(expr1, expr2) and 
              np.array_equal(expr2, expr3) and 
              np.array_equal(expr1, expr3))
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Identity Matrix Multiplication"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    # Create identity matrices
    I3 = np.eye(3)  # 3×3 identity (for right multiplication)
    I2 = np.eye(2)  # 2×2 identity (for left multiplication)
    
    # Verify A @ I = A
    right_mult = A @ I3
    
    # Verify I @ A = A
    left_mult = I2 @ A
    
    # Check if both properties hold
    result = np.array_equal(A, right_mult) and np.array_equal(A, left_mult)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Matrix Multiplication is NOT Commutative"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # Compute A @ B
    AB = A @ B
    
    # Compute B @ A
    BA = B @ A
    
    # Check if they are different
    result = not np.array_equal(AB, BA)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Transpose of Matrix Product"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # Compute (A @ B).T
    left_side = (A @ B).T
    
    # Compute B.T @ A.T
    right_side = B.T @ A.T
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Matrix Multiplication with Different Dimensions"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    # Calculate A @ B
    result = A @ B
    
    # Get the shape
    shape = result.shape
    
    return result, shape


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:")
    print(exercise_1_solution())
    
    print("\nExercise 2 Result:")
    print(exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:")
    result, shape = exercise_10_solution()
    print(f"Shape: {shape}")
    print(f"Result:\n{result}")

