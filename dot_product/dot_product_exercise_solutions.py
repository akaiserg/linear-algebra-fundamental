"""
Dot Product Exercise - SOLUTIONS

This file contains the solutions to the dot product exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Dot Product of Vectors"""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # Method 1: Using np.dot()
    result = np.dot(a, b)
    
    # Method 2: Using @ operator
    # result = a @ b
    
    # Method 3: Manual calculation
    # result = np.sum(a * b)
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Dot Product Using np.dot()"""
    a = np.array([2, 3])
    b = np.array([5, 7])
    
    # Use np.dot() function
    result = np.dot(a, b)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Dot Product Using @ Operator"""
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    
    # Use @ operator (matrix multiplication)
    result = a @ b
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Dot Product Property - Commutative"""
    a = np.array([1, 3, 5])
    b = np.array([2, 4, 6])
    
    # Compute a · b
    left_side = np.dot(a, b)
    
    # Compute b · a
    right_side = np.dot(b, a)
    
    # Check if they are equal
    result = np.isclose(left_side, right_side)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Dot Product Property - Distributive"""
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    
    # Compute a · (b + c)
    left_side = np.dot(a, b + c)
    
    # Compute a · b + a · c
    right_side = np.dot(a, b) + np.dot(a, c)
    
    # Check if they are equal
    result = np.isclose(left_side, right_side)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Dot Product Property - Scalar Multiplication"""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    k = 3
    
    # Compute (ka) · b
    expr1 = np.dot(k * a, b)
    
    # Compute k(a · b)
    expr2 = k * np.dot(a, b)
    
    # Compute a · (kb)
    expr3 = np.dot(a, k * b)
    
    # Check if all three are equal
    result = (np.isclose(expr1, expr2) and 
              np.isclose(expr2, expr3) and 
              np.isclose(expr1, expr3))
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Orthogonal Vectors"""
    a = np.array([1, 0])
    b = np.array([0, 1])
    
    # Compute dot product
    dot_product = np.dot(a, b)
    
    # Check if dot product is 0 (orthogonal)
    result = np.isclose(dot_product, 0.0)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Matrix-Vector Dot Product"""
    A = np.array([[1, 2],
                  [3, 4]])
    v = np.array([5, 6])
    
    # Matrix-vector multiplication
    result = A @ v  # or np.dot(A, v)
    
    # This computes: [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Matrix-Matrix Dot Product"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # Matrix multiplication
    result = A @ B  # or np.dot(A, B)
    
    # This computes:
    # [[1*5 + 2*7, 1*6 + 2*8],
    #  [3*5 + 4*7, 3*6 + 4*8]]
    # = [[19, 22],
    #    [43, 50]]
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Magnitude Using Dot Product"""
    v = np.array([3, 4])
    
    # Magnitude = sqrt(v · v)
    result = np.sqrt(np.dot(v, v))
    
    # Alternative: np.linalg.norm(v)
    # result = np.linalg.norm(v)
    
    return result


def exercise_11_solution():
    """Solution to Exercise 11: Vectors Pointing in Same Direction (θ = 0°)"""
    a = np.array([3, 4])
    b = np.array([6, 8])  # b = 2a, so same direction
    
    # Compute a · b
    dot_product = np.dot(a, b)
    
    # Compute ||a|| ||b||
    magnitude_product = np.linalg.norm(a) * np.linalg.norm(b)
    
    # When θ = 0°, cos(0°) = 1, so a · b = ||a|| ||b||
    result = np.isclose(dot_product, magnitude_product)
    
    return result


def exercise_12_solution():
    """Solution to Exercise 12: Vectors Pointing in Opposite Directions (θ = 180°)"""
    a = np.array([2, 3])
    b = np.array([-4, -6])  # b = -2a, so opposite direction
    
    # Compute a · b
    dot_product = np.dot(a, b)
    
    # Compute -||a|| ||b||
    negative_magnitude_product = -np.linalg.norm(a) * np.linalg.norm(b)
    
    # When θ = 180°, cos(180°) = -1, so a · b = -||a|| ||b||
    result = np.isclose(dot_product, negative_magnitude_product)
    
    return result


def exercise_13_solution():
    """Solution to Exercise 13: Calculate Angle Between Vectors"""
    a = np.array([1, 0])
    b = np.array([1, 1])
    
    # Using the formula: a · b = ||a|| ||b|| cos(θ)
    # Solve for cos(θ): cos(θ) = (a · b) / (||a|| ||b||)
    dot_product = np.dot(a, b)
    magnitude_product = np.linalg.norm(a) * np.linalg.norm(b)
    cos_theta = dot_product / magnitude_product
    
    # Clamp cos_theta to [-1, 1] to avoid floating point precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    theta_rad = np.arccos(cos_theta)
    result = np.degrees(theta_rad)
    
    # Alternative one-liner:
    # result = np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    
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
    
    print("\nExercise 9 Result:")
    print(exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())
    
    print("\nExercise 11 Result:", exercise_11_solution())
    
    print("\nExercise 12 Result:", exercise_12_solution())
    
    print("\nExercise 13 Result:", exercise_13_solution())

