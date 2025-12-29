"""
Identity Matrix Exercise - SOLUTIONS

This file contains the solutions to the identity matrix exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Create Identity Matrix"""
    # Method 1: Using np.eye()
    result = np.eye(3)
    
    # Method 2: Using np.identity()
    # result = np.identity(3)
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Identity as Multiplicative Identity"""
    A = np.array([[2, 5, 1],
                  [3, 1, 4],
                  [1, 2, 3]])
    
    # Create identity matrix of same size as A
    I = np.eye(A.shape[0])
    
    # Compute A @ I
    product = A @ I
    
    # Check if A @ I equals A
    result = np.array_equal(product, A)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Identity with Vectors"""
    v = np.array([1, 2, 3, 4])
    
    # Create identity matrix matching vector dimension
    I = np.eye(len(v))
    
    # Compute I @ v
    product = I @ v
    
    # Check if I @ v equals v
    result = np.array_equal(product, v)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Identity Transpose Property"""
    # Create 4x4 identity matrix
    I = np.eye(4)
    
    # Check if I^T equals I
    result = np.array_equal(I, I.T)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Identity Determinant"""
    # Create 5x5 identity matrix
    I = np.eye(5)
    
    # Calculate determinant
    det_I = np.linalg.det(I)
    
    # Check if determinant equals 1
    result = np.isclose(det_I, 1.0)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Identity Power Property"""
    # Create 3x3 identity matrix
    I = np.eye(3)
    
    # Compute I^2
    I_squared = I @ I
    
    # Check if I^2 equals I
    result = np.array_equal(I, I_squared)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Identity in Matrix Addition"""
    A = np.array([[1, 2],
                  [3, 4]])
    
    # Create identity matrix
    I = np.eye(2)
    
    # Create zero matrix
    zero = np.zeros((2, 2))
    
    # Compute A @ I + 0
    left_side = A @ I + zero
    
    # Compute A @ I
    right_side = A @ I
    
    # Check if they are equal
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Identity and Inverse Relationship"""
    A = np.array([[2, 1],
                  [1, 3]])
    
    # Compute inverse of A
    A_inv = np.linalg.inv(A)
    
    # Create identity matrix
    I = np.eye(2)
    
    # Compute A @ A^(-1)
    product = A @ A_inv
    
    # Check if A @ A^(-1) equals I (using allclose for floating point comparison)
    result = np.allclose(product, I)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Scalar Multiple of Identity"""
    k = 3
    
    # Method 1: Multiply identity by scalar
    I = np.eye(3)
    result = k * I
    
    # Method 2: Direct creation
    # result = np.eye(3) * 3
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Identity Matrix Size"""
    # Create identity matrices of different sizes
    I2 = np.eye(2)
    I3 = np.eye(3)
    I4 = np.eye(4)
    
    # Return tuple of their shapes
    result = (I2.shape, I3.shape, I4.shape)
    
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
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:")
    print(exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

