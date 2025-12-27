"""
Matrix Determinant Exercise - SOLUTIONS

This file contains the solutions to the determinant exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Determinant (2x2)"""
    A = np.array([[3, 1],
                  [2, 4]])
    
    # Use np.linalg.det() to calculate determinant
    result = np.linalg.det(A)
    
    # For 2x2: det = a*d - b*c = 3*4 - 1*2 = 12 - 2 = 10
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Determinant of 3x3 Matrix"""
    A = np.array([[1, 2, 3],
                  [0, 1, 4],
                  [5, 6, 0]])
    
    # Use np.linalg.det() for any square matrix
    result = np.linalg.det(A)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Determinant Property - Scalar Multiplication"""
    A = np.array([[1, 3],
                  [2, 4]])
    k = 2
    
    # Compute det(kA)
    left_side = np.linalg.det(k * A)
    
    # Compute k^n * det(A) where n=2 (2x2 matrix)
    n = A.shape[0]  # Get dimension
    right_side = (k ** n) * np.linalg.det(A)
    
    # Check if they are equal (using np.isclose for floating point comparison)
    result = np.isclose(left_side, right_side)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Determinant Property - Product"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # Compute det(AB)
    left_side = np.linalg.det(A @ B)  # or np.dot(A, B)
    
    # Compute det(A) * det(B)
    right_side = np.linalg.det(A) * np.linalg.det(B)
    
    # Check if they are equal
    result = np.isclose(left_side, right_side)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Determinant Property - Transpose"""
    A = np.array([[2, 5, 1],
                  [3, 1, 4],
                  [1, 2, 3]])
    
    # Compute det(A)
    det_A = np.linalg.det(A)
    
    # Compute det(A^T)
    det_A_T = np.linalg.det(A.T)
    
    # Check if they are equal
    result = np.isclose(det_A, det_A_T)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Determinant of Identity Matrix"""
    # Create a 3x3 identity matrix
    I = np.eye(3)  # or np.identity(3)
    
    # Calculate determinant
    det_I = np.linalg.det(I)
    
    # Check if it equals 1
    result = np.isclose(det_I, 1.0)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Singular Matrix (Zero Determinant)"""
    A = np.array([[1, 2],
                  [2, 4]])  # Second row is 2× first row (linearly dependent)
    
    # Calculate determinant
    det_A = np.linalg.det(A)
    
    # Check if it's singular (determinant = 0)
    result = np.isclose(det_A, 0.0)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Determinant After Row Operations"""
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    # Calculate original determinant
    det_A = np.linalg.det(A)
    
    # Swap two rows (e.g., swap row 0 and row 1)
    A_swapped = A.copy()
    A_swapped[[0, 1]] = A_swapped[[1, 0]]  # Swap rows 0 and 1
    
    # Calculate determinant after swap
    det_A_swapped = np.linalg.det(A_swapped)
    
    # Check if det_A_swapped = -1 * det_A
    result = np.isclose(det_A_swapped, -1 * det_A)
    
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

