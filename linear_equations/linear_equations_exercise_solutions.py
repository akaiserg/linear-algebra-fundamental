"""
Linear Equations Exercise - SOLUTIONS

This file contains the solutions to the linear equations exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Solve Simple Linear Equation"""
    # Solve 2x + 3 = 11
    # Rearrange: 2x = 11 - 3 = 8
    # x = 8 / 2 = 4
    
    result = (11 - 3) / 2
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Solve System of Two Linear Equations"""
    A = np.array([[2, 3],
                  [4, 5]])
    b = np.array([7, 13])
    
    # Use np.linalg.solve() to solve Ax = b
    result = np.linalg.solve(A, b)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Solve System of Three Linear Equations"""
    A = np.array([[1, 2, 3],
                  [2, 3, 1],
                  [3, 1, 2]])
    b = np.array([14, 11, 11])
    
    # Solve the system
    result = np.linalg.solve(A, b)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Verify Solution"""
    A = np.array([[1, 2],
                  [3, 4]])
    b = np.array([5, 11])
    x = np.array([1, 2])
    
    # Compute A @ x and check if it equals b
    Ax = A @ x
    result = np.allclose(Ax, b)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Check if Matrix is Invertible"""
    A = np.array([[1, 2],
                  [2, 4]])
    
    # Check if determinant is non-zero (invertible)
    det_A = np.linalg.det(A)
    result = not np.isclose(det_A, 0.0)
    
    # Alternative: Check if matrix has full rank
    # result = np.linalg.matrix_rank(A) == A.shape[0]
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Solve Using Matrix Inverse"""
    A = np.array([[2, 1],
                  [1, 3]])
    b = np.array([5, 7])
    
    # Solve using x = A^(-1) @ b
    A_inv = np.linalg.inv(A)
    result = A_inv @ b
    
    # Note: np.linalg.solve() is preferred over inv() for numerical stability
    # result = np.linalg.solve(A, b)  # Better approach
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Overdetermined System (Least Squares)"""
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    b = np.array([3, 7, 11])
    
    # Method 1: Using np.linalg.lstsq()
    result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Method 2: Using pseudo-inverse
    # result = np.linalg.pinv(A) @ b
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Underdetermined System"""
    A = np.array([[1, 2, 3]])
    b = np.array([6])
    
    # Use pseudo-inverse to find a solution
    result = np.linalg.pinv(A) @ b
    
    # Alternative: Using lstsq
    # result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Homogeneous System"""
    A = np.array([[1, 2],
                  [2, 4]])
    b = np.array([0, 0])
    
    # Non-trivial solutions exist if det(A) = 0 (A is singular)
    det_A = np.linalg.det(A)
    result = np.isclose(det_A, 0.0)
    
    # Alternative: Check if null space is non-zero
    # null_space = scipy.linalg.null_space(A)
    # result = null_space.shape[1] > 0
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Vector Form of Linear Equation"""
    a = np.array([3, 4])
    b = 10
    
    # Find a solution [x, y] such that a · [x, y] = b
    # There are infinitely many solutions. One approach:
    # Pick x = 0, then 4y = 10, so y = 2.5
    # Or pick y = 1, then 3x + 4 = 10, so x = 2
    
    # Method 1: Pick x = 0
    # result = np.array([0, b / a[1]])
    
    # Method 2: Pick y = 1
    result = np.array([(b - a[1]) / a[0], 1])
    
    # Method 3: Use pseudo-inverse (finds minimum norm solution)
    # a_2d = a.reshape(1, -1)  # Make it a row vector
    # result = np.linalg.pinv(a_2d) @ np.array([b])
    
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

