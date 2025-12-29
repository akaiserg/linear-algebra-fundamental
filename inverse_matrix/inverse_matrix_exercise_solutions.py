"""
Matrix Inverse Exercise - SOLUTIONS

This file contains the solutions to the matrix inverse exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Matrix Inverse (2x2)"""
    A = np.array([[3, 1],
                  [2, 4]])
    
    # Use np.linalg.inv() to calculate inverse
    result = np.linalg.inv(A)
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Inverse of 3x3 Matrix"""
    A = np.array([[1, 2, 0],
                  [0, 1, 1],
                  [1, 0, 1]])
    
    # Calculate inverse
    result = np.linalg.inv(A)
    
    # Verification: A @ result should equal identity matrix
    # verification = A @ result  # Should be I
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Verify Inverse Property"""
    A = np.array([[2, 1],
                  [1, 1]])
    
    # Compute A × A^(-1)
    A_inv = np.linalg.inv(A)
    product = A @ A_inv
    
    # Check if it equals identity matrix
    identity = np.eye(2)
    result = np.allclose(product, identity)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Inverse Property - Double Inverse"""
    A = np.array([[1, 3],
                  [2, 5]])
    
    # Compute (A^(-1))^(-1)
    A_inv = np.linalg.inv(A)
    A_inv_inv = np.linalg.inv(A_inv)
    
    # Check if it equals A
    result = np.allclose(A_inv_inv, A)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Inverse Property - Transpose"""
    A = np.array([[2, 1],
                  [3, 4]])
    
    # Compute (A^T)^(-1)
    A_T_inv = np.linalg.inv(A.T)
    
    # Compute (A^(-1))^T
    A_inv_T = np.linalg.inv(A).T
    
    # Check if they are equal
    result = np.allclose(A_T_inv, A_inv_T)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Inverse of Product"""
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    # Compute (AB)^(-1)
    AB = A @ B
    AB_inv = np.linalg.inv(AB)
    
    # Compute B^(-1) A^(-1) (note the order is reversed!)
    B_inv_A_inv = np.linalg.inv(B) @ np.linalg.inv(A)
    
    # Check if they are equal
    result = np.allclose(AB_inv, B_inv_A_inv)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Check if Matrix is Invertible"""
    A = np.array([[1, 2],
                  [2, 4]])  # This matrix is singular (det = 0)
    
    # A matrix is invertible if its determinant is not zero
    det_A = np.linalg.det(A)
    result = not np.isclose(det_A, 0.0)
    
    # Alternative: Try to compute inverse and catch exception
    # try:
    #     A_inv = np.linalg.inv(A)
    #     result = True
    # except np.linalg.LinAlgError:
    #     result = False
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Solve Linear System Using Inverse"""
    A = np.array([[2, 1],
                  [1, 1]])
    b = np.array([5, 3])
    
    # Solution: x = A^(-1) b
    A_inv = np.linalg.inv(A)
    result = A_inv @ b
    
    # Alternative: Use np.linalg.solve(A, b) which is more numerically stable
    # result = np.linalg.solve(A, b)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Inverse of Identity Matrix"""
    # Create a 3x3 identity matrix
    I = np.eye(3)
    
    # Compute inverse
    I_inv = np.linalg.inv(I)
    
    # Check if I^(-1) = I
    result = np.allclose(I_inv, I)
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Inverse of Scalar Multiple"""
    A = np.array([[1, 2],
                  [3, 4]])
    k = 2
    
    # Compute (kA)^(-1)
    kA_inv = np.linalg.inv(k * A)
    
    # Compute (1/k) A^(-1)
    one_over_k_A_inv = (1 / k) * np.linalg.inv(A)
    
    # Check if they are equal
    result = np.allclose(kA_inv, one_over_k_A_inv)
    
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
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

