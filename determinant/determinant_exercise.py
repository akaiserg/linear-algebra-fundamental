"""
Matrix Determinant Exercise

Complete the functions below to practice determinant operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Determinant
    Calculate the determinant of a 2x2 matrix.
    
    A = [[3, 1],
         [2, 4]]
    
    Expected result: 10
    (Formula: det = a*d - b*c = 3*4 - 1*2 = 12 - 2 = 10)
    """
    A = np.array([[3, 1],
                  [2, 4]])
    
    
    result = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    #result = np.linalg.det(A)
    
    return result


def exercise_2():
    """
    Exercise 2: Determinant of 3x3 Matrix
    Calculate the determinant of a 3x3 matrix.
    
    A = [[1, 2, 3],
         [0, 1, 4],
         [5, 6, 0]]
    
    Expected result: 1
    """
    A = np.array([[1, 2, 3],
                  [0, 1, 4],
                  [5, 6, 0]])
    
    
    result = A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) - A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0]) + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0])
    #result = np.linalg.det(A)
    
    return result


def exercise_3():
    """
    Exercise 3: Determinant Property - Scalar Multiplication
    Verify that det(kA) = k^n * det(A) for an n×n matrix A.
    
    Given matrix A and scalar k=2, verify this property.
    Return True if the property holds, False otherwise.
    """
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
    print(left_side)
    print(right_side)
    print(result)
    return result


def exercise_4():
    """
    Exercise 4: Determinant Property - Product
    Verify that det(AB) = det(A) * det(B)
    
    Given matrices A and B, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
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
    print(left_side)
    print(right_side)
    print(result)
    
    return result


def exercise_5():
    """
    Exercise 5: Determinant Property - Transpose
    Verify that det(A) = det(A^T)
    
    Given matrix A, compute det(A) and det(A^T), then check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[2, 5, 1],
                  [3, 1, 4],
                  [1, 2, 3]])
    
    
    result = np.isclose(np.linalg.det(A), np.linalg.det(A.T))
    print(np.linalg.det(A))
    print(np.linalg.det(A.T))
    print(result)
    
    return result


def exercise_6():
    """
    Exercise 6: Determinant of Identity Matrix
    The determinant of an identity matrix is always 1.
    
    Create a 3x3 identity matrix and verify its determinant is 1.
    Return True if det(I) = 1, False otherwise.
    """
    # TODO: Create a 3x3 identity matrix and check its determinant
    # Your code here:
    I = np.eye(3)
    result = np.isclose(np.linalg.det(I), 1.0)
    print(np.linalg.det(I))
    print(result)
    
    return result


def exercise_7():
    """
    Exercise 7: Singular Matrix (Zero Determinant)
    A matrix is singular (not invertible) if its determinant is 0.
    
    Check if the given matrix is singular.
    Return True if singular (det = 0), False otherwise.
    """
    A = np.array([[1, 2],
                  [2, 4]])  # Second row is 2× first row (linearly dependent)
    
    
    result = np.isclose(np.linalg.det(A), 0.0)
    print(np.linalg.det(A))
    print(result)
    
    return result


def exercise_8():
    """
    Exercise 8: Determinant After Row Operations
    Swapping two rows multiplies the determinant by -1.
    
    Given matrix A, swap two rows and verify that the new determinant
    equals -1 times the original determinant.
    Return True if the property holds, False otherwise.
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    # Swap two rows of A
    A_swapped = A.copy()
    A_swapped[[0, 1]] = A_swapped[[1, 0]]
    
    # Compare determinants
    result = np.isclose(np.linalg.det(A_swapped), -1 * np.linalg.det(A))
    print(np.linalg.det(A_swapped))
    print(np.linalg.det(A))
    print(result)
    
    
    return result


# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Basic Determinant (2x2)")
    result1 = exercise_1()
    expected1 = 10.0
    if result1 is not None and np.isclose(result1, expected1):
        print(f"✓ Correct! Determinant = {result1}")
    else:
        print(f"✗ Incorrect. Expected: {expected1}")
        if result1 is not None:
            print(f"Got: {result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Determinant of 3x3 Matrix")
    result2 = exercise_2()
    expected2 = 1.0
    if result2 is not None and np.isclose(result2, expected2):
        print(f"✓ Correct! Determinant = {result2}")
    else:
        print(f"✗ Incorrect. Expected: {expected2}")
        if result2 is not None:
            print(f"Got: {result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Determinant Property - Scalar Multiplication")
    result3 = exercise_3()
    print("--> result3", result3)
    if result3 :
        print("✓ Correct! det(kA) = k^n * det(A)")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 4
    print("\nExercise 4: Determinant Property - Product")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! det(AB) = det(A) * det(B)")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 5
    print("\nExercise 5: Determinant Property - Transpose")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! det(A) = det(A^T)")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 6
    print("\nExercise 6: Determinant of Identity Matrix")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! det(I) = 1")
    else:
        print("✗ Incorrect. Identity matrix determinant should be 1.")
    
    # Test Exercise 7
    print("\nExercise 7: Singular Matrix (Zero Determinant)")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! The matrix is singular (det = 0)")
    else:
        print("✗ Incorrect. This matrix should be singular.")
    
    # Test Exercise 8
    print("\nExercise 8: Determinant After Row Swap")
    result8 = exercise_8()
    if result8:
        print("✓ Correct! Swapping rows multiplies det by -1")
    else:
        print("✗ Incorrect. Swapping two rows should multiply det by -1.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

