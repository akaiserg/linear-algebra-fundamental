"""
Matrix Inverse Exercise

Complete the functions below to practice matrix inverse operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Matrix Inverse
    Calculate the inverse of a 2x2 matrix.
    
    A = [[3, 1],
         [2, 4]]
    
    Expected result: [[ 0.4, -0.1],
                      [-0.2,  0.3]]
    """
    A = np.array([[3, 1],
                  [2, 4]])
    
    result = np.linalg.inv(A)
    print(result)
    return result


def exercise_2():
    """
    Exercise 2: Inverse of 3x3 Matrix
    Calculate the inverse of a 3x3 matrix.
    
    A = [[1, 2, 0],
         [0, 1, 1],
         [1, 0, 1]]
    
    Verify your result by checking if A × A^(-1) = I
    """
    A = np.array([[1, 2, 0],
                  [0, 1, 1],
                  [1, 0, 1]])
    
    result = np.linalg.inv(A)
    print(result)
    print(A @ result)
    return result


def exercise_3():
    """
    Exercise 3: Verify Inverse Property
    Verify that A × A^(-1) = I (identity matrix).
    
    Given matrix A, compute A × A^(-1) and check if it equals the identity matrix.
    Return True if equal, False otherwise.
    """
    A = np.array([[2, 1],
                  [1, 1]])
    
    result = A @ np.linalg.inv(A)
    print(result)
    
    return np.array_equal(result, np.eye(2))


def exercise_4():
    """
    Exercise 4: Inverse Property - Double Inverse
    Verify that (A^(-1))^(-1) = A
    
    Given matrix A, compute (A^(-1))^(-1) and check if it equals A.
    Return True if equal, False otherwise.
    """
    A = np.array([[1, 3],
                  [2, 5]])
    
    
    result = np.linalg.inv(np.linalg.inv(A))
    print(result)
    print(A)
    return np.allclose(result, A)


def exercise_5():
    """
    Exercise 5: Inverse Property - Transpose
    Verify that (A^T)^(-1) = (A^(-1))^T
    
    Given matrix A, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[2, 1],
                  [3, 4]])
    
    left_side = np.linalg.inv(A.T)
    right_side = np.linalg.inv(A).T
    print(left_side)
    print(right_side)
    return np.allclose(left_side, right_side)


def exercise_6():
    """
    Exercise 6: Inverse of Product
    Verify that (AB)^(-1) = B^(-1) A^(-1) (note the order reversal!)
    
    Given matrices A and B, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    
    left_side = np.linalg.inv(A @ B)
    right_side = np.linalg.inv(B) @ np.linalg.inv(A)
    print(left_side)
    print(right_side)
    return np.allclose(left_side, right_side)


def exercise_7():
    """
    Exercise 7: Check if Matrix is Invertible
    A matrix is invertible (non-singular) if its determinant is not zero.
    
    Check if the given matrix is invertible.
    Return True if invertible, False otherwise.
    """
    A = np.array([[1, 2],
                  [2, 4]])  # This matrix is singular (det = 0)
    
   
    result = np.isclose(np.linalg.det(A), 0.0)
    print(result)
    return not result


def exercise_8():
    """
    Exercise 8: Solve Linear System Using Inverse
    Solve the system Ax = b using the inverse: x = A^(-1) b
    
    A = [[2, 1],
         [1, 1]]
    b = [5, 3]
    
    Expected result: [2, 1]
    """
    A = np.array([[2, 1],
                  [1, 1]])
    b = np.array([5, 3])
    
   
    result = np.linalg.inv(A) @ b
    print(result)
    return result


def exercise_9():
    """
    Exercise 9: Inverse of Identity Matrix
    The inverse of an identity matrix is itself: I^(-1) = I
    
    Create a 3x3 identity matrix and verify that its inverse equals itself.
    Return True if I^(-1) = I, False otherwise.
    """
    
    I = np.eye(3)
    result = np.linalg.inv(I)
    print(result)
    return np.array_equal(result, I)
    


def exercise_10():
    """
    Exercise 10: Inverse of Scalar Multiple
    Verify that (kA)^(-1) = (1/k) A^(-1) for scalar k ≠ 0
    
    Given matrix A and scalar k=2, verify this property.
    Return True if the property holds, False otherwise.
    """
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

# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Basic Matrix Inverse (2x2)")
    result1 = exercise_1()
    expected1 = np.array([[0.4, -0.1],
                         [-0.2, 0.3]])
    if result1 is not None and np.allclose(result1, expected1):
        print(f"✓ Correct! Inverse:\n{result1}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected1}")
        if result1 is not None:
            print(f"Got:\n{result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Inverse of 3x3 Matrix")
    result2 = exercise_2()
    A = np.array([[1, 2, 0],
                  [0, 1, 1],
                  [1, 0, 1]])
    if result2 is not None:
        # Verify: A × A^(-1) should be identity
        verification = A @ result2
        if np.allclose(verification, np.eye(3)):
            print(f"✓ Correct! A × A^(-1) = I")
        else:
            print(f"✗ Incorrect. A × A^(-1) should equal identity matrix")
    else:
        print("✗ Incorrect. No result returned.")
    
    # Test Exercise 3
    print("\nExercise 3: Verify Inverse Property")
    result3 = exercise_3()
    if result3:
        print("✓ Correct! A × A^(-1) = I")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 4
    print("\nExercise 4: Inverse Property - Double Inverse")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! (A^(-1))^(-1) = A")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 5
    print("\nExercise 5: Inverse Property - Transpose")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! (A^T)^(-1) = (A^(-1))^T")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 6
    print("\nExercise 6: Inverse of Product")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! (AB)^(-1) = B^(-1) A^(-1)")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 7
    print("\nExercise 7: Check if Matrix is Invertible")
    result7 = exercise_7()
    if result7 is False:
        print("✓ Correct! The matrix is singular (not invertible)")
    else:
        print("✗ Incorrect. This matrix should not be invertible (det = 0).")
    
    # Test Exercise 8
    print("\nExercise 8: Solve Linear System Using Inverse")
    result8 = exercise_8()
    expected8 = np.array([2, 1])
    if result8 is not None and np.allclose(result8, expected8):
        print(f"✓ Correct! Solution x = {result8}")
    else:
        print(f"✗ Incorrect. Expected: {expected8}")
        if result8 is not None:
            print(f"Got: {result8}")
    
    # Test Exercise 9
    print("\nExercise 9: Inverse of Identity Matrix")
    result9 = exercise_9()
    if result9:
        print("✓ Correct! I^(-1) = I")
    else:
        print("✗ Incorrect. Identity matrix inverse should equal itself.")
    
    # Test Exercise 10
    print("\nExercise 10: Inverse of Scalar Multiple")
    result10 = exercise_10()
    if result10:
        print("✓ Correct! (kA)^(-1) = (1/k) A^(-1)")
    else:
        print("✗ Incorrect. The property should hold.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

