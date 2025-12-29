"""
Identity Matrix Exercise

Complete the functions below to practice identity matrix operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Create Identity Matrix
    Create a 3x3 identity matrix.
    
    Expected result: [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]
    """
    
    result = np.eye(3)
    print(result)
    return result


def exercise_2():
    """
    Exercise 2: Identity as Multiplicative Identity
    Verify that A @ I = A for any matrix A.
    
    Given matrix A, compute A @ I and check if it equals A.
    Return True if equal, False otherwise.
    """
    A = np.array([[2, 5, 1],
                  [3, 1, 4],
                  [1, 2, 3]])
    
    result = A @ np.eye(3)
    
    return np.array_equal(result, A)


def exercise_3():
    """
    Exercise 3: Identity with Vectors
    Verify that I @ v = v for any vector v.
    
    Given vector v, compute I @ v and check if it equals v.
    Return True if equal, False otherwise.
    """
    v = np.array([1, 2, 3, 4])
    
    result = np.eye(4) @ v
    print(result)

    return np.array_equal(result, v)


def exercise_4():
    """
    Exercise 4: Identity Transpose Property
    Verify that I^T = I (identity matrix is symmetric).
    
    Create a 4x4 identity matrix and check if its transpose equals itself.
    Return True if equal, False otherwise.
    """

    result = np.eye(4).T
    print(result)
    return np.array_equal(result, np.eye(4))

def exercise_5():
    """
    Exercise 5: Identity Determinant
    The determinant of an identity matrix is always 1.
    
    Create a 5x5 identity matrix and verify its determinant is 1.
    Return True if det(I) = 1, False otherwise.
    """
   
    result = np.eye(5)
    print(result)
    return np.isclose(np.linalg.det(result), 1.0)


def exercise_6():
    """
    Exercise 6: Identity Power Property
    Verify that I^n = I for any positive integer n.
    
    Create a 3x3 identity matrix and verify that I^2 = I.
    Return True if equal, False otherwise.
    """
    result = np.eye(3)
    print(result)
    return np.array_equal(result, result @ result)


def exercise_7():
    """
    Exercise 7: Identity in Matrix Addition
    Verify that A + 0 = A, where 0 is a zero matrix (not identity, but related).
    
    Actually, let's verify: A + (A - A) = A, which uses identity properties.
    Or better: A @ I + 0 = A @ I
    
    Given matrix A, verify that A @ I + 0 = A @ I.
    Return True if equal, False otherwise.
    """
    
    A = np.array([[1, 2],
                  [3, 4]])
    I = np.eye(2)
    zero = np.zeros((2, 2))
    left_side = A @ I + zero
    right_side = A @ I
    print(left_side)
    print(right_side)
    return np.array_equal(left_side, right_side)


def exercise_8():
    """
    Exercise 8: Identity and Inverse Relationship
    Verify that A @ A^(-1) = I for an invertible matrix A.
    
    Given matrix A, compute A @ A^(-1) and check if it equals I.
    Return True if equal, False otherwise.
    """
    A = np.array([[2, 1],
                  [1, 3]])
    
    result = A @ np.linalg.inv(A)
    print(result)
    return np.allclose(result, np.eye(2))


def exercise_9():
    """
    Exercise 9: Scalar Multiple of Identity
    Create a scalar multiple of identity matrix: kI where k=3.
    
    Create a 3x3 matrix that is 3 times the identity matrix.
    
    Expected result: [[3, 0, 0],
                      [0, 3, 0],
                      [0, 0, 3]]
    """
    k = 3
    
    result = k * np.eye(3)
    print(result)
    return result


def exercise_10():
    """
    Exercise 10: Identity Matrix Size
    Create identity matrices of different sizes and return their shapes.
    
    Create identity matrices of sizes 2, 3, and 4, and return a tuple
    containing their shapes: (shape2, shape3, shape4)
    
    Expected result: ((2, 2), (3, 3), (4, 4))
    """
    I2 = np.eye(2)
    I3 = np.eye(3)
    I4 = np.eye(4)
    print(I2.shape)
    print(I3.shape)
    print(I4.shape)
    return (I2.shape, I3.shape, I4.shape)


# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Create Identity Matrix")
    result1 = exercise_1()
    expected1 = np.eye(3)
    if result1 is not None and np.array_equal(result1, expected1):
        print(f"✓ Correct! Identity matrix:\n{result1}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected1}")
        if result1 is not None:
            print(f"Got:\n{result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Identity as Multiplicative Identity")
    result2 = exercise_2()
    if result2:
        print("✓ Correct! A @ I = A")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 3
    print("\nExercise 3: Identity with Vectors")
    result3 = exercise_3()
    if result3:
        print("✓ Correct! I @ v = v")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 4
    print("\nExercise 4: Identity Transpose Property")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! I^T = I")
    else:
        print("✗ Incorrect. Identity matrix should be symmetric.")
    
    # Test Exercise 5
    print("\nExercise 5: Identity Determinant")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! det(I) = 1")
    else:
        print("✗ Incorrect. Determinant of identity should be 1.")
    
    # Test Exercise 6
    print("\nExercise 6: Identity Power Property")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! I^2 = I")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 7
    print("\nExercise 7: Identity in Matrix Addition")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! A @ I + 0 = A @ I")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 8
    print("\nExercise 8: Identity and Inverse Relationship")
    result8 = exercise_8()
    if result8:
        print("✓ Correct! A @ A^(-1) = I")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 9
    print("\nExercise 9: Scalar Multiple of Identity")
    result9 = exercise_9()
    expected9 = np.array([[3, 0, 0],
                          [0, 3, 0],
                          [0, 0, 3]])
    if result9 is not None and np.array_equal(result9, expected9):
        print(f"✓ Correct! Result:\n{result9}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected9}")
        if result9 is not None:
            print(f"Got:\n{result9}")
    
    # Test Exercise 10
    print("\nExercise 10: Identity Matrix Size")
    result10 = exercise_10()
    expected10 = ((2, 2), (3, 3), (4, 4))
    if result10 is not None and result10 == expected10:
        print(f"✓ Correct! Shapes: {result10}")
    else:
        print(f"✗ Incorrect. Expected: {expected10}")
        if result10 is not None:
            print(f"Got: {result10}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

