"""
Matrix Multiplication Exercise

Complete the functions below to practice matrix multiplication operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic 2x2 Matrix Multiplication
    Calculate the product of two 2x2 matrices.
    
    A = [[1, 2],
         [3, 4]]
    B = [[5, 6],
         [7, 8]]
    
    Expected result: [[19, 22],
                      [43, 50]]
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    result = A @ B
    print(result)
    return result


def exercise_2():
    """
    Exercise 2: Matrix Multiplication Using np.dot()
    Calculate matrix product using np.dot() function.
    
    A = [[1, 2, 3],
         [4, 5, 6]]
    B = [[7, 8],
         [9, 10],
         [11, 12]]
    
    Expected result: [[58, 64],
                      [139, 154]]
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    result = np.dot(A, B)
    print(result)
    
    return result


def exercise_3():
    """
    Exercise 3: Matrix-Vector Multiplication
    Calculate the product of a matrix and a vector.
    
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    v = [10, 11, 12]
    
    Expected result: [68, 167, 266]
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    v = np.array([10, 11, 12])
    
    result = A @ v
    print(result)
    return result


def exercise_4():
    """
    Exercise 4: Matrix Multiplication Property - Associative
    Verify that (AB)C = A(BC) (matrix multiplication is associative).
    
    Given matrices A, B, and C, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    C = np.array([[9, 10],
                  [11, 12]])
    
    left_side = (A @ B) @ C
    right_side = A @ (B @ C)
    print(left_side)
    print(right_side)
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_5():
    """
    Exercise 5: Matrix Multiplication Property - Distributive
    Verify that A(B + C) = AB + AC (distributive property).
    
    Given matrices A, B, and C, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    C = np.array([[9, 10],
                  [11, 12]])
    
    left_side = A @ (B + C)
    right_side = A @ B + A @ C
    print(left_side)
    print(right_side)
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_6():
    """
    Exercise 6: Matrix Multiplication Property - Scalar Multiplication
    Verify that k(AB) = (kA)B = A(kB)
    
    Given matrices A and B, and scalar k=2, verify this property.
    Return True if all three are equal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    k = 2
    
    left_side = k * (A @ B)
    right_side = (k * A) @ B
    third_side = A @ (k * B)
    print(left_side)
    print(right_side)
    print(third_side)
    
    return np.array_equal(left_side, right_side) and np.array_equal(right_side, third_side) and np.array_equal(left_side, third_side)


def exercise_7():
    """
    Exercise 7: Identity Matrix Multiplication
    Verify that A @ I = A and I @ A = A (when dimensions match).
    
    Given matrix A, create appropriate identity matrices and verify.
    Return True if both properties hold, False otherwise.
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    I3 = np.eye(3)
    I2 = np.eye(2)
    left_side = A @ I3
    right_side = I2 @ A
    print(left_side)
    print(right_side)
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_8():
    """
    Exercise 8: Matrix Multiplication is NOT Commutative
    Verify that A @ B ≠ B @ A in general (matrix multiplication is not commutative).
    
    Given matrices A and B, compute both products and check if they are different.
    Return True if they are different, False if they happen to be equal.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    AB = A @ B
    BA = B @ A
    print(AB)
    print(BA)
    result = not np.array_equal(AB, BA) # not equal 
    
    return result


def exercise_9():
    """
    Exercise 9: Transpose of Matrix Product
    Verify that (AB)^T = B^T A^T (transpose of product reverses order).
    
    Given matrices A and B, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    
    left_side = (A @ B).T
    right_side = B.T @ A.T
    print(left_side)
    print(right_side)
    result = np.array_equal(left_side, right_side)
    
    return result


def exercise_10():
    """
    Exercise 10: Matrix Multiplication with Different Dimensions
    Calculate the product of matrices with different dimensions.
    
    A = [[1, 2, 3],
         [4, 5, 6]]  (2×3)
    B = [[7, 8],
         [9, 10],
         [11, 12]]  (3×2)
    
    Expected result shape: (2, 2)
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    result = A @ B
    shape = result.shape
    print(result)
    print(shape)
    
    return result, shape


# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Basic 2x2 Matrix Multiplication")
    result1 = exercise_1()
    expected1 = np.array([[19, 22],
                          [43, 50]])
    if result1 is not None and np.array_equal(result1, expected1):
        print(f"✓ Correct! Result:\n{result1}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected1}")
        if result1 is not None:
            print(f"Got:\n{result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Matrix Multiplication Using np.dot()")
    result2 = exercise_2()
    expected2 = np.array([[58, 64],
                          [139, 154]])
    if result2 is not None and np.array_equal(result2, expected2):
        print(f"✓ Correct! Result:\n{result2}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected2}")
        if result2 is not None:
            print(f"Got:\n{result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Matrix-Vector Multiplication")
    result3 = exercise_3()
    expected3 = np.array([68, 167, 266])
    if result3 is not None and np.array_equal(result3, expected3):
        print(f"✓ Correct! Result: {result3}")
    else:
        print(f"✗ Incorrect. Expected: {expected3}")
        if result3 is not None:
            print(f"Got: {result3}")
    
    # Test Exercise 4
    print("\nExercise 4: Matrix Multiplication Property - Associative")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! (AB)C = A(BC)")
    else:
        print("✗ Incorrect. The associative property should hold.")
    
    # Test Exercise 5
    print("\nExercise 5: Matrix Multiplication Property - Distributive")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! A(B + C) = AB + AC")
    else:
        print("✗ Incorrect. The distributive property should hold.")
    
    # Test Exercise 6
    print("\nExercise 6: Matrix Multiplication Property - Scalar Multiplication")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! k(AB) = (kA)B = A(kB)")
    else:
        print("✗ Incorrect. The scalar multiplication property should hold.")
    
    # Test Exercise 7
    print("\nExercise 7: Identity Matrix Multiplication")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! A @ I = A and I @ A = A")
    else:
        print("✗ Incorrect. Identity matrix properties should hold.")
    
    # Test Exercise 8
    print("\nExercise 8: Matrix Multiplication is NOT Commutative")
    result8 = exercise_8()
    if result8:
        print("✓ Correct! A @ B ≠ B @ A (in general)")
    else:
        print("✗ Incorrect. These matrices should have different products.")
    
    # Test Exercise 9
    print("\nExercise 9: Transpose of Matrix Product")
    result9 = exercise_9()
    if result9:
        print("✓ Correct! (AB)^T = B^T A^T")
    else:
        print("✗ Incorrect. The transpose property should hold.")
    
    # Test Exercise 10
    print("\nExercise 10: Matrix Multiplication with Different Dimensions")
    result10, shape10 = exercise_10()
    expected10 = np.array([[58, 64],
                           [139, 154]])
    expected_shape = (2, 2)
    if result10 is not None and np.array_equal(result10, expected10) and shape10 == expected_shape:
        print(f"✓ Correct! Result shape: {shape10}")
        print(f"Result:\n{result10}")
    else:
        print(f"✗ Incorrect. Expected shape: {expected_shape}")
        if result10 is not None:
            print(f"Got shape: {shape10}")
            print(f"Result:\n{result10}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

