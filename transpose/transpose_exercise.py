"""
Matrix Transpose Exercise

Complete the functions below to practice transpose operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Transpose
    Given the matrix A, return its transpose.
    
    A = [[1, 2, 3],
         [4, 5, 6]]
    
    Expected result: [[1, 4],
                      [2, 5],
                      [3, 6]]
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    
    result = A.T
    
    return result


def exercise_2():
    """
    Exercise 2: Transpose Property
    Verify that (A^T)^T = A for any matrix A.
    
    Given matrix A, compute (A^T)^T and check if it equals A.
    Return True if they are equal, False otherwise.
    """
    A = np.array([[2, 5, 8],
                  [1, 3, 7],
                  [9, 4, 6]])
    
    
    result = np.array_equal((A.T).T, A)
    
    return result


def exercise_3():
    """
    Exercise 3: Transpose of Sum
    Verify that (A + B)^T = A^T + B^T
    
    Given matrices A and B, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[1, 3],
                  [2, 4]])
    B = np.array([[5, 7],
                  [6, 8]])
        
    result =  np.array_equal((A + B).T, (A.T + B.T))
    
    return result


def exercise_4():
    """
    Exercise 4: Transpose of Product
    Verify that (AB)^T = B^T A^T (note the order reversal!)
    
    Given matrices A and B, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
        
    result = np.array_equal((A*B).T,(B.T * A.T))
    
    return result


def exercise_5():
    """
    Exercise 5: Convert Row Vector to Column Vector
    Given a row vector, convert it to a column vector using transpose.
    
    vector = [10, 20, 30, 40]
    Expected shape: (4, 1)
    """
    vector = np.array([10, 20, 30, 40])
    
    # TODO: Convert the row vector to a column vector
    # Hint: You might need to reshape first, then transpose
    # Your code here:
    
    result =vector.reshape(-1, 1)
    
    return result


def exercise_6():
    """
    Exercise 6: Check if Matrix is Symmetric
    A matrix is symmetric if A = A^T
    
    Check if the given matrix is symmetric.
    Return True if symmetric, False otherwise.
    """
    A = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    
    
    result = np.array_equal(A, A.T)
    
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
    print("\nExercise 1: Basic Transpose")
    result1 = exercise_1()
    expected1 = np.array([[1, 4],
                          [2, 5],
                          [3, 6]])
    if result1 is not None and np.array_equal(result1, expected1):
        print("✓ Correct!")
    else:
        print("✗ Incorrect. Expected:")
        print(expected1)
        if result1 is not None:
            print("Got:")
            print(result1)
    
    # Test Exercise 2
    print("\nExercise 2: Transpose Property")
    result2 = exercise_2()
    print(result2)
    if result2 is True:
        print("✓ Correct!")
    else:
        print("✗ Incorrect. (A^T)^T should equal A")
    
    # Test Exercise 3
    print("\nExercise 3: Transpose of Sum")
    result3 = exercise_3()
    if result3 is True:
        print("✓ Correct!")
    else:
        print("✗ Incorrect. (A + B)^T should equal A^T + B^T")
    
    # Test Exercise 4
    print("\nExercise 4: Transpose of Product")
    result4 = exercise_4()
    if result4 is True:
        print("✓ Correct!")
    else:
        print("✗ Incorrect. (AB)^T should equal B^T A^T")
    
    # Test Exercise 5
    print("\nExercise 5: Row to Column Vector")
    result5 = exercise_5()
    if result5 is not None and result5.shape == (4, 1):
        print("✓ Correct! Shape:", result5.shape)
    else:
        print("✗ Incorrect. Expected shape: (4, 1)")
        if result5 is not None:
            print("Got shape:", result5.shape)
    
    # Test Exercise 6
    print("\nExercise 6: Check if Symmetric")
    result6 = exercise_6()
    if result6 is True:
        print("✓ Correct! The matrix is symmetric.")
    else:
        print("✗ Incorrect. The matrix should be symmetric (A = A^T)")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

