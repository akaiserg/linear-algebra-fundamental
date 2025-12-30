"""
Null Space, Nullity, and Kernel Exercise

Complete the functions below to practice null space operations.
Run this file to check your solutions.
"""

import numpy as np
from scipy.linalg import null_space


def exercise_1():
    """
    Exercise 1: Find Null Space
    Find the null space of matrix A.
    
    A = [[1, 2],
         [2, 4]]
    
    The null space should contain vectors x such that Ax = 0.
    Return the null space basis vectors as a numpy array.
    """
    A = np.array([[1, 2],
                  [2, 4]])
    
    result = null_space(A)
    print(result)
    return result


def exercise_2():
    """
    Exercise 2: Calculate Nullity
    Calculate the nullity (dimension of null space) of matrix A.
    
    A = [[1, 2, 3],
         [2, 4, 6],
         [3, 6, 9]]
    
    Expected result: 2 (nullity)
    """
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
    result = null_space(A)
    print(result)
    return result.shape[1]


def exercise_3():
    """
    Exercise 3: Verify Null Space Property
    Verify that for any vector x in the null space, Ax = 0.
    
    Given matrix A and a vector x, check if Ax = 0.
    Return True if Ax is close to zero, False otherwise.
    """
    A = np.array([[1, 1, 1],
                  [2, 2, 2]])
    x = np.array([1, -1, 0])  # This should be in the null space
    
    result = A @ x
    print(result)
    return np.allclose(result, 0)

def exercise_4():
    """
    Exercise 4: Rank-Nullity Theorem
    Verify the Rank-Nullity Theorem: rank(A) + nullity(A) = number of columns
    
    Given matrix A, compute rank + nullity and check if it equals number of columns.
    Return True if the theorem holds, False otherwise.
    """
    A = np.array([[1, 2, 3, 4],
                  [0, 1, 2, 3],
                  [0, 0, 0, 0]])
    
   
    left_side = np.linalg.matrix_rank(A) + null_space(A).shape[1]
    right_side = A.shape[1]
    print(left_side, right_side)
    return left_side == right_side


def exercise_5():
    """
    Exercise 5: Null Space of Full Rank Matrix
    A full rank matrix (rank = number of columns) has nullity = 0.
    
    Check if the given matrix is full rank and verify its nullity is 0.
    Return True if nullity is 0, False otherwise.
    """
    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])  # Identity matrix (full rank)
    
   
    result = null_space(A)
    print(result)
    return result.shape[1] == 0

def exercise_6():
    """
    Exercise 6: Null Space and Linear Dependence
    If columns of A are linearly dependent, nullity > 0.
    
    Check if the given matrix has linearly dependent columns by checking nullity.
    Return True if nullity > 0 (dependent columns), False otherwise.
    """
    A = np.array([[1, 2, 3],
                  [2, 4, 6]])  # Third column is 1.5× second column
    
    result = null_space(A)
    print(result)
    return result.shape[1] > 0

def exercise_7():
    """
    Exercise 7: Null Space Using SVD
    Find nullity using SVD (Singular Value Decomposition).
    Nullity equals the number of zero (or near-zero) singular values.
    
    Given matrix A, find nullity using SVD.
    Return the nullity.
    """
    A = np.array([[1, 2],
                  [2, 4],
                  [3, 6]])
    
    
    # Perform SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    
    # Nullity equals number of zero (or near-zero) singular values
    # Use a small tolerance for floating point comparison
    zero_singular_values = np.sum(np.isclose(s, 0, atol=1e-10))
    result = zero_singular_values
    print(result)
    return result

def exercise_8():
    """
    Exercise 8: Null Space of Transpose
    Find the null space of A^T and compare with null space of A.
    
    Given matrix A, find nullity of A and nullity of A^T.
    Return a tuple (nullity_A, nullity_AT).
    """
    A = np.array([[1, 2, 3],
                  [2, 4, 6]])
    
   
    nullity_A = null_space(A).shape[1]
    nullity_AT = null_space(A.T).shape[1]
    print(nullity_A, nullity_AT)
    return (nullity_A, nullity_AT)

def exercise_9():
    """
    Exercise 9: Null Space of Product
    If C = AB, then null space of C contains null space of B.
    
    Given matrices A and B, check if nullity(AB) >= nullity(B).
    Return True if the property holds, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[1, 1],
                  [-1, -1]])  # Has null space
    
   
    nullity_AB = null_space(A @ B).shape[1]
    nullity_B = null_space(B).shape[1]
    print(nullity_AB, nullity_B)
    return nullity_AB >= nullity_B

def exercise_10():
    """
    Exercise 10: Manual Null Space Calculation
    Manually solve Ax = 0 to find a basis vector for the null space.
    
    Given matrix A = [[1, 1], [2, 2]], solve Ax = 0.
    The equation gives: x1 + x2 = 0, so x2 = -x1.
    Return a basis vector for the null space (e.g., [1, -1]).
    """
    A = np.array([[1, 1],
                  [2, 2]])
    
    result = null_space(A)
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
    print("\nExercise 1: Find Null Space")
    result1 = exercise_1()
    if result1 is not None:
        # Check if result is in null space
        A = np.array([[1, 2], [2, 4]])
        verification = A @ result1
        if np.allclose(verification, 0) and result1.shape[0] == 2:
            print(f"✓ Correct! Null space basis:\n{result1}")
        else:
            print("✗ Incorrect. Result should satisfy Ax = 0")
    else:
        print("✗ Incorrect. Expected null space basis vectors.")
    
    # Test Exercise 2
    print("\nExercise 2: Calculate Nullity")
    result2 = exercise_2()
    expected2 = 2
    if result2 is not None and result2 == expected2:
        print(f"✓ Correct! Nullity = {result2}")
    else:
        print(f"✗ Incorrect. Expected: {expected2}")
        if result2 is not None:
            print(f"Got: {result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Verify Null Space Property")
    result3 = exercise_3()
    if result3:
        print("✓ Correct! Ax = 0 for vector in null space")
    else:
        print("✗ Incorrect. Ax should be close to zero.")
    
    # Test Exercise 4
    print("\nExercise 4: Rank-Nullity Theorem")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! rank(A) + nullity(A) = number of columns")
    else:
        print("✗ Incorrect. The Rank-Nullity Theorem should hold.")
    
    # Test Exercise 5
    print("\nExercise 5: Null Space of Full Rank Matrix")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! Full rank matrix has nullity = 0")
    else:
        print("✗ Incorrect. Full rank matrix should have nullity = 0.")
    
    # Test Exercise 6
    print("\nExercise 6: Null Space and Linear Dependence")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! Linearly dependent columns → nullity > 0")
    else:
        print("✗ Incorrect. This matrix should have nullity > 0.")
    
    # Test Exercise 7
    print("\nExercise 7: Null Space Using SVD")
    result7 = exercise_7()
    expected7 = 1
    if result7 is not None and result7 == expected7:
        print(f"✓ Correct! Nullity from SVD = {result7}")
    else:
        print(f"✗ Incorrect. Expected: {expected7}")
        if result7 is not None:
            print(f"Got: {result7}")
    
    # Test Exercise 8
    print("\nExercise 8: Null Space of Transpose")
    result8 = exercise_8()
    if result8 is not None and isinstance(result8, tuple) and len(result8) == 2:
        print(f"✓ Correct! Nullity of A: {result8[0]}, Nullity of A^T: {result8[1]}")
    else:
        print("✗ Incorrect. Expected tuple (nullity_A, nullity_AT).")
    
    # Test Exercise 9
    print("\nExercise 9: Null Space of Product")
    result9 = exercise_9()
    if result9:
        print("✓ Correct! nullity(AB) >= nullity(B)")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 10
    print("\nExercise 10: Manual Null Space Calculation")
    result10 = exercise_10()
    if result10 is not None:
        A = np.array([[1, 1], [2, 2]])
        verification = A @ result10
        if np.allclose(verification, 0):
            print(f"✓ Correct! Basis vector: {result10}")
        else:
            print("✗ Incorrect. Vector should satisfy Ax = 0")
    else:
        print("✗ Incorrect. Expected a basis vector.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

