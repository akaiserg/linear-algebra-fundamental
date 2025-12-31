"""
Outer Product Exercise

Complete the functions below to practice outer product operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Outer Product
    Calculate the outer product of two vectors.
    
    a = [1, 2, 3]
    b = [4, 5]
    
    Expected result: [[4,  5],
                      [8, 10],
                      [12, 15]]
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    
   
    return np.outer(a, b)

def exercise_2():
    """
    Exercise 2: Outer Product Using np.outer()
    Calculate the outer product using np.outer() function.
    
    a = [2, 3]
    b = [5, 7, 9]
    
    Expected result: [[10, 14, 18],
                      [15, 21, 27]]
    """
    a = np.array([2, 3])
    b = np.array([5, 7, 9])
    
   
    return np.outer(a, b)


def exercise_3():
    """
    Exercise 3: Outer Product Using Matrix Multiplication
    Calculate the outer product using matrix multiplication (reshape method).
    
    a = [1, 2]
    b = [3, 4, 5]
    
    Expected result: [[3,  4,  5],
                      [6,  8, 10]]
    """
    a = np.array([1, 2])
    b = np.array([3, 4, 5])
    
    return a.reshape(-1, 1) @ b.reshape(1, -1)


def exercise_4():
    """
    Exercise 4: Outer Product Property - Distributive
    Verify that (a + b) ⊗ c = a ⊗ c + b ⊗ c (distributive property).
    
    Given vectors a, b, and c, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    
    left_side = np.outer(a + b, c)
    right_side = np.outer(a, c) + np.outer(b, c)
    return np.allclose(left_side, right_side)


def exercise_5():
    """
    Exercise 5: Outer Product Property - Scalar Multiplication
    Verify that (ka) ⊗ b = k(a ⊗ b) = a ⊗ (kb)
    
    Given vectors a and b, and scalar k=3, verify this property.
    Return True if all three are equal, False otherwise.
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    k = 3
    
   
    left_side = np.outer(k * a, b)
    right_side = k * np.outer(a, b)
    third_side = np.outer(a, k * b)
    return np.allclose(left_side, right_side) and np.allclose(right_side, third_side) and np.allclose(left_side, third_side)

def exercise_6():
    """
    Exercise 6: Outer Product vs Dot Product
    Compare the outer product and dot product of the same vectors.
    
    a = [1, 2, 3]
    b = [4, 5, 6]
    
    Return a tuple (outer_product, dot_product) where:
    - outer_product is the outer product (a matrix)
    - dot_product is the dot product (a scalar)
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
   
    outer_product = np.outer(a, b)
    dot_product = np.dot(a, b)
    return (outer_product, dot_product)

def exercise_7():
    """
    Exercise 7: Rank of Outer Product Matrix
    The outer product of two non-zero vectors always produces a rank-1 matrix.
    
    Calculate the rank of the outer product matrix.
    
    a = [2, 3, 4]
    b = [5, 6]
    
    Expected result: 1 (rank-1 matrix)
    """
    a = np.array([2, 3, 4])
    b = np.array([5, 6])
    
   
    return np.linalg.matrix_rank(np.outer(a, b))


def exercise_8():
    """
    Exercise 8: Outer Product Dimensions
    For vectors a (length m) and b (length n), the outer product has shape (m, n).
    
    Given vectors a and b, return the shape of their outer product.
    
    a = [1, 2, 3, 4]
    b = [5, 6, 7]
    
    Expected result: (4, 3)
    """
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7])
    
   
    return np.outer(a, b).shape

def exercise_9():
    """
    Exercise 9: Outer Product with Standard Basis Vectors
    Create a matrix using outer product of standard basis vectors.
    
    Create the matrix [[1, 0],
                      [0, 0]]
    using outer product of standard basis vectors e1 = [1, 0] and e1 = [1, 0].
    
    Return the resulting matrix.
    """
   
    return np.outer(np.array([1, 0]), np.array([1, 0]))


def exercise_10():
    """
    Exercise 10: Outer Product Transpose Property
    Verify that (a ⊗ b)^T = b ⊗ a
    
    Given vectors a and b, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    
   
    left_side = np.outer(a, b).T
    right_side = np.outer(b, a)
    return np.allclose(left_side, right_side)

# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Basic Outer Product")
    result1 = exercise_1()
    expected1 = np.array([[4,  5],
                          [8, 10],
                          [12, 15]])
    if result1 is not None and np.array_equal(result1, expected1):
        print(f"✓ Correct! Outer product:\n{result1}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected1}")
        if result1 is not None:
            print(f"Got:\n{result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Outer Product Using np.outer()")
    result2 = exercise_2()
    expected2 = np.array([[10, 14, 18],
                          [15, 21, 27]])
    if result2 is not None and np.array_equal(result2, expected2):
        print(f"✓ Correct! Outer product:\n{result2}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected2}")
        if result2 is not None:
            print(f"Got:\n{result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Outer Product Using Matrix Multiplication")
    result3 = exercise_3()
    expected3 = np.array([[3,  4,  5],
                          [6,  8, 10]])
    if result3 is not None and np.array_equal(result3, expected3):
        print(f"✓ Correct! Outer product:\n{result3}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected3}")
        if result3 is not None:
            print(f"Got:\n{result3}")
    
    # Test Exercise 4
    print("\nExercise 4: Outer Product Property - Distributive")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! (a + b) ⊗ c = a ⊗ c + b ⊗ c")
    else:
        print("✗ Incorrect. The distributive property should hold.")
    
    # Test Exercise 5
    print("\nExercise 5: Outer Product Property - Scalar Multiplication")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! (ka) ⊗ b = k(a ⊗ b) = a ⊗ (kb)")
    else:
        print("✗ Incorrect. The scalar multiplication property should hold.")
    
    # Test Exercise 6
    print("\nExercise 6: Outer Product vs Dot Product")
    result6 = exercise_6()
    if result6 is not None:
        outer, dot = result6
        expected_outer = np.array([[4,  5,  6],
                                   [8, 10, 12],
                                   [12, 15, 18]])
        expected_dot = 32
        if np.array_equal(outer, expected_outer) and dot == expected_dot:
            print(f"✓ Correct! Outer product (matrix):\n{outer}")
            print(f"  Dot product (scalar): {dot}")
        else:
            print(f"✗ Incorrect. Expected outer:\n{expected_outer}")
            print(f"  Expected dot: {expected_dot}")
            if outer is not None:
                print(f"  Got outer:\n{outer}")
            if dot is not None:
                print(f"  Got dot: {dot}")
    else:
        print("✗ Incorrect. Expected a tuple (outer_product, dot_product)")
    
    # Test Exercise 7
    print("\nExercise 7: Rank of Outer Product Matrix")
    result7 = exercise_7()
    expected7 = 1
    if result7 is not None and result7 == expected7:
        print(f"✓ Correct! Rank = {result7}")
    else:
        print(f"✗ Incorrect. Expected: {expected7}")
        if result7 is not None:
            print(f"Got: {result7}")
    
    # Test Exercise 8
    print("\nExercise 8: Outer Product Dimensions")
    result8 = exercise_8()
    expected8 = (4, 3)
    if result8 is not None and result8 == expected8:
        print(f"✓ Correct! Shape: {result8}")
    else:
        print(f"✗ Incorrect. Expected: {expected8}")
        if result8 is not None:
            print(f"Got: {result8}")
    
    # Test Exercise 9
    print("\nExercise 9: Outer Product with Standard Basis Vectors")
    result9 = exercise_9()
    expected9 = np.array([[1, 0],
                          [0, 0]])
    if result9 is not None and np.array_equal(result9, expected9):
        print(f"✓ Correct! Matrix:\n{result9}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected9}")
        if result9 is not None:
            print(f"Got:\n{result9}")
    
    # Test Exercise 10
    print("\nExercise 10: Outer Product Transpose Property")
    result10 = exercise_10()
    if result10:
        print("✓ Correct! (a ⊗ b)^T = b ⊗ a")
    else:
        print("✗ Incorrect. The transpose property should hold.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

