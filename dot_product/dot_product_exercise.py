"""
Dot Product Exercise

Complete the functions below to practice dot product operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Dot Product of Vectors
    Calculate the dot product of two vectors.
    
    a = [1, 2, 3]
    b = [4, 5, 6]
    
    Expected result: 32
    (1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32)
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    result = (1*4 + 2*5 + 3*6)
    
    return result


def exercise_2():
    """
    Exercise 2: Dot Product Using np.dot()
    Calculate the dot product using np.dot() function.
    
    a = [2, 3]
    b = [5, 7]
    
    Expected result: 31
    """
    a = np.array([2, 3])
    b = np.array([5, 7])
    
    
    result = np.dot(a, b)
    
    return result


def exercise_3():
    """
    Exercise 3: Dot Product Using @ Operator
    Calculate the dot product using the @ operator (matrix multiplication).
    
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    
    Expected result: 70
    """
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    
    
    result = a @ b
    
    return result


def exercise_4():
    """
    Exercise 4: Dot Product Property - Commutative
    Verify that a · b = b · a (dot product is commutative).
    
    Given vectors a and b, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    a = np.array([1, 3, 5])
    b = np.array([2, 4, 6])
    
    
    result = a @ b == b @ a
    
    return result


def exercise_5():
    """
    Exercise 5: Dot Product Property - Distributive
    Verify that a · (b + c) = a · b + a · c (distributive property).
    
    Given vectors a, b, and c, compute both sides and check if they are equal.
    Return True if equal, False otherwise.
    """
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    
    result = a @ (b+c) == a @ b + a @ c
    return result


def exercise_6():
    """
    Exercise 6: Dot Product Property - Scalar Multiplication
    Verify that (ka) · b = k(a · b) = a · (kb)
    
    Given vectors a and b, and scalar k=3, verify this property.
    Return True if all three are equal, False otherwise.
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    k = 3
    
    
    result = (k*a) @ b == k*(a @ b)
    
    return result


def exercise_7():
    """
    Exercise 7: Orthogonal Vectors (Perpendicular)
    Two vectors are orthogonal if their dot product is 0.
    
    Check if the given vectors are orthogonal.
    Return True if orthogonal, False otherwise.
    """
    a = np.array([1, 0])
    b = np.array([0, 1])
    
    
    result = a @ b == 0
    
    return result


def exercise_8():
    """
    Exercise 8: Dot Product of Matrix-Vector Multiplication
    Calculate the dot product of a matrix and a vector.
    
    A = [[1, 2],
         [3, 4]]
    v = [5, 6]
    
    Expected result: [17, 39]
    (Each row of A dotted with v)
    """
    A = np.array([[1, 2],
                  [3, 4]])
    v = np.array([5, 6])
    
    
    result = A @ v
    
    return result


def exercise_9():
    """
    Exercise 9: Dot Product of Two Matrices
    Calculate the matrix product (dot product) of two matrices.
    
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
    
    return result


def exercise_10():
    """
    Exercise 10: Magnitude (Length) of Vector Using Dot Product
    The magnitude (length) of a vector v is sqrt(v · v).
    
    Calculate the magnitude of vector v using dot product.
    v = [3, 4]
    
    Expected result: 5.0
    (sqrt(3*3 + 4*4) = sqrt(9 + 16) = sqrt(25) = 5)
    """
    v = np.array([3, 4])
    
   
    result = np.sqrt(v @ v)
    
    return result


def exercise_11():
    """
    Exercise 11: Vectors Pointing in Same Direction (θ = 0°)
    When vectors point in the same direction, the dot product is positive and maximum.
    The formula: a · b = ||a|| ||b|| cos(θ)
    When θ = 0°, cos(0°) = 1, so a · b = ||a|| ||b||
    
    Given two vectors pointing in the same direction, verify that:
    a · b = ||a|| ||b||
    
    a = [3, 4]
    b = [6, 8]  (b is 2×a, same direction)
    
    Return True if the property holds, False otherwise.
    """
    a = np.array([3, 4])
    b = np.array([6, 8])  # b = 2a, so same direction
    
    
    result = a @ b == np.linalg.norm(a) * np.linalg.norm(b)
    a_norm_l2=a[0]**2 + a[1]**2
    b_norm_l2=b[0]**2 + b[1]**2
    
    print(a @ b)
    print(a_norm_l2 , "and", b_norm_l2)    
    return result


def exercise_12():
    """
    Exercise 12: Vectors Pointing in Opposite Directions (θ = 180°)
    When vectors point in opposite directions, the dot product is negative.
    The formula: a · b = ||a|| ||b|| cos(θ)
    When θ = 180°, cos(180°) = -1, so a · b = -||a|| ||b||
    
    Given two vectors pointing in opposite directions, verify that:
    a · b = -||a|| ||b||
    
    a = [2, 3]
    b = [-4, -6]  (b = -2a, opposite direction)
    
    Return True if the property holds, False otherwise.
    """
    a = np.array([2, 3])
    b = np.array([-4, -6])  # b = -2a, so opposite direction
    
    
    dot_product = a @ b
    magnitude_product = np.linalg.norm(a) * np.linalg.norm(b)
    cos_theta = dot_product / magnitude_product
    theta_rad = np.arccos(cos_theta)
    result = np.degrees(theta_rad)
    print(result)
    print(dot_product)
    print(magnitude_product)
    print(cos_theta)
    print(theta_rad)
    return result


def exercise_13():
    """
    Exercise 13: Calculate Angle Between Vectors
    Using the formula: a · b = ||a|| ||b|| cos(θ)
    We can solve for θ: cos(θ) = (a · b) / (||a|| ||b||)
    Then: θ = arccos((a · b) / (||a|| ||b||))
    
    Calculate the angle (in degrees) between vectors a and b.
    
    a = [1, 0]
    b = [1, 1]  (45 degrees from a)
    
    Expected result: approximately 45.0 degrees
    """
    a = np.array([1, 0])
    b = np.array([1, 1])
    
   
    dot_product = a @ b
    magnitude_product = np.linalg.norm(a) * np.linalg.norm(b)
    cos_theta = dot_product / magnitude_product
    theta_rad = np.arccos(cos_theta)
    result = np.degrees(theta_rad)
    print(result)
    print(dot_product)
    print(magnitude_product)
    print(cos_theta)
    print(theta_rad)
    
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
    print("\nExercise 1: Basic Dot Product of Vectors")
    result1 = exercise_1()
    expected1 = 32
    if result1 is not None and result1 == expected1:
        print(f"✓ Correct! Dot product = {result1}")
    else:
        print(f"✗ Incorrect. Expected: {expected1}")
        if result1 is not None:
            print(f"Got: {result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Dot Product Using np.dot()")
    result2 = exercise_2()
    expected2 = 31
    if result2 is not None and result2 == expected2:
        print(f"✓ Correct! Dot product = {result2}")
    else:
        print(f"✗ Incorrect. Expected: {expected2}")
        if result2 is not None:
            print(f"Got: {result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Dot Product Using @ Operator")
    result3 = exercise_3()
    expected3 = 70
    if result3 is not None and result3 == expected3:
        print(f"✓ Correct! Dot product = {result3}")
    else:
        print(f"✗ Incorrect. Expected: {expected3}")
        if result3 is not None:
            print(f"Got: {result3}")
    
    # Test Exercise 4
    print("\nExercise 4: Dot Product Property - Commutative")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! a · b = b · a")
    else:
        print("✗ Incorrect. Dot product should be commutative.")
    
    # Test Exercise 5
    print("\nExercise 5: Dot Product Property - Distributive")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! a · (b + c) = a · b + a · c")
    else:
        print("✗ Incorrect. The distributive property should hold.")
    
    # Test Exercise 6
    print("\nExercise 6: Dot Product Property - Scalar Multiplication")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! (ka) · b = k(a · b) = a · (kb)")
    else:
        print("✗ Incorrect. The scalar multiplication property should hold.")
    
    # Test Exercise 7
    print("\nExercise 7: Orthogonal Vectors")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! The vectors are orthogonal (dot product = 0)")
    else:
        print("✗ Incorrect. These vectors should be orthogonal.")
    
    # Test Exercise 8
    print("\nExercise 8: Matrix-Vector Dot Product")
    result8 = exercise_8()
    expected8 = np.array([17, 39])
    if result8 is not None and np.array_equal(result8, expected8):
        print(f"✓ Correct! Result = {result8}")
    else:
        print(f"✗ Incorrect. Expected: {expected8}")
        if result8 is not None:
            print(f"Got: {result8}")
    
    # Test Exercise 9
    print("\nExercise 9: Matrix-Matrix Dot Product")
    result9 = exercise_9()
    expected9 = np.array([[19, 22],
                          [43, 50]])
    if result9 is not None and np.array_equal(result9, expected9):
        print(f"✓ Correct! Result:\n{result9}")
    else:
        print(f"✗ Incorrect. Expected:\n{expected9}")
        if result9 is not None:
            print(f"Got:\n{result9}")
    
    # Test Exercise 10
    print("\nExercise 10: Magnitude Using Dot Product")
    result10 = exercise_10()
    expected10 = 5.0
    if result10 is not None and np.isclose(result10, expected10):
        print(f"✓ Correct! Magnitude = {result10}")
    else:
        print(f"✗ Incorrect. Expected: {expected10}")
        if result10 is not None:
            print(f"Got: {result10}")
    
    # Test Exercise 11
    print("\nExercise 11: Vectors Pointing in Same Direction (θ = 0°)")
    result11 = exercise_11()
    if result11:
        print("✓ Correct! When θ = 0°, a · b = ||a|| ||b||")
    else:
        print("✗ Incorrect. The property should hold for parallel vectors.")
    
    # Test Exercise 12
    print("\nExercise 12: Vectors Pointing in Opposite Directions (θ = 180°)")
    result12 = exercise_12()
    if result12:
        print("✓ Correct! When θ = 180°, a · b = -||a|| ||b||")
    else:
        print("✗ Incorrect. The property should hold for opposite vectors.")
    
    # Test Exercise 13
    print("\nExercise 13: Calculate Angle Between Vectors")
    result13 = exercise_13()
    expected13 = 45.0
    if result13 is not None and np.isclose(result13, expected13, atol=0.1):
        print(f"✓ Correct! Angle = {result13}°")
    else:
        print(f"✗ Incorrect. Expected: approximately {expected13}°")
        if result13 is not None:
            print(f"Got: {result13}°")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

