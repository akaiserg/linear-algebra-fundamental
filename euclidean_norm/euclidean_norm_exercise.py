"""
Euclidean Norm Exercise

Complete the functions below to practice Euclidean norm operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Euclidean Norm
    Calculate the Euclidean norm (length) of a vector.
    
    v = [3, 4]
    
    Expected result: 5.0
    (sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5)
    """
    v = np.array([3, 4])
    
    result = np.sqrt(v[0]**2 + v[1]**2)    
    
    return result


def exercise_2():
    """
    Exercise 2: Euclidean Norm Using np.linalg.norm()
    Calculate the Euclidean norm using np.linalg.norm().
    
    v = [1, 2, 2]
    
    Expected result: 3.0
    (sqrt(1² + 2² + 2²) = sqrt(1 + 4 + 4) = sqrt(9) = 3)
    """
    v = np.array([1, 2, 2])
    
    result = np.linalg.norm(v)
    
    return result


def exercise_3():
    """
    Exercise 3: Euclidean Norm Using Dot Product
    The Euclidean norm can be calculated as sqrt(v · v).
    
    v = [5, 12]
    
    Expected result: 13.0
    (sqrt(5² + 12²) = sqrt(25 + 144) = sqrt(169) = 13)
    """
    v = np.array([5, 12])
    
    result = np.sqrt(v @ v)
    
    return result


def exercise_4():
    """
    Exercise 4: Unit Vector (Normalized Vector)
    A unit vector has Euclidean norm = 1.
    Normalize a vector by dividing it by its norm: v_unit = v / ||v||
    
    Normalize the vector v to get a unit vector.
    
    v = [3, 4]
    
    Expected result: [0.6, 0.8]
    (v / ||v|| = [3, 4] / 5 = [0.6, 0.8])
    """
    v = np.array([3, 4])
    
   
    result = v / np.linalg.norm(v)
    
    return result


def exercise_5():
    """
    Exercise 5: Verify Unit Vector Has Norm 1
    After normalizing a vector, verify that its norm equals 1.
    
    v = [1, 1, 1]
    
    Return True if the normalized vector has norm 1, False otherwise.
    """
    v = np.array([1, 1, 1])
    
    
    v_normalized = v / np.linalg.norm(v)
    
    # Check if norm equals 1
    norm_normalized = np.linalg.norm(v_normalized)
    result = np.isclose(norm_normalized, 1.0)
    
    return result


def exercise_6():
    """
    Exercise 6: Distance Between Two Points
    The Euclidean distance between two points (vectors) is the norm of their difference.
    distance = ||a - b||
    
    Calculate the distance between points a and b.
    
    a = [1, 2]
    b = [4, 6]
    
    Expected result: 5.0
    (||[1,2] - [4,6]|| = ||[-3,-4]|| = sqrt(9 + 16) = 5)
    """
    a = np.array([1, 2])
    b = np.array([4, 6])
    
    result = np.linalg.norm(a - b)
    
    return result


def exercise_7():
    """
    Exercise 7: Norm Property - Scalar Multiplication
    Verify that ||kv|| = |k| × ||v||
    
    Given vector v and scalar k, verify this property.
    Return True if the property holds, False otherwise.
    """
    v = np.array([3, 4])
    k = -2
    
    
    result = np.isclose(np.linalg.norm(k * v), abs(k) * np.linalg.norm(v))
    
    return result


def exercise_8():
    """
    Exercise 8: Triangle Inequality
    The triangle inequality states: ||a + b|| ≤ ||a|| + ||b||
    
    Given vectors a and b, verify that the triangle inequality holds.
    Return True if it holds, False otherwise.
    """
    a = np.array([1, 2])
    b = np.array([3, 4])
    
    # TODO: Check if ||a + b|| ≤ ||a|| + ||b||
   
    left_side = np.linalg.norm(a + b)
    right_side = np.linalg.norm(a) + np.linalg.norm(b)
    result = left_side <= right_side
    print(left_side)
    print(right_side)
    print(result)
    
    return result


def exercise_9():
    """
    Exercise 9: Zero Vector Has Zero Norm
    The zero vector has Euclidean norm 0.
    
    Check if the zero vector has norm 0.
    Return True if norm is 0, False otherwise.
    """
    v = np.array([0, 0, 0])
    
     
    result = np.isclose(np.linalg.norm(v), 0.0)
    
    return result


def exercise_10():
    """
    Exercise 10: Norm of Matrix (Frobenius Norm)
    For a matrix, the Frobenius norm is the square root of the sum of squares of all elements.
    It's like treating the matrix as a vector and computing its Euclidean norm.
    
    A = [[1, 2],
         [3, 4]]
    
    Expected result: sqrt(1² + 2² + 3² + 4²) = sqrt(30) ≈ 5.477
    """
    A = np.array([[1, 2],
                  [3, 4]])
    
    
    result = np.linalg.norm(A)
    
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
    print("\nExercise 1: Basic Euclidean Norm")
    result1 = exercise_1()
    expected1 = 5.0
    if result1 is not None and np.isclose(result1, expected1):
        print(f"✓ Correct! Norm = {result1}")
    else:
        print(f"✗ Incorrect. Expected: {expected1}")
        if result1 is not None:
            print(f"Got: {result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Euclidean Norm Using np.linalg.norm()")
    result2 = exercise_2()
    expected2 = 3.0
    if result2 is not None and np.isclose(result2, expected2):
        print(f"✓ Correct! Norm = {result2}")
    else:
        print(f"✗ Incorrect. Expected: {expected2}")
        if result2 is not None:
            print(f"Got: {result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Euclidean Norm Using Dot Product")
    result3 = exercise_3()
    expected3 = 13.0
    if result3 is not None and np.isclose(result3, expected3):
        print(f"✓ Correct! Norm = {result3}")
    else:
        print(f"✗ Incorrect. Expected: {expected3}")
        if result3 is not None:
            print(f"Got: {result3}")
    
    # Test Exercise 4
    print("\nExercise 4: Unit Vector (Normalized Vector)")
    result4 = exercise_4()
    expected4 = np.array([0.6, 0.8])
    if result4 is not None and np.allclose(result4, expected4):
        print(f"✓ Correct! Unit vector = {result4}")
    else:
        print(f"✗ Incorrect. Expected: {expected4}")
        if result4 is not None:
            print(f"Got: {result4}")
    
    # Test Exercise 5
    print("\nExercise 5: Verify Unit Vector Has Norm 1")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! Normalized vector has norm 1")
    else:
        print("✗ Incorrect. Normalized vector should have norm 1.")
    
    # Test Exercise 6
    print("\nExercise 6: Distance Between Two Points")
    result6 = exercise_6()
    expected6 = 5.0
    if result6 is not None and np.isclose(result6, expected6):
        print(f"✓ Correct! Distance = {result6}")
    else:
        print(f"✗ Incorrect. Expected: {expected6}")
        if result6 is not None:
            print(f"Got: {result6}")
    
    # Test Exercise 7
    print("\nExercise 7: Norm Property - Scalar Multiplication")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! ||kv|| = |k| × ||v||")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 8
    print("\nExercise 8: Triangle Inequality")
    result8 = exercise_8()
    if result8:
        print("✓ Correct! Triangle inequality holds: ||a + b|| ≤ ||a|| + ||b||")
    else:
        print("✗ Incorrect. Triangle inequality should hold.")
    
    # Test Exercise 9
    print("\nExercise 9: Zero Vector Has Zero Norm")
    result9 = exercise_9()
    if result9:
        print("✓ Correct! Zero vector has norm 0")
    else:
        print("✗ Incorrect. Zero vector should have norm 0.")
    
    # Test Exercise 10
    print("\nExercise 10: Norm of Matrix (Frobenius Norm)")
    result10 = exercise_10()
    expected10 = np.sqrt(30)  # sqrt(1² + 2² + 3² + 4²) = sqrt(30)
    if result10 is not None and np.isclose(result10, expected10):
        print(f"✓ Correct! Frobenius norm = {result10:.4f}")
    else:
        print(f"✗ Incorrect. Expected: {expected10:.4f}")
        if result10 is not None:
            print(f"Got: {result10}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

