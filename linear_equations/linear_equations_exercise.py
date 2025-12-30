"""
Linear Equations Exercise

Complete the functions below to practice solving linear equations in vector and matrix form.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Solve Simple Linear Equation (Vector Form)
    Solve the equation: 2x + 3 = 11
    
    Rearrange to: 2x = 11 - 3 = 8, so x = 4
    
    Expected result: 4.0
    """
    result = (11 - 3) / 2
    print(result)
    
    return result


def exercise_2():
    """
    Exercise 2: Solve System of Two Linear Equations (Matrix Form)
    Solve the system:
    2x + 3y = 7
    4x + 5y = 13
    
    In matrix form: Ax = b, where:
    A = [[2, 3],
         [4, 5]]
    b = [7, 13]
    x = [x, y]
    
    Expected result: [2.0, 1.0]
    """
    A = np.array([[2, 3],
                  [4, 5]])
    b = np.array([7, 13])
    
    result = np.linalg.solve(A, b)
    print(result)
    return result


def exercise_3():
    """
    Exercise 3: Solve System of Three Linear Equations
    Solve the system:
    x + 2y + 3z = 14
    2x + 3y + z = 11
    3x + y + 2z = 11
    
    Expected result: [1.0, 2.0, 3.0]
    """
    A = np.array([[1, 2, 3],
                  [2, 3, 1],
                  [3, 1, 2]])
    b = np.array([14, 11, 11])
    
    result = np.linalg.solve(A, b)
    print(result)
    
    return result


def exercise_4():
    """
    Exercise 4: Verify Solution
    Given matrix A, vector b, and solution x, verify that Ax = b.
    
    A = [[1, 2],
         [3, 4]]
    b = [5, 11]
    x = [1, 2]
    
    Return True if Ax = b, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    b = np.array([5, 11])
    x = np.array([1, 2])
    
   
    result = A @ x
    print(result)
    return np.allclose(result, b)

def exercise_5():
    """
    Exercise 5: Check if Matrix is Invertible
    A system Ax = b has a unique solution if A is invertible (det(A) ≠ 0).
    
    Check if the given matrix is invertible.
    
    A = [[1, 2],
         [2, 4]]  # Second row is 2× first row (linearly dependent)
    
    Return True if invertible, False otherwise.
    """
    A = np.array([[1, 2],
                  [2, 4]])
    
   
    result = np.linalg.det(A)
    print(result)
    return not np.isclose(result, 0.0)

def exercise_6():
    """
    Exercise 6: Solve Using Matrix Inverse
    Solve Ax = b by computing x = A^(-1) * b
    
    A = [[2, 1],
         [1, 3]]
    b = [5, 7]
    
    Expected result: [1.6, 1.8] (approximately)
    """
    A = np.array([[2, 1],
                  [1, 3]])
    b = np.array([5, 7])
    
   
    result = np.linalg.inv(A) @ b
    print(result)
    return result

def exercise_7():
    """
    Exercise 7: Overdetermined System (More Equations than Unknowns)
    Solve using least squares when system is overdetermined.
    
    A = [[1, 2],
         [3, 4],
         [5, 6]]
    b = [3, 7, 11]
    
    Use np.linalg.lstsq() or np.linalg.pinv() for least squares solution.
    """
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    b = np.array([3, 7, 11])
    
   
    result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(result, residuals, rank, s)
    return result

def exercise_8():
    """
    Exercise 8: Underdetermined System (More Unknowns than Equations)
    Solve an underdetermined system (infinite solutions).
    
    x + 2y + 3z = 6
    
    In matrix form:
    A = [[1, 2, 3]]
    b = [6]
    
    Use np.linalg.pinv() (pseudo-inverse) to find a solution.
    """
    A = np.array([[1, 2, 3]])
    b = np.array([6])
    
   
    result = np.linalg.pinv(A) @ b
    print(result)

    # Alternative: Using lstsq
    # result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return result

def exercise_9():
    """
    Exercise 9: Homogeneous System (b = 0)
    Solve the homogeneous system Ax = 0.
    
    A = [[1, 2],
         [2, 4]]
    b = [0, 0]
    
    For homogeneous systems, x = 0 is always a solution (trivial solution).
    Check if there are non-trivial solutions (null space).
    Return True if non-trivial solutions exist, False otherwise.
    """
    A = np.array([[1, 2],
                  [2, 4]])
    b = np.array([0, 0])
    
     # Non-trivial solutions exist if det(A) = 0 (A is singular)
    det_A = np.linalg.det(A)
    result = np.isclose(det_A, 0.0)
    
    # Alternative: Check if null space is non-zero
    # null_space = scipy.linalg.null_space(A)
    # result = null_space.shape[1] > 0
    
    return result


def exercise_10():
    """
    Exercise 10: Vector Form of Linear Equation
    A linear equation can be written in vector form: a · x = b
    
    Given: 3x + 4y = 10
    In vector form: [3, 4] · [x, y] = 10
    
    Find one solution [x, y] that satisfies this equation.
    (There are infinitely many solutions, find any one)
    
    Expected: Any solution where 3x + 4y = 10, e.g., [2, 1] or [0, 2.5]
    """
    a = np.array([3, 4])
    b = 10
    
   # Method 2: Pick y = 1
    result = np.array([(b - a[1]) / a[0], 1])
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
    print("\nExercise 1: Solve Simple Linear Equation")
    result1 = exercise_1()
    expected1 = 4.0
    if result1 is not None and np.isclose(result1, expected1):
        print(f"✓ Correct! x = {result1}")
    else:
        print(f"✗ Incorrect. Expected: {expected1}")
        if result1 is not None:
            print(f"Got: {result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Solve System of Two Linear Equations")
    result2 = exercise_2()
    expected2 = np.array([2.0, 1.0])
    if result2 is not None and np.allclose(result2, expected2):
        print(f"✓ Correct! Solution = {result2}")
    else:
        print(f"✗ Incorrect. Expected: {expected2}")
        if result2 is not None:
            print(f"Got: {result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Solve System of Three Linear Equations")
    result3 = exercise_3()
    expected3 = np.array([1.0, 2.0, 3.0])
    if result3 is not None and np.allclose(result3, expected3):
        print(f"✓ Correct! Solution = {result3}")
    else:
        print(f"✗ Incorrect. Expected: {expected3}")
        if result3 is not None:
            print(f"Got: {result3}")
    
    # Test Exercise 4
    print("\nExercise 4: Verify Solution")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! Ax = b")
    else:
        print("✗ Incorrect. The solution should satisfy Ax = b.")
    
    # Test Exercise 5
    print("\nExercise 5: Check if Matrix is Invertible")
    result5 = exercise_5()
    if result5 is False:  # Matrix is NOT invertible (singular)
        print("✓ Correct! Matrix is singular (not invertible)")
    else:
        print("✗ Incorrect. This matrix should be singular.")
    
    # Test Exercise 6
    print("\nExercise 6: Solve Using Matrix Inverse")
    result6 = exercise_6()
    expected6 = np.array([1.6, 1.8])
    if result6 is not None and np.allclose(result6, expected6):
        print(f"✓ Correct! Solution = {result6}")
    else:
        print(f"✗ Incorrect. Expected: {expected6}")
        if result6 is not None:
            print(f"Got: {result6}")
    
    # Test Exercise 7
    print("\nExercise 7: Overdetermined System (Least Squares)")
    result7 = exercise_7()
    if result7 is not None:
        # Verify it's a reasonable solution
        A = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([3, 7, 11])
        residual = np.linalg.norm(A @ result7 - b)
        if residual < 1.0:  # Small residual
            print(f"✓ Correct! Least squares solution = {result7}")
        else:
            print(f"✗ Solution found but residual too large: {residual}")
    else:
        print("✗ Incorrect. Expected a solution vector.")
    
    # Test Exercise 8
    print("\nExercise 8: Underdetermined System")
    result8 = exercise_8()
    if result8 is not None:
        A = np.array([[1, 2, 3]])
        b = np.array([6])
        if np.isclose(A @ result8, b):
            print(f"✓ Correct! Solution = {result8}")
        else:
            print(f"✗ Solution doesn't satisfy the equation.")
    else:
        print("✗ Incorrect. Expected a solution vector.")
    
    # Test Exercise 9
    print("\nExercise 9: Homogeneous System")
    result9 = exercise_9()
    if result9:
        print("✓ Correct! Non-trivial solutions exist (null space is non-zero)")
    else:
        print("✗ Incorrect. This system should have non-trivial solutions.")
    
    # Test Exercise 10
    print("\nExercise 10: Vector Form of Linear Equation")
    result10 = exercise_10()
    if result10 is not None:
        a = np.array([3, 4])
        b = 10
        if np.isclose(np.dot(a, result10), b):
            print(f"✓ Correct! Solution = {result10}")
            print(f"  Verification: {a} · {result10} = {np.dot(a, result10)}")
        else:
            print(f"✗ Solution doesn't satisfy the equation.")
    else:
        print("✗ Incorrect. Expected a solution vector.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

