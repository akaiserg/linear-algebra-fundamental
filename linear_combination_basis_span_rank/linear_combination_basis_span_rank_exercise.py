"""
Linear Combination, Basis, Span, and Rank Exercise

Complete the functions below to practice these fundamental linear algebra concepts.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Linear Combination
    A linear combination of vectors is: c₁v₁ + c₂v₂ + ... + cₙvₙ
    
    Compute the linear combination: 2*v₁ + 3*v₂ - v₃
    
    v₁ = [1, 0]
    v₂ = [0, 1]
    v₃ = [1, 1]
    
    Expected result: [1, 2]
    (2*[1,0] + 3*[0,1] - [1,1] = [2,0] + [0,3] - [1,1] = [1,2])
    """
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    v3 = np.array([1, 1])
    
    # TODO: Compute 2*v₁ + 3*v₂ - v₃
    result = 2*v1 + 3*v2 - v3
    result = result 
    
    return result


def exercise_2():
    """
    Exercise 2: Check Linear Independence
    Vectors are linearly independent if no vector can be written as a linear
    combination of the others. One way to check: put vectors as columns in a matrix
    and check if the rank equals the number of vectors.
    
    Check if the given vectors are linearly independent.
    
    v₁ = [1, 2]
    v₂ = [3, 4]
    
    Return True if independent, False otherwise.
    """
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    
    A = np.column_stack([v1, v2])
    rank = np.linalg.matrix_rank(A)
    print(rank)
    print(A.shape)
    result = (rank == A.shape[1])

    return result


def exercise_3():
    """
    Exercise 3: Check Linear Dependence
    Vectors are linearly dependent if at least one can be written as a linear
    combination of the others.
    
    Check if the given vectors are linearly dependent.
    
    v₁ = [1, 2]
    v₂ = [2, 4]  (v₂ = 2*v₁, so dependent)
    
    Return True if dependent, False otherwise.
    """
    v1 = np.array([1, 2])
    v2 = np.array([2, 4])
    
    A = np.column_stack([v1, v2])
    rank = np.linalg.matrix_rank(A)
    result = (rank < A.shape[1])
    print(rank)
    print(A.shape)
    
    return result


def exercise_4():
    """
    Exercise 4: Standard Basis Vectors
    The standard basis for ℝⁿ consists of vectors with 1 in one position and 0 elsewhere.
    For ℝ²: e₁ = [1, 0], e₂ = [0, 1]
    
    Express the vector v = [5, 7] as a linear combination of standard basis vectors.
    Return the coefficients [c₁, c₂] such that v = c₁*e₁ + c₂*e₂
    
    Expected result: [5, 7]
    """
    v = np.array([5, 7])
    
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    result = np.linalg.solve(np.column_stack([e1, e2]), v)
    print(result)
    print(v)
    print(e1)
    print(e2)   
    return result


def exercise_5():
    """
    Exercise 5: Span of Vectors
    The span of a set of vectors is all possible linear combinations of those vectors.
    
    Check if vector w is in the span of {v₁, v₂}.
    
    v₁ = [1, 0]
    v₂ = [0, 1]
    w = [3, 4]
    
    Return True if w is in span{v₁, v₂}, False otherwise.
    """
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    w = np.array([3, 4])
    
     # Create matrix A with v1, v2 as columns
    A = np.column_stack([v1, v2])
    
    # Check if w is in span{v₁, v₂} by solving A*x = w
    # If solution exists, w is in the span
    try:
        coefficients = np.linalg.solve(A, w)
        # Check if solution is valid (no errors means solution exists)
        result = True
    except np.linalg.LinAlgError:
        # System has no solution or infinite solutions
        # Check using rank: if rank([A|w]) = rank(A), then w is in span
        A_augmented = np.column_stack([A, w])
        rank_A = np.linalg.matrix_rank(A)
        rank_augmented = np.linalg.matrix_rank(A_augmented)
        result = (rank_A == rank_augmented)
    
    return result



def exercise_6():
    """
    Exercise 6: Matrix Rank
    The rank of a matrix is the dimension of the column space (span of columns).
    It equals the number of linearly independent columns.
    
    Calculate the rank of matrix A.
    
    A = [[1, 2, 3],
         [0, 1, 2],
         [0, 0, 1]]
    
    Expected result: 3 (all columns are linearly independent)
    """
    A = np.array([[1, 2, 3],
                  [0, 1, 2],
                  [0, 0, 1]])
    
    result = np.linalg.matrix_rank(A)
    
    return result


def exercise_7():
    """
    Exercise 7: Rank of Linearly Dependent Columns
    Calculate the rank of a matrix with linearly dependent columns.
    
    A = [[1, 2, 3],
         [2, 4, 6],
         [3, 6, 9]]
    
    Note: Column 2 = 2*Column 1, Column 3 = 3*Column 1
    Expected result: 1 (only 1 linearly independent column)
    """
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
    result = np.linalg.matrix_rank(A)
    
    return result


def exercise_8():
    """
    Exercise 8: Check if Vectors Form a Basis
    A set of vectors forms a basis if they are:
    1. Linearly independent
    2. Span the entire space
    
    For ℝ², check if {v₁, v₂} forms a basis.
    
    v₁ = [1, 0]
    v₂ = [0, 1]
    
    Return True if they form a basis, False otherwise.
    """
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    A = np.column_stack([v1, v2])
    rank = np.linalg.matrix_rank(A)
    dimension = 2  # ℝ²
    result = (rank == dimension and A.shape[1] == dimension)
    print(rank)
    print(dimension)
    print(A.shape)
    print(result)
    print(v1)
    print(v2)
    print(A)
    
    return result


def exercise_9():
    """
    Exercise 9: Find Coefficients for Linear Combination
    Given vectors v₁, v₂ and a target vector w, find coefficients c₁, c₂
    such that w = c₁*v₁ + c₂*v₂
    
    v₁ = [1, 2]
    v₂ = [3, 4]
    w = [7, 10]
    
    Expected result: [1, 2] (w = 1*v₁ + 2*v₂)
    """
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    w = np.array([7, 10])
    
    A = np.column_stack([v1, v2])
    result = np.linalg.solve(A, w)
    print(result)
    print(w)
    print(v1)
    print(v2)
    print(A)
    
    return result


def exercise_10():
    """
    Exercise 10: Dimension of Span
    The dimension of the span equals the rank of the matrix formed by the vectors.
    
    Find the dimension of span{v₁, v₂, v₃}.
    
    v₁ = [1, 0, 0]
    v₂ = [0, 1, 0]
    v₃ = [1, 1, 0]
    
    Note: v₃ = v₁ + v₂, so dimension is 2
    Expected result: 2
    """
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])
    
    A = np.column_stack([v1, v2, v3])
    result = np.linalg.matrix_rank(A)
    print(result)
    print(A)
    print(v1)
    print(v2)
    print(v3)
    
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
    print("\nExercise 1: Linear Combination")
    result1 = exercise_1()
    expected1 = np.array([1, 2])
    if result1 is not None and np.array_equal(result1, expected1):
        print(f"✓ Correct! Result = {result1}")
    else:
        print(f"✗ Incorrect. Expected: {expected1}")
        if result1 is not None:
            print(f"Got: {result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Check Linear Independence")
    result2 = exercise_2()
    if result2:
        print("✓ Correct! The vectors are linearly independent")
    else:
        print("✗ Incorrect. These vectors should be linearly independent.")
    
    # Test Exercise 3
    print("\nExercise 3: Check Linear Dependence")
    result3 = exercise_3()
    if result3:
        print("✓ Correct! The vectors are linearly dependent")
    else:
        print("✗ Incorrect. These vectors should be linearly dependent.")
    
    # Test Exercise 4
    print("\nExercise 4: Standard Basis Vectors")
    result4 = exercise_4()
    expected4 = np.array([5, 7])
    if result4 is not None and np.array_equal(result4, expected4):
        print(f"✓ Correct! Coefficients = {result4}")
    else:
        print(f"✗ Incorrect. Expected: {expected4}")
        if result4 is not None:
            print(f"Got: {result4}")
    
    # Test Exercise 5
    print("\nExercise 5: Span of Vectors")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! w is in span{v₁, v₂}")
    else:
        print("✗ Incorrect. w should be in span{v₁, v₂}.")
    
    # Test Exercise 6
    print("\nExercise 6: Matrix Rank")
    result6 = exercise_6()
    expected6 = 3
    if result6 is not None and result6 == expected6:
        print(f"✓ Correct! Rank = {result6}")
    else:
        print(f"✗ Incorrect. Expected: {expected6}")
        if result6 is not None:
            print(f"Got: {result6}")
    
    # Test Exercise 7
    print("\nExercise 7: Rank of Linearly Dependent Columns")
    result7 = exercise_7()
    expected7 = 1
    if result7 is not None and result7 == expected7:
        print(f"✓ Correct! Rank = {result7}")
    else:
        print(f"✗ Incorrect. Expected: {expected7}")
        if result7 is not None:
            print(f"Got: {result7}")
    
    # Test Exercise 8
    print("\nExercise 8: Check if Vectors Form a Basis")
    result8 = exercise_8()
    if result8:
        print("✓ Correct! The vectors form a basis")
    else:
        print("✗ Incorrect. These vectors should form a basis.")
    
    # Test Exercise 9
    print("\nExercise 9: Find Coefficients for Linear Combination")
    result9 = exercise_9()
    expected9 = np.array([1, 2])
    if result9 is not None and np.allclose(result9, expected9):
        print(f"✓ Correct! Coefficients = {result9}")
    else:
        print(f"✗ Incorrect. Expected: {expected9}")
        if result9 is not None:
            print(f"Got: {result9}")
    
    # Test Exercise 10
    print("\nExercise 10: Dimension of Span")
    result10 = exercise_10()
    expected10 = 2
    if result10 is not None and result10 == expected10:
        print(f"✓ Correct! Dimension = {result10}")
    else:
        print(f"✗ Incorrect. Expected: {expected10}")
        if result10 is not None:
            print(f"Got: {result10}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

