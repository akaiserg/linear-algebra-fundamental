"""
Linear Combination, Basis, Span, and Rank Exercise - SOLUTIONS

This file contains the solutions to the exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Linear Combination"""
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    v3 = np.array([1, 1])
    
    # Compute: 2*v₁ + 3*v₂ - v₃
    result = 2*v1 + 3*v2 - v3
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Check Linear Independence"""
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    
    # Create matrix with vectors as columns
    A = np.column_stack([v1, v2])
    
    # Check rank - if rank equals number of columns, vectors are independent
    rank = np.linalg.matrix_rank(A)
    num_vectors = A.shape[1]
    
    result = (rank == num_vectors)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Check Linear Dependence"""
    v1 = np.array([1, 2])
    v2 = np.array([2, 4])  # v₂ = 2*v₁
    
    # Create matrix with vectors as columns
    A = np.column_stack([v1, v2])
    
    # Check rank - if rank < number of columns, vectors are dependent
    rank = np.linalg.matrix_rank(A)
    num_vectors = A.shape[1]
    
    result = (rank < num_vectors)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Standard Basis Vectors"""
    v = np.array([5, 7])
    
    # For standard basis, coefficients are just the vector components
    # v = 5*[1,0] + 7*[0,1] = [5, 7]
    result = v.copy()  # Coefficients are [5, 7]
    
    # Alternative: explicitly compute
    # e1 = np.array([1, 0])
    # e2 = np.array([0, 1])
    # result = np.array([v[0], v[1]])  # [5, 7]
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Span of Vectors"""
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


def exercise_6_solution():
    """Solution to Exercise 6: Matrix Rank"""
    A = np.array([[1, 2, 3],
                  [0, 1, 2],
                  [0, 0, 1]])
    
    # Calculate rank using np.linalg.matrix_rank()
    result = np.linalg.matrix_rank(A)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Rank of Linearly Dependent Columns"""
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
    # Calculate rank - should be 1 since columns are multiples of each other
    result = np.linalg.matrix_rank(A)
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Check if Vectors Form a Basis"""
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    # Create matrix with vectors as columns
    A = np.column_stack([v1, v2])
    
    # For ℝ², a basis must have:
    # 1. 2 linearly independent vectors
    # 2. Rank = 2 (dimension of ℝ²)
    rank = np.linalg.matrix_rank(A)
    dimension = 2  # ℝ²
    
    result = (rank == dimension and A.shape[1] == dimension)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Find Coefficients for Linear Combination"""
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    w = np.array([7, 10])
    
    # Create matrix A with v1, v2 as columns
    A = np.column_stack([v1, v2])
    
    # Solve A*x = w to find coefficients
    # This gives us x such that x[0]*v1 + x[1]*v2 = w
    result = np.linalg.solve(A, w)
    
    # Verify: result[0]*v1 + result[1]*v2 should equal w
    # verification = result[0]*v1 + result[1]*v2
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Dimension of Span"""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])
    
    # Create matrix with vectors as columns
    A = np.column_stack([v1, v2, v3])
    
    # Dimension of span = rank of matrix
    result = np.linalg.matrix_rank(A)
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:", exercise_1_solution())
    
    print("\nExercise 2 Result:", exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())

