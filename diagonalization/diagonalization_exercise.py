"""
Matrix Diagonalization Exercise

Complete the functions below to practice diagonalization operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Find Eigenvalues and Eigenvectors
    Find the eigenvalues and eigenvectors of matrix A.
    
    A = [[4, 1],
         [2, 3]]
    
    Return a tuple (eigenvalues, eigenvectors) where eigenvectors are columns.
    """
    A = np.array([[4, 1],
                  [2, 3]])
    
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return eigenvalues, eigenvectors


def exercise_2():
    """
    Exercise 2: Create Diagonal Matrix from Eigenvalues
    Create a diagonal matrix D with the eigenvalues of A on the diagonal.
    
    A = [[5, 2],
         [2, 5]]
    
    Expected D: [[7, 0],
                 [0, 3]]
    (approximately, eigenvalues are 7 and 3)
    """
    A = np.array([[5, 2],
                  [2, 5]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    print(D)
    print(eigenvectors)
    return D


def exercise_3():
    """
    Exercise 3: Verify Diagonalization A = P D P^(-1)
    Given matrix A, verify that A = P D P^(-1) where:
    - D is the diagonal matrix of eigenvalues
    - P is the matrix of eigenvectors (as columns)
    
    A = [[4, 1],
         [2, 3]]
    
    Return True if A = P D P^(-1), False otherwise.
    """
    A = np.array([[4, 1],
                  [2, 3]])
   
    eigenvalues, eigenvectors = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    return np.allclose(A, P @ D @ P_inv)

def exercise_4():
    """
    Exercise 4: Compute A^k Using Diagonalization
    Compute A^3 using diagonalization: A^k = P D^k P^(-1)
    
    A = [[2, 1],
         [1, 2]]
    
    Return A^3 computed using diagonalization.
    """
    A = np.array([[2, 1],
                  [1, 2]])
    k = 3
    
   
    eigenvalues, eigenvectors = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    return P @ np.linalg.matrix_power(D, k) @ P_inv

def exercise_5():
    """
    Exercise 5: Determinant from Eigenvalues
    Verify that det(A) = product of eigenvalues.
    
    A = [[3, 1],
         [1, 3]]
    
    Return True if det(A) equals the product of eigenvalues, False otherwise.
    """
    A = np.array([[3, 1],
                  [1, 3]])
    
   
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(np.prod(eigenvalues), np.linalg.det(A))

def exercise_6():
    """
    Exercise 6: Trace from Eigenvalues
    Verify that trace(A) = sum of eigenvalues.
    
    A = [[5, 2],
         [2, 5]]
    
    Return True if trace(A) equals the sum of eigenvalues, False otherwise.
    """
    A = np.array([[5, 2],
                  [2, 5]])
    
   
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(np.sum(eigenvalues), np.trace(A))

def exercise_7():
    """
    Exercise 7: Check if Matrix is Diagonalizable
    A matrix is diagonalizable if it has n linearly independent eigenvectors,
    where n is the dimension of the matrix.
    
    Check if matrix A is diagonalizable.
    
    A = [[1, 1],
         [0, 1]]
    
    Return True if diagonalizable, False otherwise.
    """
    A = np.array([[1, 1],
                  [0, 1]])
    
   
   # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # A matrix is diagonalizable if it has n linearly independent eigenvectors
    # where n is the dimension of the matrix
    # Check the rank of the eigenvector matrix
    rank = np.linalg.matrix_rank(eigenvectors)
    n = A.shape[0]
    
    # If rank equals dimension, we have enough linearly independent eigenvectors
    result = rank == n
    print(result)

    # Note: This matrix is NOT diagonalizable (defective matrix)
    # It has only one linearly independent eigenvector
    
    return result


def exercise_8():
    """
    Exercise 8: Diagonalize a Symmetric Matrix
    Symmetric matrices are always diagonalizable and have orthogonal eigenvectors.
    
    Diagonalize the symmetric matrix A and verify that eigenvectors are orthogonal.
    
    A = [[1, 2],
         [2, 1]]
    
    Return True if eigenvectors are orthogonal (dot product = 0), False otherwise.
    """
    A = np.array([[1, 2],
                  [2, 1]])
    
  
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(eigenvectors[:, 0] @ eigenvectors[:, 1], 0)

def exercise_9():
    """
    Exercise 9: Reconstruct Matrix from Diagonalization
    Given eigenvalues and eigenvectors, reconstruct the original matrix.
    
    Eigenvalues: [6, 1]
    Eigenvectors (as columns): [[1, 1], [1, -1]]
    
    Reconstruct the original matrix A = P D P^(-1).
    """
    eigenvalues = np.array([6, 1])
    eigenvectors = np.array([[1, 1],
                             [1, -1]])
    
   
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    return P @ D @ P_inv

def exercise_10():
    """
    Exercise 10: Power of Matrix Using Diagonalization
    Compute A^5 using diagonalization for a 3x3 matrix.
    
    A = [[2, 0, 0],
         [0, 3, 0],
         [0, 0, 4]]
    
    Return A^5.
    """
    A = np.array([[2, 0, 0],
                  [0, 3, 0],
                  [0, 0, 4]])
    k = 5
    
   
    eigenvalues, eigenvectors = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    return P @ np.linalg.matrix_power(D, k) @ P_inv

# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Find Eigenvalues and Eigenvectors")
    result1 = exercise_1()
    if result1[0] is not None and result1[1] is not None:
        eigenvalues, eigenvectors = result1
        print(f"✓ Found eigenvalues: {eigenvalues}")
        print(f"  Eigenvectors shape: {eigenvectors.shape}")
        # Verify they are correct
        A = np.array([[4, 1], [2, 3]])
        expected_eigenvalues, _ = np.linalg.eig(A)
        if np.allclose(np.sort(eigenvalues), np.sort(expected_eigenvalues)):
            print("  ✓ Eigenvalues are correct!")
        else:
            print("  ✗ Eigenvalues don't match")
    else:
        print("✗ Incorrect. Need to return (eigenvalues, eigenvectors)")
    
    # Test Exercise 2
    print("\nExercise 2: Create Diagonal Matrix from Eigenvalues")
    result2 = exercise_2()
    if result2 is not None:
        print(f"Diagonal matrix D:\n{result2}")
        A = np.array([[5, 2], [2, 5]])
        expected_eigenvalues = np.linalg.eig(A)[0]
        expected_D = np.diag(expected_eigenvalues)
        if np.allclose(result2, expected_D):
            print("✓ Correct!")
        else:
            print("✗ Incorrect. D should contain eigenvalues on diagonal")
    else:
        print("✗ Incorrect. Need to return diagonal matrix D")
    
    # Test Exercise 3
    print("\nExercise 3: Verify Diagonalization A = P D P^(-1)")
    result3 = exercise_3()
    if result3:
        print("✓ Correct! A = P D P^(-1)")
    else:
        print("✗ Incorrect. The diagonalization should hold.")
    
    # Test Exercise 4
    print("\nExercise 4: Compute A^3 Using Diagonalization")
    result4 = exercise_4()
    if result4 is not None:
        A = np.array([[2, 1], [1, 2]])
        expected = np.linalg.matrix_power(A, 3)
        if np.allclose(result4, expected):
            print(f"✓ Correct! A^3:\n{result4}")
        else:
            print(f"✗ Incorrect. Expected:\n{expected}")
            print(f"Got:\n{result4}")
    else:
        print("✗ Incorrect. Need to compute A^3")
    
    # Test Exercise 5
    print("\nExercise 5: Determinant from Eigenvalues")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! det(A) = product of eigenvalues")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 6
    print("\nExercise 6: Trace from Eigenvalues")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! trace(A) = sum of eigenvalues")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 7
    print("\nExercise 7: Check if Matrix is Diagonalizable")
    result7 = exercise_7()
    # Matrix [[1, 1], [0, 1]] is NOT diagonalizable (defective)
    if result7 is not True:
        print("✓ Correct! The matrix is not diagonalizable")
    else:
        print("✗ Incorrect. This matrix should not be diagonalizable.")
    
    # Test Exercise 8
    print("\nExercise 8: Diagonalize a Symmetric Matrix")
    result8 = exercise_8()
    if result8:
        print("✓ Correct! Eigenvectors of symmetric matrix are orthogonal")
    else:
        print("✗ Incorrect. Symmetric matrices have orthogonal eigenvectors.")
    
    # Test Exercise 9
    print("\nExercise 9: Reconstruct Matrix from Diagonalization")
    result9 = exercise_9()
    if result9 is not None:
        print(f"Reconstructed matrix A:\n{result9}")
        # Verify by checking eigenvalues
        eigenvalues_check, _ = np.linalg.eig(result9)
        if np.allclose(np.sort(eigenvalues_check), np.sort([6, 1])):
            print("✓ Correct! Matrix has the correct eigenvalues")
        else:
            print("✗ Incorrect. Check your reconstruction.")
    else:
        print("✗ Incorrect. Need to reconstruct the matrix")
    
    # Test Exercise 10
    print("\nExercise 10: Power of Matrix Using Diagonalization")
    result10 = exercise_10()
    if result10 is not None:
        A = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
        expected = np.linalg.matrix_power(A, 5)
        if np.allclose(result10, expected):
            print(f"✓ Correct! A^5:\n{result10}")
        else:
            print(f"✗ Incorrect. Expected:\n{expected}")
    else:
        print("✗ Incorrect. Need to compute A^5")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

