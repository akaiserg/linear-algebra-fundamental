"""
Eigenvalues and Eigenvectors Exercise

Complete the functions below to practice eigenvalue and eigenvector operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Eigenvalue and Eigenvector Computation
    Calculate the eigenvalues and eigenvectors of a 2x2 matrix.
    
    A = [[3, 1],
         [0, 2]]
    
    Return a tuple (eigenvalues, eigenvectors) where eigenvectors are columns.
    Expected eigenvalues: approximately [3, 2]
    """
    A = np.array([[3, 1],
                  [0, 2]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return eigenvalues, eigenvectors


def exercise_2():
    """
    Exercise 2: Verify Eigenvalue Property (Av = λv)
    Verify that for each eigenvector v and eigenvalue λ, Av = λv holds.
    
    Given matrix A, compute eigenvalues and eigenvectors, then verify
    that Av = λv for each pair.
    
    A = [[4, 2],
         [1, 3]]
    
    Return True if the property holds for all eigenvalues/eigenvectors, False otherwise.
    """
    A = np.array([[4, 2],
                  [1, 3]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    left_side = A @ eigenvectors
    right_side = eigenvalues * eigenvectors
    print(left_side)
    print(right_side)
    return np.allclose(left_side, right_side)


def exercise_3():
    """
    Exercise 3: Eigenvalues of Diagonal Matrix
    For a diagonal matrix, the eigenvalues are the diagonal elements.
    
    Given diagonal matrix D, extract the eigenvalues and verify they equal
    the diagonal elements.
    
    D = [[5, 0, 0],
         [0, 3, 0],
         [0, 0, 7]]
    
    Return True if eigenvalues equal diagonal elements, False otherwise.
    """
    D = np.array([[5, 0, 0],
                  [0, 3, 0],
                  [0, 0, 7]])
    
    eigenvalues, eigenvectors = np.linalg.eig(D)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(eigenvalues, np.diag(D))   
    


def exercise_4():
    """
    Exercise 4: Sum of Eigenvalues Equals Trace
    The sum of eigenvalues equals the trace (sum of diagonal elements) of the matrix.
    
    Given matrix A, verify that sum of eigenvalues = trace(A).
    
    A = [[2, 1],
         [1, 2]]
    
    Return True if the property holds, False otherwise.
    """
    A = np.array([[2, 1],
                  [1, 2]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(np.sum(eigenvalues), np.trace(A))    
    

def exercise_5():
    """
    Exercise 5: Product of Eigenvalues Equals Determinant
    The product of eigenvalues equals the determinant of the matrix.
    
    Given matrix A, verify that product of eigenvalues = det(A).
    
    A = [[3, 1],
         [2, 4]]
    
    Return True if the property holds, False otherwise.
    """
    A = np.array([[3, 1],
                  [2, 4]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(np.prod(eigenvalues), np.linalg.det(A))


def exercise_6():
    """
    Exercise 6: Eigenvalues of Transpose
    A matrix and its transpose have the same eigenvalues.
    
    Given matrix A, verify that eigenvalues of A equal eigenvalues of A^T.
    
    A = [[1, 3],
         [2, 4]]
    
    Return True if they are equal, False otherwise.
    """
    A = np.array([[1, 3],
                  [2, 4]])
   
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(eigenvalues, np.linalg.eig(A.T)[0])

def exercise_7():
    """
    Exercise 7: Eigenvalue Decomposition
    A matrix can be decomposed as A = PΛP⁻¹ where:
    - P is the matrix of eigenvectors (as columns)
    - Λ is the diagonal matrix of eigenvalues
    
    Given matrix A, perform the decomposition and reconstruct A.
    
    A = [[4, 1],
         [2, 3]]
    
    Return True if A can be reconstructed from its decomposition, False otherwise.
    """
    A = np.array([[4, 1],
                  [2, 3]])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(eigenvectors)
    Lambda = np.diag(eigenvalues)
    P = eigenvectors
    P_inv = np.linalg.inv(P)
    return np.allclose(A, P @ Lambda @ P_inv)


def exercise_8():
    """
    Exercise 8: Eigenvalues of Identity Matrix
    All eigenvalues of an identity matrix are 1.
    
    Create a 3x3 identity matrix and verify all eigenvalues are 1.
    
    Return True if all eigenvalues equal 1, False otherwise.
    """
   
    I = np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eig(I)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(eigenvalues, np.ones(3))

def exercise_9():
    """
    Exercise 9: Symmetric Matrix Has Real Eigenvalues
    Symmetric matrices always have real eigenvalues (no imaginary part).
    
    Given symmetric matrix S, verify that all eigenvalues are real.
    
    S = [[1, 2],
         [2, 1]]
    
    Return True if all eigenvalues are real, False otherwise.
    """
    S = np.array([[1, 2],
                  [2, 1]])
    
    eigenvalues, eigenvectors = np.linalg.eig(S)
    print(eigenvalues)
    print(eigenvectors)
    return np.allclose(eigenvalues.imag, 0)
    
    return result


def exercise_10():
    """
    Exercise 10: Power of Matrix Using Eigenvalues
    If A has eigenvalue λ with eigenvector v, then A^n has eigenvalue λ^n
    with the same eigenvector v.
    
    Given matrix A, compute A^2 and verify that if λ is an eigenvalue of A,
    then λ^2 is an eigenvalue of A^2.
    
    A = [[2, 1],
         [1, 2]]
    
    Return True if the property holds, False otherwise.
    """
    A = np.array([[2, 1],
                  [1, 2]])
    
      # Compute eigenvalues of A
    eigenvals_A, _ = np.linalg.eig(A)
    
    # Compute A^2
    A_squared = A @ A
    
    # Compute eigenvalues of A^2
    eigenvals_A2, _ = np.linalg.eig(A_squared)
    
    # Verify: if λ is eigenvalue of A, then λ^2 is eigenvalue of A^2
    # Square the eigenvalues of A
    eigenvals_A_squared = eigenvals_A ** 2
    
    # Compare (need to sort since order might differ)
    result = np.allclose(np.sort(eigenvals_A_squared), np.sort(eigenvals_A2))
    
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
    print("\nExercise 1: Basic Eigenvalue and Eigenvector Computation")
    result1 = exercise_1()
    if result1[0] is not None and result1[1] is not None:
        eigenvals, eigenvecs = result1
        expected_eigenvals = np.array([3.0, 2.0])
        if np.allclose(np.sort(eigenvals), np.sort(expected_eigenvals)):
            print(f"✓ Correct! Eigenvalues: {eigenvals}")
        else:
            print(f"✗ Incorrect. Expected eigenvalues approximately [3, 2]")
            print(f"Got: {eigenvals}")
    else:
        print("✗ Incorrect. Return eigenvalues and eigenvectors.")
    
    # Test Exercise 2
    print("\nExercise 2: Verify Eigenvalue Property (Av = λv)")
    result2 = exercise_2()
    if result2:
        print("✓ Correct! Av = λv holds for all eigenvalues/eigenvectors")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 3
    print("\nExercise 3: Eigenvalues of Diagonal Matrix")
    result3 = exercise_3()
    if result3:
        print("✓ Correct! Eigenvalues equal diagonal elements")
    else:
        print("✗ Incorrect. For diagonal matrices, eigenvalues = diagonal elements.")
    
    # Test Exercise 4
    print("\nExercise 4: Sum of Eigenvalues Equals Trace")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! Sum of eigenvalues = trace of matrix")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 5
    print("\nExercise 5: Product of Eigenvalues Equals Determinant")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! Product of eigenvalues = determinant")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 6
    print("\nExercise 6: Eigenvalues of Transpose")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! A and A^T have the same eigenvalues")
    else:
        print("✗ Incorrect. The property should hold.")
    
    # Test Exercise 7
    print("\nExercise 7: Eigenvalue Decomposition")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! A can be reconstructed from PΛP⁻¹")
    else:
        print("✗ Incorrect. The decomposition should work.")
    
    # Test Exercise 8
    print("\nExercise 8: Eigenvalues of Identity Matrix")
    result8 = exercise_8()
    if result8:
        print("✓ Correct! All eigenvalues of identity matrix are 1")
    else:
        print("✗ Incorrect. Identity matrix eigenvalues should all be 1.")
    
    # Test Exercise 9
    print("\nExercise 9: Symmetric Matrix Has Real Eigenvalues")
    result9 = exercise_9()
    if result9:
        print("✓ Correct! Symmetric matrices have real eigenvalues")
    else:
        print("✗ Incorrect. Symmetric matrices should have real eigenvalues.")
    
    # Test Exercise 10
    print("\nExercise 10: Power of Matrix Using Eigenvalues")
    result10 = exercise_10()
    if result10:
        print("✓ Correct! If λ is eigenvalue of A, then λ^n is eigenvalue of A^n")
    else:
        print("✗ Incorrect. The property should hold.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

