"""
Singular Value Decomposition (SVD) Exercise

Complete the functions below to practice SVD operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic SVD Decomposition
    Perform SVD on a matrix and return the three components: U, S, V.
    
    A = [[1, 2],
         [3, 4]]
    
    Return a tuple (U, S, V) where:
    - U: left singular vectors
    - S: singular values (as a 1D array)
    - V: right singular vectors (V^T in some conventions)
    """
    A = np.array([[1, 2],
                  [3, 4]])
    
   
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return (U, S, Vt)

def exercise_2():
    """
    Exercise 2: Reconstruct Matrix from SVD
    Given the SVD components U, S, V, reconstruct the original matrix A.
    
    The reconstruction formula: A = U @ diag(S) @ V^T
    
    A = [[1, 2],
         [3, 4]]
    
    Return the reconstructed matrix.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_matrix = np.diag(S)
    return U @ S_matrix @ Vt

def exercise_3():
    """
    Exercise 3: Singular Values
    Extract and return only the singular values from SVD.
    
    A = [[2, 0],
         [0, 3]]
    
    Expected result: array with singular values (should be [3, 2] in descending order)
    """
    A = np.array([[2, 0],
                  [0, 3]])
    
   
    S = np.linalg.svd(A, compute_uv=False)
    return S

def exercise_4():
    """
    Exercise 4: Rank of Matrix Using SVD
    The rank of a matrix equals the number of non-zero singular values.
    
    A = [[1, 2, 3],
         [2, 4, 6],
         [3, 6, 9]]  # Rank 1 (rows are multiples of [1, 2, 3])
    
    Return the rank of the matrix using SVD.
    """
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    
  
    S = np.linalg.svd(A, compute_uv=False)
    return np.sum(S > 1e-10)

def exercise_5():
    """
    Exercise 5: Low-Rank Approximation
    Create a rank-k approximation of a matrix using SVD.
    
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    k = 1  # Approximate with rank 1
    
    Return the rank-1 approximation of A.
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    k = 1
    
   
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    return U_k @ np.diag(S_k) @ Vt_k

def exercise_6():
    """
    Exercise 6: SVD Property - Orthogonal Matrices
    Verify that U and V from SVD are orthogonal matrices.
    A matrix Q is orthogonal if Q^T Q = I.
    
    A = [[1, 2],
         [3, 4]]
    
    Check if U^T U = I and V^T V = I.
    Return True if both are orthogonal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    left_side = U.T @ U
    right_side = np.eye(U.shape[1])
    return np.allclose(left_side, right_side)



def exercise_7():
    """
    Exercise 7: SVD of Transpose
    Verify that SVD(A^T) has the same singular values as SVD(A),
    but U and V are swapped.
    
    A = [[1, 2],
         [3, 4]]
    
    Check if singular values of A and A^T are the same.
    Return True if they are equal, False otherwise.
    """
    A = np.array([[1, 2],
                  [3, 4]])
    
  
    S_A = np.linalg.svd(A, compute_uv=False)
    S_AT = np.linalg.svd(A.T, compute_uv=False)
    return np.allclose(S_A, S_AT)

def exercise_8():
    """
    Exercise 8: SVD for Square Matrix
    For a square matrix, verify that the number of singular values
    equals the matrix dimension.
    
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    
    Return the number of singular values.
    """
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    
    S = np.linalg.svd(A, compute_uv=False)
    return len(S)

def exercise_9():
    """
    Exercise 9: SVD for Rectangular Matrix
    For an m×n matrix with full_matrices=False, SVD returns:
    - U: m×min(m,n) matrix
    - S: min(m,n) singular values
    - V^T: min(m,n)×n matrix
    
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8]]  # 2×4 matrix
    
    Return the shapes of U, S, and V^T as a tuple: (U_shape, S_length, Vt_shape)
    """
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
    
   
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return (U.shape, len(S), Vt.shape)

def exercise_10():
    """
    Exercise 10: Compression Using SVD
    Calculate the compression ratio when using rank-k approximation.
    
    Original matrix: m×n
    Rank-k approximation storage:
    - U: m×k
    - S: k singular values
    - V: n×k
    
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]  # 3×4 = 12 elements
    k = 2
    
    Return the compression ratio (original_size / compressed_size)
    Original size = m × n
    Compressed size = m×k + k + n×k
    """
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    k = 2
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    m, n = A.shape
    original_size = m * n
    compressed_size = m * k + k + n * k
    return original_size / compressed_size


# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Basic SVD Decomposition")
    result1 = exercise_1()
    if result1 is not None and len(result1) == 3:
        U, S, V = result1
        print(f"✓ Correct! Got U shape: {U.shape}, S length: {len(S)}, V shape: {V.shape}")
    else:
        print("✗ Incorrect. Expected tuple (U, S, V)")
    
    # Test Exercise 2
    print("\nExercise 2: Reconstruct Matrix from SVD")
    result2 = exercise_2()
    A = np.array([[1, 2],
                  [3, 4]])
    if result2 is not None and np.allclose(result2, A):
        print(f"✓ Correct! Reconstructed matrix matches original")
    else:
        print("✗ Incorrect. Reconstructed matrix should match original")
        if result2 is not None:
            print(f"Got:\n{result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Singular Values")
    result3 = exercise_3()
    if result3 is not None:
        # Singular values should be [3, 2] in descending order
        expected = np.array([3.0, 2.0])
        if np.allclose(np.sort(result3)[::-1], expected):
            print(f"✓ Correct! Singular values: {result3}")
        else:
            print(f"✗ Incorrect. Expected approximately [3, 2], got {result3}")
    else:
        print("✗ Incorrect. Expected array of singular values")
    
    # Test Exercise 4
    print("\nExercise 4: Rank of Matrix Using SVD")
    result4 = exercise_4()
    if result4 == 1:
        print(f"✓ Correct! Rank = {result4}")
    else:
        print(f"✗ Incorrect. Expected rank = 1, got {result4}")
    
    # Test Exercise 5
    print("\nExercise 5: Low-Rank Approximation")
    result5 = exercise_5()
    if result5 is not None:
        # Check if result has correct shape
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        if result5.shape == A.shape:
            print(f"✓ Correct! Rank-1 approximation shape: {result5.shape}")
        else:
            print(f"✗ Incorrect. Expected shape {A.shape}, got {result5.shape}")
    else:
        print("✗ Incorrect. Expected rank-1 approximation matrix")
    
    # Test Exercise 6
    print("\nExercise 6: SVD Property - Orthogonal Matrices")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! U and V are orthogonal matrices")
    else:
        print("✗ Incorrect. U and V should be orthogonal")
    
    # Test Exercise 7
    print("\nExercise 7: SVD of Transpose")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! Singular values of A and A^T are the same")
    else:
        print("✗ Incorrect. Singular values should be the same")
    
    # Test Exercise 8
    print("\nExercise 8: SVD for Square Matrix")
    result8 = exercise_8()
    if result8 == 3:
        print(f"✓ Correct! Number of singular values = {result8}")
    else:
        print(f"✗ Incorrect. Expected 3 singular values, got {result8}")
    
    # Test Exercise 9
    print("\nExercise 9: SVD for Rectangular Matrix")
    result9 = exercise_9()
    expected9 = ((2, 2), 2, (2, 4))  # U: 2×2, S: 2 values, V^T: 2×4 (with full_matrices=False)
    if result9 == expected9:
        print(f"✓ Correct! Shapes: U{result9[0]}, S length {result9[1]}, V^T{result9[2]}")
    else:
        print(f"✗ Incorrect. Expected {expected9}, got {result9}")
    
    # Test Exercise 10
    print("\nExercise 10: Compression Using SVD")
    result10 = exercise_10()
    # Original: 3×4 = 12, Compressed: 3×2 + 2 + 4×2 = 6 + 2 + 8 = 16
    # Wait, that's wrong. Let me recalculate:
    # Original: 3×4 = 12 elements
    # Compressed: U (3×2) + S (2) + V^T (2×4) = 6 + 2 + 8 = 16
    # Actually V is n×n = 4×4, but we only need first k columns
    # So: U (3×2) + S (2) + V first k columns (4×2) = 6 + 2 + 8 = 16
    # Ratio = 12/16 = 0.75 (compression actually increases size for small k!)
    # But for larger matrices, compression helps
    if result10 is not None:
        print(f"✓ Got compression ratio: {result10:.4f}")
        print("  Note: For small matrices, compression may not reduce size")
    else:
        print("✗ Incorrect. Expected compression ratio (float)")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()

