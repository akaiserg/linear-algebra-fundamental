"""
Matrix Transpose Example

The transpose of a matrix is obtained by flipping the matrix over its diagonal,
switching the row and column indices of the matrix.
"""

import numpy as np


def demonstrate_transpose():
    """Demonstrate matrix transpose operations"""
    
    print("=" * 60)
    print("Matrix Transpose Examples")
    print("=" * 60)
    
    # Example 1: Simple 2x3 matrix
    print("\n1. Transpose of a 2x3 matrix:")
    matrix_a = np.array([[1, 2, 3],
                         [4, 5, 6]])
    print(f"Original matrix (2x3):\n{matrix_a}")
    print(f"Transpose (3x2):\n{matrix_a.T}")
    print(f"Shape: {matrix_a.shape} -> {matrix_a.T.shape}")
    
    # Example 2: Square matrix
    print("\n2. Transpose of a square matrix:")
    matrix_b = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
    print(f"Original matrix (3x3):\n{matrix_b}")
    print(f"Transpose (3x3):\n{matrix_b.T}")
    
    # Example 3: Using np.transpose() function
    print("\n3. Using np.transpose() function:")
    matrix_c = np.array([[1, 4],
                         [2, 5],
                         [3, 6]])
    print(f"Original matrix (3x2):\n{matrix_c}")
    print(f"Transpose using np.transpose(): \n{np.transpose(matrix_c)}")
    print(f"Transpose using .T: \n{matrix_c.T}")
    
    # Example 4: Properties of transpose
    print("\n4. Properties of transpose:")
    print("   - (A^T)^T = A")
    print(f"   Original:\n{matrix_a}")
    print(f"   (A^T)^T:\n{(matrix_a.T).T}")
    print(f"   Are they equal? {np.array_equal(matrix_a, (matrix_a.T).T)}")
    
    # Example 5: Transpose of a vector
    print("\n5. Transpose of a vector:")
    vector = np.array([1, 2, 3, 4])
    print(f"Row vector: {vector}")
    print(f"Shape: {vector.shape}")
    print(f"Column vector (transpose):\n{vector.T}")
    print(f"Shape after transpose: {vector.T.shape}")
    print(f"Note: 1D arrays don't change with transpose in NumPy")
    
    # To get a proper column vector, reshape:
    print(f"\n   Proper column vector (using reshape):\n{vector.reshape(-1, 1)}")
    print(f"   Shape: {vector.reshape(-1, 1).shape}")


if __name__ == "__main__":
    demonstrate_transpose()

