# Identity Matrix

The **identity matrix** is a square matrix with 1s on the main diagonal and 0s everywhere else. It acts as the multiplicative identity for matrices, meaning that multiplying any matrix by the identity matrix leaves it unchanged.

## Definition

An n×n identity matrix I has the form:

```
I = [[1, 0, 0, ..., 0],
     [0, 1, 0, ..., 0],
     [0, 0, 1, ..., 0],
     ...
     [0, 0, 0, ..., 1]]
```

## Key Properties

1. **Multiplicative Identity**: A @ I = I @ A = A for any matrix A
2. **Vector Preservation**: I @ v = v for any vector v
3. **Symmetric**: I^T = I (transpose equals itself)
4. **Determinant**: det(I) = 1
5. **Power Property**: I^n = I for any positive integer n
6. **Inverse**: I^(-1) = I (identity is its own inverse)

## In NumPy

```python
import numpy as np

# Method 1: np.eye(n)
I = np.eye(3)  # Creates 3x3 identity matrix

# Method 2: np.identity(n)
I = np.identity(3)  # Creates 3x3 identity matrix

# Scalar multiple
kI = 3 * np.eye(3)  # Creates [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
```

## Why Identity Matrix is Important

1. **Matrix Operations**: Essential in matrix multiplication, serving as the "1" for matrices
2. **Solving Linear Systems**: Used in Gaussian elimination and finding matrix inverses
3. **Inverse Relationship**: A @ A^(-1) = A^(-1) @ A = I defines the inverse
4. **Linear Transformations**: Represents the "do nothing" transformation (no change)
5. **Eigenvalues**: Identity matrix has eigenvalue 1 with multiplicity n
6. **Matrix Decomposition**: Used in various matrix factorization techniques
7. **Computer Graphics**: Represents no transformation in coordinate systems
8. **Machine Learning**: Used in regularization (e.g., (A^T A + λI)^(-1))

## Relationship with Inverse

For an invertible matrix A:
- A @ A^(-1) = I
- A^(-1) @ A = I

This is the defining property of matrix inverses.

## Running the Examples

```bash
python identity_matrix_example.py
python identity_matrix_exercise.py
```

