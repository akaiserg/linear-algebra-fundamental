# Matrix Inverse

The **inverse** of a square matrix A, denoted as A^(-1), is a matrix such that:
```
A × A^(-1) = A^(-1) × A = I
```
where I is the identity matrix.

## Definition

For a matrix A to have an inverse, it must be:
- **Square** (same number of rows and columns)
- **Non-singular** (determinant ≠ 0)

If det(A) = 0, the matrix is **singular** and does not have an inverse.

## Key Properties

1. **A × A^(-1) = A^(-1) × A = I** - Definition of inverse
2. **(A^(-1))^(-1) = A** - Double inverse returns original
3. **(A^T)^(-1) = (A^(-1))^T** - Inverse of transpose equals transpose of inverse
4. **(AB)^(-1) = B^(-1) A^(-1)** - Inverse of product (order reverses!)
5. **(kA)^(-1) = (1/k) A^(-1)** - Inverse of scalar multiple (k ≠ 0)
6. **det(A^(-1)) = 1 / det(A)** - Determinant of inverse

## In NumPy

```python
import numpy as np

A = np.array([[3, 1],
              [2, 4]])

# Calculate inverse
A_inv = np.linalg.inv(A)

# Verify: A @ A_inv should equal identity
identity_check = A @ A_inv
```

## Solving Linear Systems

The inverse is used to solve systems of linear equations:
```
Ax = b  →  x = A^(-1) b
```

However, `np.linalg.solve(A, b)` is preferred as it's more numerically stable.

## Why Matrix Inverse is Important

1. **Solving Linear Systems**: Finding solutions to Ax = b
2. **Matrix Decomposition**: Used in LU, QR, and other decompositions
3. **Change of Basis**: Transforming coordinates between different bases
4. **Computer Graphics**: Transforming and rotating objects
5. **Machine Learning**: 
   - Computing covariance matrix inverses
   - Regularization techniques (ridge regression)
   - Kalman filters
6. **Cryptography**: Some encryption algorithms use matrix inverses
7. **Control Theory**: System analysis and controller design

## Important Notes

- **Not all matrices are invertible**: Only square matrices with non-zero determinant
- **Numerical stability**: For large matrices, `np.linalg.solve()` is preferred over computing the inverse
- **Computational cost**: Computing the inverse is O(n³) for n×n matrices

## Running the Examples

```bash
python inverse_matrix_example.py
python inverse_matrix_exercise.py
```

