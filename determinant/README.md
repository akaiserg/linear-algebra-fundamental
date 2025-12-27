# Matrix Determinant

The **determinant** is a scalar value that can be computed from a square matrix. It provides crucial information about the matrix's properties and behavior.

## Definition

For a 2×2 matrix:
```
A = [[a, b],
     [c, d]]

det(A) = ad - bc
```

For larger matrices, the determinant is computed using expansion by minors or other methods.

## Key Properties

1. **det(A) = det(A^T)** - Determinant of transpose equals determinant of original
2. **det(AB) = det(A) × det(B)** - Determinant of product equals product of determinants
3. **det(kA) = k^n × det(A)** - Scalar multiplication (n is matrix dimension)
4. **det(I) = 1** - Determinant of identity matrix is always 1
5. **det(A) = 0** if and only if A is singular (not invertible)

## Row Operations

- **Swapping two rows**: Multiplies determinant by -1
- **Multiplying a row by scalar k**: Multiplies determinant by k
- **Adding a multiple of one row to another**: Does not change determinant

## In NumPy

```python
import numpy as np

A = np.array([[3, 1],
              [2, 4]])

det_A = np.linalg.det(A)
```

## Why Determinant is Important

1. **Invertibility**: A matrix is invertible if and only if det(A) ≠ 0
2. **Volume/Area**: In 2D/3D, absolute value of determinant represents area/volume scaling factor
3. **Linear Independence**: Zero determinant indicates linearly dependent rows/columns
4. **Eigenvalues**: Product of eigenvalues equals determinant
5. **Solving Systems**: Used in Cramer's rule for solving linear systems

## Running the Examples

```bash
python determinant_example.py
python determinant_exercise.py
```

