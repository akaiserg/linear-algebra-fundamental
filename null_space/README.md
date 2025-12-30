# Null Space, Nullity, and Kernel

The **null space** (also called **kernel**) of a matrix A is the set of all vectors **x** such that **Ax = 0**. The **nullity** is the dimension of the null space.

## Definition

For a matrix A (m×n), the null space is:

```
Null(A) = {x ∈ ℝⁿ : Ax = 0}
```

- **Null space**: The set of all solutions to Ax = 0
- **Nullity**: The dimension of the null space (number of basis vectors)
- **Kernel**: Another name for null space (used in functional analysis)

## Key Properties

1. **Rank-Nullity Theorem**: `rank(A) + nullity(A) = n` (number of columns)
2. **Full Rank**: If `rank(A) = n`, then `nullity(A) = 0` (only zero vector in null space)
3. **Linear Dependence**: If `nullity(A) > 0`, columns of A are linearly dependent
4. **Subspace**: Null space is always a vector subspace

## In NumPy/SciPy

```python
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 2],
              [2, 4]])

# Find null space
null_space_vectors = null_space(A)

# Calculate nullity
nullity = null_space_vectors.shape[1]

# Alternative: nullity = number of columns - rank
nullity = A.shape[1] - np.linalg.matrix_rank(A)
```

## Finding Null Space

### Method 1: Using SciPy
```python
from scipy.linalg import null_space
null_space_vectors = null_space(A)
```

### Method 2: Using SVD
```python
U, s, Vt = np.linalg.svd(A, full_matrices=True)
# Columns of V corresponding to zero singular values form null space
```

### Method 3: Solving Ax = 0
Solve the homogeneous system of equations Ax = 0.

## Why Null Space is Important

1. **Solving Linear Systems**: Determines if Ax = b has unique solutions
   - If nullity > 0, there are infinitely many solutions (if solution exists)
   - If nullity = 0, solution is unique (if it exists)

2. **Linear Independence**: Nullity > 0 indicates linearly dependent columns

3. **Eigenvalue Problems**: Null space of (A - λI) gives eigenvectors

4. **Machine Learning**:
   - **Regularization**: Understanding overfitting through null space
   - **Feature Selection**: Identifying redundant features
   - **Dimensionality Reduction**: PCA uses null space concepts

5. **Signal Processing**: Finding signals that map to zero (null space filtering)

6. **Control Theory**: Analyzing system controllability and observability

7. **Computer Graphics**: Finding transformations that map to zero

## Relationship to Other Concepts

- **Column Space**: Null space of A^T is orthogonal to column space of A
- **Row Space**: Null space of A is orthogonal to row space of A
- **Rank**: `rank(A) = number of columns - nullity(A)`

## Running the Examples

```bash
python null_space_example.py
python null_space_exercise.py
```

