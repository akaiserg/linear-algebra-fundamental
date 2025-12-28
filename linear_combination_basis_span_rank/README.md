# Linear Combination, Basis, Span, and Rank

These are fundamental concepts in linear algebra that describe how vectors relate to each other and the spaces they create.

## Linear Combination

A **linear combination** of vectors **v₁, v₂, ..., vₙ** is:

```
c₁v₁ + c₂v₂ + ... + cₙvₙ
```

where **c₁, c₂, ..., cₙ** are scalars (coefficients).

## Linear Independence

Vectors are **linearly independent** if no vector can be written as a linear combination of the others. Otherwise, they are **linearly dependent**.

**Check**: Create a matrix with vectors as columns. If `rank = number of vectors`, they are independent.

```python
A = np.column_stack([v1, v2, v3])
rank = np.linalg.matrix_rank(A)
is_independent = (rank == A.shape[1])
```

## Span

The **span** of a set of vectors is the set of all possible linear combinations of those vectors. It represents all vectors that can be "reached" using those vectors.

**Check if vector w is in span**: Solve `A*x = w` where A has the spanning vectors as columns.

## Basis

A **basis** for a vector space is a set of vectors that:
1. Are **linearly independent**
2. **Span** the entire space

The **standard basis** for ℝⁿ consists of vectors with 1 in one position and 0 elsewhere:
- ℝ²: **e₁** = [1, 0], **e₂** = [0, 1]
- ℝ³: **e₁** = [1, 0, 0], **e₂** = [0, 1, 0], **e₃** = [0, 0, 1]

## Rank

The **rank** of a matrix is:
- The dimension of the **column space** (span of columns)
- The number of **linearly independent columns**
- The number of **linearly independent rows**

```python
rank = np.linalg.matrix_rank(A)
```

## In NumPy

```python
import numpy as np

# Linear combination
result = 2*v1 + 3*v2 - v3

# Check linear independence
A = np.column_stack([v1, v2, v3])
rank = np.linalg.matrix_rank(A)
is_independent = (rank == A.shape[1])

# Check if vector is in span
A = np.column_stack([v1, v2])
coefficients = np.linalg.solve(A, w)  # Solves A*x = w

# Matrix rank
rank = np.linalg.matrix_rank(A)
```

## Why These Concepts Are Important

1. **Understanding Vector Spaces**: Basis and span define the structure of vector spaces
2. **Solving Linear Systems**: Rank determines if systems have solutions
3. **Dimensionality**: Rank gives the dimension of the space spanned by columns
4. **Machine Learning**:
   - **Feature Space**: Understanding the span of feature vectors
   - **Dimensionality Reduction**: Using rank to find lower-dimensional representations
   - **Overfitting**: Rank relates to model complexity
5. **Data Analysis**: Rank indicates the true dimensionality of data
6. **Computer Graphics**: Basis vectors define coordinate systems
7. **Signal Processing**: Basis functions for signal representation

## Running the Examples

```bash
python linear_combination_basis_span_rank_example.py
python linear_combination_basis_span_rank_exercise.py
```

