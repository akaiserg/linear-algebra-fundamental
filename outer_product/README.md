# Outer Product

The **outer product** (also called tensor product) of two vectors creates a matrix where each element (i, j) is the product of the i-th element of the first vector and the j-th element of the second vector.

## Definition

For two vectors **a** = [a₁, a₂, ..., aₘ] and **b** = [b₁, b₂, ..., bₙ]:

```
a ⊗ b = [[a₁b₁, a₁b₂, ..., a₁bₙ],
         [a₂b₁, a₂b₂, ..., a₂bₙ],
         ...
         [aₘb₁, aₘb₂, ..., aₘbₙ]]
```

The result is an m×n matrix where element (i, j) = aᵢ × bⱼ.

## Properties

1. **Distributive**: (a + b) ⊗ c = a ⊗ c + b ⊗ c
2. **Scalar Multiplication**: (ka) ⊗ b = k(a ⊗ b) = a ⊗ (kb)
3. **Transpose**: (a ⊗ b)^T = b ⊗ a
4. **Rank-1 Matrix**: Outer product of two non-zero vectors always produces a rank-1 matrix

## Relationship to Matrix Multiplication

The outer product can be computed as:
```
a ⊗ b = a.reshape(-1, 1) @ b.reshape(1, -1)
```

This is equivalent to multiplying a column vector by a row vector.

## In NumPy

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5])

# Method 1: np.outer()
result = np.outer(a, b)

# Method 2: Matrix multiplication
result = a.reshape(-1, 1) @ b.reshape(1, -1)
```

## Key Differences from Dot Product

| Property | Outer Product | Dot Product |
|---------|---------------|-------------|
| Input | Two vectors | Two vectors |
| Output | Matrix (m×n) | Scalar |
| Notation | a ⊗ b | a · b |
| Result | Each element is aᵢbⱼ | Sum of aᵢbᵢ |

## Why Outer Product is Important

1. **Rank-1 Matrices**: Creates rank-1 matrices, which are fundamental in matrix decompositions
2. **Matrix Factorization**: Used in SVD, QR decomposition, and other factorizations
3. **Tensor Operations**: Foundation for tensor products in higher dimensions
4. **Basis Construction**: Can create basis matrices from basis vectors
5. **Machine Learning**:
   - Feature interactions in recommendation systems
   - Low-rank approximations
   - Matrix completion problems
6. **Signal Processing**: Used in correlation matrices and covariance calculations
7. **Quantum Mechanics**: Essential in tensor product spaces

## Running the Examples

```bash
python outer_product_example.py
python outer_product_exercise.py
```

