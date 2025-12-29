# Matrix Multiplication

Matrix multiplication is a fundamental operation in linear algebra that combines two matrices to produce a third matrix. Unlike element-wise multiplication, matrix multiplication follows specific rules based on dot products.

## Definition

For matrices A (m×n) and B (n×p), the product C = AB is an m×p matrix where:

```
C[i,j] = sum of (A[i,k] * B[k,j]) for k = 0 to n-1
```

In other words, element C[i,j] is the dot product of row i of A and column j of B.

## Dimension Requirements

For A @ B to be valid:
- **Number of columns in A** must equal **number of rows in B**
- If A is m×n and B is n×p, then A @ B is m×p

## Properties

1. **Associative**: (AB)C = A(BC)
2. **Distributive**: A(B + C) = AB + AC and (A + B)C = AC + BC
3. **Scalar Multiplication**: k(AB) = (kA)B = A(kB)
4. **NOT Commutative**: AB ≠ BA in general
5. **Transpose**: (AB)^T = B^T A^T (order reverses!)
6. **Identity**: AI = A and IA = A (when dimensions match)

## In NumPy

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Method 1: @ operator (recommended)
result = A @ B

# Method 2: np.dot()
result = np.dot(A, B)

# Method 3: np.matmul()
result = np.matmul(A, B)
```

## Matrix-Vector Multiplication

Matrix-vector multiplication is a special case:
- If A is m×n and v is n×1 (or n,), then A @ v is m×1 (or m,)
- Each row of A is dotted with the vector v

## Why Matrix Multiplication is Important

1. **Linear Transformations**: Representing rotations, scaling, shearing, and other transformations
2. **Solving Linear Systems**: Ax = b form
3. **Computer Graphics**: Transforming coordinates, applying transformations to objects
4. **Machine Learning**: 
   - Neural networks: forward propagation uses matrix multiplication
   - Weight updates and gradient calculations
   - Feature transformations
5. **Data Processing**: Combining datasets, feature engineering
6. **Quantum Mechanics**: Representing quantum states and operations
7. **Economics**: Input-output models, Markov chains
8. **Signal Processing**: Filtering, convolution operations

## Common Mistakes

1. **Dimension Mismatch**: Trying to multiply incompatible matrices
2. **Assuming Commutativity**: AB ≠ BA in general
3. **Confusing with Element-wise**: Using * instead of @ for matrix multiplication
4. **Transpose Order**: Forgetting that (AB)^T = B^T A^T (order reverses)

## Running the Examples

```bash
python matrix_multiplication_example.py
python matrix_multiplication_exercise.py
```

