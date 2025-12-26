# Matrix Transpose

The **transpose** of a matrix is obtained by flipping the matrix over its main diagonal, switching the row and column indices.

## Definition

If A is an m×n matrix, then the transpose of A, denoted as A^T or A', is an n×m matrix where:

```
(A^T)_{ij} = A_{ji}
```

## Properties

1. **(A^T)^T = A** - Double transpose returns the original matrix
2. **(A + B)^T = A^T + B^T** - Transpose of sum equals sum of transposes
3. **(kA)^T = kA^T** - Transpose of scalar multiple
4. **(AB)^T = B^T A^T** - Transpose of product (order reverses)

## In NumPy

- Use `.T` attribute: `matrix.T`
- Use `np.transpose()` function: `np.transpose(matrix)`

## Example

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T
# Result:
# [[1 4]
#  [2 5]
#  [3 6]]
```

## Running the Example

```bash
python transpose_example.py
```

