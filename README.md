# linear-algebra-fundamental
Core linear algebra concepts implemented in Python

## Setup

1. Create and activate the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the setup:
   ```bash
   python example.py
   ```

## Requirements

- Python 3.8+
- NumPy >= 1.24.0

## Concepts

### Matrix Transpose

The **transpose** of a matrix is obtained by flipping the matrix over its main diagonal, switching the row and column indices. If A is an m×n matrix, then A^T (transpose) is an n×m matrix where each element (A^T)_{ij} = A_{ji}.

#### Example

Given a matrix:
```
A = [[1, 2, 3],
     [4, 5, 6]]
```

The transpose A^T is:
```
A^T = [[1, 4],
       [2, 5],
       [3, 6]]
```

In NumPy:
```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T  # or np.transpose(A)
# Result: [[1, 4],
#          [2, 5],
#          [3, 6]]
```

#### Why Transpose is Useful

1. **Matrix Operations**: Essential for matrix multiplication when dimensions need to match. For example, to multiply A (m×n) with B (n×p), we might need B^T.

2. **Solving Linear Systems**: Used in solving systems of linear equations, especially in least squares problems where we compute (A^T A)^(-1) A^T b.

3. **Orthogonal Matrices**: A matrix Q is orthogonal if Q^T Q = I. Orthogonal matrices preserve lengths and angles, crucial in rotations and reflections.

4. **Symmetric Matrices**: A matrix is symmetric if A = A^T. Symmetric matrices have special properties (real eigenvalues, orthogonal eigenvectors) important in many applications.

5. **Data Science & Machine Learning**: 
   - Converting row vectors to column vectors and vice versa
   - Feature transformations in datasets
   - Computing covariance matrices (which are symmetric: Cov = (X^T X) / n)
   - Neural networks: backpropagation uses transpose for gradient calculations

6. **Signal Processing**: Used in filtering, transformations, and representing signals in different bases.

7. **Computer Graphics**: Transforming coordinate systems, rotating objects, and applying transformations.

See the [transpose example](./transpose/transpose_example.py) for more detailed demonstrations.
