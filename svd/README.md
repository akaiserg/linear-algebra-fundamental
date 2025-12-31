# Singular Value Decomposition (SVD)

**Singular Value Decomposition (SVD)** is a fundamental matrix factorization technique that decomposes any matrix into three components, revealing its essential structure.

## Definition

For any matrix **A** (m×n), SVD decomposes it as:

```
A = U Σ V^T
```

where:
- **U**: m×m orthogonal matrix (left singular vectors)
- **Σ**: m×n diagonal matrix with singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
- **V^T**: n×n orthogonal matrix (right singular vectors, transposed)

For computational efficiency, we often use the "economy" SVD:
- **U**: m×min(m,n)
- **Σ**: min(m,n) singular values
- **V^T**: min(m,n)×n

## Key Properties

1. **Uniqueness**: Singular values are unique (up to ordering)
2. **Orthogonality**: U and V are orthogonal matrices (U^T U = I, V^T V = I)
3. **Singular Values**: Always non-negative, in descending order
4. **Rank**: Number of non-zero singular values equals matrix rank
5. **Transpose**: SVD(A^T) has same singular values as SVD(A)

## In NumPy

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

# Full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Economy SVD (default)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Reconstruct
A_reconstructed = U @ np.diag(S) @ Vt
```

## Applications

### 1. **Low-Rank Approximation**
Keep only the k largest singular values to approximate a matrix:
```python
U_k = U[:, :k]
S_k = S[:k]
Vt_k = Vt[:k, :]
A_k = U_k @ np.diag(S_k) @ Vt_k
```

### 2. **Dimensionality Reduction**
- **PCA (Principal Component Analysis)**: SVD is the mathematical foundation
- **Image Compression**: Represent images with fewer components
- **Data Compression**: Reduce storage while preserving important information

### 3. **Matrix Rank**
Count non-zero singular values (with tolerance):
```python
rank = np.sum(S > 1e-10)
```

### 4. **Pseudoinverse**
Compute Moore-Penrose pseudoinverse:
```python
A_pinv = Vt.T @ np.diag(1/S) @ U.T
```

### 5. **Solving Linear Systems**
For overdetermined/underdetermined systems

### 6. **Noise Reduction**
Remove components with small singular values

## Why SVD is Important

1. **Universal Decomposition**: Works for any matrix (square, rectangular, singular, non-singular)

2. **Numerical Stability**: More stable than eigenvalue decomposition for non-square matrices

3. **Data Analysis**: 
   - **Recommendation Systems**: Matrix factorization (Netflix, Amazon)
   - **Latent Semantic Analysis**: Topic modeling in text
   - **Collaborative Filtering**: User-item interactions

4. **Machine Learning**:
   - **Feature Extraction**: Reduce dimensionality
   - **Regularization**: Low-rank constraints
   - **Neural Networks**: Initialization, compression

5. **Signal Processing**:
   - **Noise Reduction**: Filter out small singular values
   - **Image Processing**: Compression, denoising

6. **Scientific Computing**:
   - **Solving Least Squares**: Overdetermined systems
   - **Condition Number**: Ratio of largest to smallest singular value

## Geometric Interpretation

- **U**: Rotates/reflects input space
- **Σ**: Scales along principal axes
- **V^T**: Rotates/reflects output space

SVD finds the best orthogonal coordinate system for the matrix transformation.

## Running the Examples

```bash
python svd_example.py
python svd_exercise.py
```

