# Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is a technique for dimensionality reduction that finds the directions of maximum variance in the data and projects the data onto these directions (principal components).

## Steps (from scratch)

1. **Center the data**: Subtract the mean of each feature so each column has mean 0.
2. **Covariance matrix**: Compute \( C = \frac{1}{n-1} X_{centered}^T X_{centered} \).
3. **Eigen decomposition**: Find eigenvalues and eigenvectors of \( C \). The eigenvectors are the principal directions; eigenvalues are the variances along those directions.
4. **Sort**: Order eigenvectors by eigenvalue (descending). The first eigenvector is the first principal component (PC1), etc.
5. **Project**: For dimensionality reduction, project centered data onto the top \( k \) components: \( X_{proj} = X_{centered} \, W \), where \( W \) is the matrix of the first \( k \) eigenvectors (as columns).
6. **Explained variance**: The fraction of total variance explained by component \( i \) is \( \lambda_i / \sum_j \lambda_j \).

## Key formulas

- **Centering**: \( X_c = X - \bar{X} \) (mean per column).
- **Covariance**: \( C = \frac{1}{n-1} X_c^T X_c \).
- **Projection**: \( Z = X_c \, W_k \) (\( W_k \) = first \( k \) eigenvectors).
- **Explained variance ratio**: \( \lambda_i / \sum_j \lambda_j \).

## Relation to SVD

PCA can be done equivalently via SVD on the centered data: \( X_c = U \Sigma V^T \). The right singular vectors \( V \) are the principal components, and the squared singular values (scaled) correspond to the eigenvalues of the covariance matrix.

## In NumPy (from scratch)

```python
import numpy as np

X = ...  # shape (n_samples, n_features)
mean = np.mean(X, axis=0)
X_centered = X - mean
cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eig(cov)
eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
# First k components
k = 2
components = eigenvectors[:, :k]
X_projected = X_centered @ components
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
```

## Why PCA is useful

1. **Dimensionality reduction**: Reduce features while retaining most of the variance.
2. **Visualization**: Project high-dimensional data to 2D/3D for plotting.
3. **Noise reduction**: Small-eigenvalue components often correspond to noise.
4. **Decorrelation**: Principal components are uncorrelated (orthogonal).
5. **Machine learning**: Preprocessing, feature extraction, and avoiding multicollinearity.

## Running the examples

```bash
python pca_example.py
python pca_exercise.py
```
