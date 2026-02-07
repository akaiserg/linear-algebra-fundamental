"""
Principal Component Analysis (PCA) Example

PCA finds directions of maximum variance in data and projects the data onto
these directions (principal components). It is used for dimensionality
reduction and visualization.

Steps (from scratch):
1. Center the data (subtract the mean of each feature)
2. Compute the covariance matrix
3. Find eigenvalues and eigenvectors of the covariance matrix
4. Sort eigenvectors by eigenvalue (descending) → principal components
5. Project data onto the top k components
6. Explained variance = eigenvalue_i / sum(eigenvalues)
"""

import numpy as np


def demonstrate_pca():
    """Demonstrate PCA operations from scratch using NumPy"""

    print("=" * 60)
    print("Principal Component Analysis (PCA) Examples")
    print("=" * 60)

    # Example 1: Center the data
    print("\n1. Center the Data:")
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])
    print(f"Data X (samples x features):\n{X}")
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    print(f"Mean per feature: {mean}")
    print(f"Centered data:\n{X_centered}")
    print("After centering, each column has mean 0:", np.allclose(np.mean(X_centered, axis=0), 0))

    # Example 2: Covariance matrix
    print("\n2. Covariance Matrix:")
    n_samples = X_centered.shape[0]
    # Covariance: (1/(n-1)) * X_centered.T @ X_centered
    cov = (X_centered.T @ X_centered) / (n_samples - 1)
    print(f"Covariance matrix (features x features):\n{cov}")
    print("Covariance matrix is symmetric:", np.allclose(cov, cov.T))

    # Example 3: Eigenvalues and eigenvectors
    print("\n3. Eigenvalues and Eigenvectors of Covariance:")
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    print(f"Eigenvalues (variances along principal directions): {eigenvalues}")
    print(f"Eigenvectors (principal components as columns):\n{eigenvectors}")

    # Example 4: Project onto first principal component
    print("\n4. Project Data onto First Principal Component:")
    pc1 = eigenvectors[:, 0:1]  # First column, keep 2D for matrix mult
    X_projected = X_centered @ pc1
    print(f"First PC (shape {pc1.shape}):\n{pc1}")
    print(f"Projected data (1D per sample):\n{X_projected.ravel()}")

    # Example 5: Explained variance ratio
    print("\n5. Explained Variance Ratio:")
    total_var = np.sum(eigenvalues)
    explained = eigenvalues / total_var
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Total variance: {total_var}")
    print(f"Explained variance ratio per component: {explained}")
    print(f"Cumulative: {np.cumsum(explained)}")

    # Example 6: Reconstruct from reduced representation
    print("\n6. Reconstruct Data from 1 PC:")
    X_reconstructed = X_projected @ pc1.T
    X_reconstructed += mean  # add mean back
    print(f"Reconstructed (using 1 PC):\n{X_reconstructed}")
    print("Reconstruction error (Frobenius norm):", np.linalg.norm(X - X_reconstructed))

    # Example 7: Full PCA function summary
    print("\n7. Summary - PCA in one flow:")
    X2 = np.array([[1, 2, 3],
                   [2, 4, 5],
                   [3, 6, 7],
                   [4, 8, 9]])
    mean2 = np.mean(X2, axis=0)
    X2_c = X2 - mean2
    cov2 = (X2_c.T @ X2_c) / (X2_c.shape[0] - 1)
    eigs, evecs = np.linalg.eig(cov2)
    idx2 = np.argsort(eigs)[::-1]
    eigs, evecs = eigs[idx2], evecs[:, idx2]
    k = 2  # keep 2 components
    components = evecs[:, :k]
    X2_projected = X2_c @ components
    print(f"Data shape: {X2.shape} -> Projected shape: {X2_projected.shape}")
    print(f"Explained variance ratio (top 2): {np.real(eigs[:k] / eigs.sum())}")


if __name__ == "__main__":
    demonstrate_pca()
