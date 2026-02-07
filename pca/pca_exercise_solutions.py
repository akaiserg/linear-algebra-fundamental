"""
Principal Component Analysis (PCA) Exercise - SOLUTIONS

Try to solve the exercises yourself before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Center the data: subtract mean per column."""
    X = np.array([[1, 4],
                  [2, 5],
                  [3, 6]])
    mean = np.mean(X, axis=0)
    result = X - mean
    return result


def exercise_2_solution():
    """Covariance matrix: (1/(n-1)) * X_centered.T @ X_centered."""
    X_centered = np.array([[-1, -1],
                           [0, 0],
                           [1, 1]])
    n = X_centered.shape[0]
    result = (X_centered.T @ X_centered) / (n - 1)
    return result


def exercise_3_solution():
    """Eigenvalues and eigenvectors of covariance matrix."""
    cov = np.array([[1, 0.5],
                    [0.5, 1]])
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return eigenvalues, eigenvectors


def exercise_4_solution():
    """Sort eigenvalues and eigenvectors in descending order of eigenvalue."""
    eigenvalues = np.array([0.5, 1.5])
    eigenvectors = np.array([[1, 1],
                             [1, -1]])
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors


def exercise_5_solution():
    """Project centered data onto principal components."""
    X_centered = np.array([[-1, -1],
                           [0, 0],
                           [1, 1]])
    sqrt2 = np.sqrt(2)
    components = np.array([[1 / sqrt2],
                           [1 / sqrt2]])
    result = X_centered @ components
    return result


def exercise_6_solution():
    """Explained variance ratio = eigenvalue_i / sum(eigenvalues)."""
    eigenvalues = np.array([3.0, 1.0, 0.5])
    total = np.sum(eigenvalues)
    result = eigenvalues / total
    return result


def exercise_7_solution():
    """Full PCA: center -> covariance -> eig -> sort -> first PC -> project."""
    X = np.array([[1, 2],
                  [2, 4],
                  [3, 6],
                  [4, 8]])
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    n = X_centered.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    pc1 = eigenvectors[:, 0:1]
    projected = X_centered @ pc1
    result = projected.ravel()
    return result


def exercise_8_solution():
    """Minimum number of components to retain target fraction of variance."""
    eigenvalues = np.array([2.5, 1.0, 0.3, 0.2])
    target = 0.9
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)
    for k in range(1, len(eigenvalues) + 1):
        if cumulative[k - 1] >= target:
            return k
    return len(eigenvalues)


if __name__ == "__main__":
    print("=" * 60)
    print("PCA Exercise Solutions - Verification")
    print("=" * 60)
    print("\nExercise 1 (centered data):\n", exercise_1_solution())
    print("\nExercise 2 (covariance):\n", exercise_2_solution())
    e, v = exercise_3_solution()
    print("\nExercise 3 eigenvalues:", e, "eigenvectors shape:", v.shape)
    print("\nExercise 4 (sorted):", exercise_4_solution())
    print("\nExercise 5 (projected):\n", exercise_5_solution())
    print("\nExercise 6 (explained variance ratio):", exercise_6_solution())
    print("\nExercise 7 (PCA 1 PC):", exercise_7_solution())
    print("\nExercise 8 (components for 90% variance):", exercise_8_solution())
