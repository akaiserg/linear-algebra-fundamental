"""
Principal Component Analysis (PCA) Exercise

Complete the functions below to practice PCA steps using NumPy.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Center the Data
    Subtract the mean of each feature (column) from the data so that
    each feature has mean 0.

    X = [[1, 4],
         [2, 5],
         [3, 6]]

    Expected: centered data has shape (3, 2) and each column mean is 0.
    Return the centered data matrix.
    """
    X = np.array([[1, 4],
                  [2, 5],
                  [3, 6]])

    
    mean = np.mean(X, axis=0)
    result = X - mean
    return result


def exercise_2():
    """
    Exercise 2: Covariance Matrix
    Given centered data X_centered (samples x features), compute the
    sample covariance matrix: (1/(n-1)) * X_centered.T @ X_centered

    X_centered = [[-1, -1],
                  [ 0,  0],
                  [ 1,  1]]

    Return the 2x2 covariance matrix.
    """
    X_centered = np.array([[-1, -1],
                           [0, 0],
                           [1, 1]])
    n = X_centered.shape[0]
    result = (X_centered.T @ X_centered) / (n - 1)
    return result


def exercise_3():
    """
    Exercise 3: Eigenvalues and Eigenvectors
    Compute the eigenvalues and eigenvectors of the covariance matrix.
    Return a tuple (eigenvalues, eigenvectors) where eigenvectors
    are columns of a 2x2 matrix (each column is an eigenvector).

    cov = [[1, 0.5],
           [0.5, 1]]

    Use np.linalg.eig(cov). Eigenvalues may be complex; we use real part.
    Return (eigenvalues as 1D array, eigenvectors as 2x2 array).
    """
    cov = np.array([[1, 0.5],
                    [0.5, 1]])

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return eigenvalues, eigenvectors


def exercise_4():
    """
    Exercise 4: Sort by Eigenvalue (Principal Components)
    Given eigenvalues and eigenvectors, sort them so that the first
    principal component (PC) has the largest eigenvalue.

    eigenvalues = [0.5, 1.5]  (order may vary from np.linalg.eig)
    eigenvectors = [[1, 1], [1, -1]]  (columns)

    Return (sorted_eigenvalues, sorted_eigenvectors) both as NumPy arrays.
    sorted_eigenvalues should be 1D descending; sorted_eigenvectors
    should be 2x2 with columns in descending order of eigenvalue.
    """
    eigenvalues = np.array([0.5, 1.5])
    eigenvectors = np.array([[1, 1],
                             [1, -1]])

    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors


def exercise_5():
    """
    Exercise 5: Project Data onto Principal Components
    Given centered data X_centered (n_samples x n_features) and
    the first k principal components (n_features x k), compute
    the projection: X_projected = X_centered @ components

    X_centered = [[-1, -1],
                  [ 0,  0],
                  [ 1,  1]]
    components = [[1/sqrt(2)],
                  [1/sqrt(2)]]  (first PC as column, k=1)

    Return the projected data (n_samples x k). Shape should be (3, 1).
    """
    X_centered = np.array([[-1, -1],
                           [0, 0],
                           [1, 1]])
    sqrt2 = np.sqrt(2)
    components = np.array([[1 / sqrt2],
                         [1 / sqrt2]])

    result = X_centered @ components
    return result

def exercise_6():
    """
    Exercise 6: Explained Variance Ratio
    Given eigenvalues (sorted descending), compute the fraction of
    total variance explained by each component: eigenvalue_i / sum(eigenvalues).

    eigenvalues = [3.0, 1.0, 0.5]

    Return a 1D array of length 3 with explained variance ratios
    (should sum to 1.0).
    """
    eigenvalues = np.array([3.0, 1.0, 0.5])
    total = np.sum(eigenvalues)
    result = eigenvalues / total
    return result


def exercise_7():
    """
    Exercise 7: PCA from Data Matrix
    Given data X (samples x features), perform PCA and return the
    projected data onto the first principal component only (1 PC).

    X = [[1, 2],
         [2, 4],
         [3, 6],
         [4, 8]]

    Steps: center -> covariance -> eig -> sort -> take first PC -> project.
    Return projected data as 1D array of length 4 (one value per sample).
    """
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

def exercise_8():
    """
    Exercise 8: Number of Components for Target Variance
    Given eigenvalues (sorted descending) and a target fraction of
    variance to retain (e.g. 0.9), find the minimum number of components
    needed so that cumulative explained variance >= target.

    eigenvalues = [2.5, 1.0, 0.3, 0.2]
    target = 0.9

    Return the number of components (integer). For this data, cumulative
    is [2.5/4, (2.5+1)/4, ...]; find first k where cumulative >= 0.9.
    """
    eigenvalues = np.array([2.5, 1.0, 0.3, 0.2])
    target = 0.9
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)
    for k in range(1, len(eigenvalues) + 1):
        if cumulative[k - 1] >= target:
            return k
    return len(eigenvalues)


# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""

    print("=" * 60)
    print("Testing Your PCA Solutions")
    print("=" * 60)

    # Exercise 1
    print("\nExercise 1: Center the Data")
    r1 = exercise_1()
    if r1 is not None and r1.shape == (3, 2) and np.allclose(np.mean(r1, axis=0), 0):
        print("✓ Correct!")
    else:
        print("✗ Incorrect. Expected shape (3,2) and column means 0.")

    # Exercise 2
    print("\nExercise 2: Covariance Matrix")
    r2 = exercise_2()
    n = 3
    expected_cov = (np.array([[-1, -1], [0, 0], [1, 1]]).T @ np.array([[-1, -1], [0, 0], [1, 1]])) / (n - 1)
    if r2 is not None and np.allclose(r2, expected_cov):
        print("✓ Correct!")
    else:
        print("✗ Incorrect. Expected covariance matrix:", expected_cov)

    # Exercise 3
    print("\nExercise 3: Eigenvalues and Eigenvectors")
    eigs, evecs = exercise_3()
    if eigs is not None and evecs is not None and eigs.shape == (2,) and evecs.shape == (2, 2):
        # Check cov @ v ≈ λ v for first eigenpair
        v = evecs[:, 0:1]
        cov = np.array([[1, 0.5], [0.5, 1]])
        lam = np.real(eigs[0])
        if np.allclose(cov @ v, lam * v):
            print("✓ Correct!")
        else:
            print("✗ Eigenpair may be wrong.")
    else:
        print("✗ Incorrect. Return (eigenvalues 1D, eigenvectors 2x2).")

    # Exercise 4
    print("\nExercise 4: Sort by Eigenvalue")
    seigs, sevecs = exercise_4()
    if seigs is not None and np.allclose(seigs, [1.5, 0.5]) and sevecs.shape == (2, 2):
        print("✓ Correct!")
    else:
        print("✗ Incorrect. Expected sorted eigenvalues [1.5, 0.5].")

    # Exercise 5
    print("\nExercise 5: Project onto PC")
    r5 = exercise_5()
    if r5 is not None and r5.shape == (3, 1):
        # Projection of [-1,-1], [0,0], [1,1] onto [1/sqrt2, 1/sqrt2] -> [-sqrt2, 0, sqrt2]
        expected = np.array([[-np.sqrt(2)], [0], [np.sqrt(2)]])
        if np.allclose(r5, expected):
            print("✓ Correct!")
        else:
            print("✗ Incorrect. Expected approximately:", expected.ravel())
    else:
        print("✗ Incorrect. Expected shape (3, 1).")

    # Exercise 6
    print("\nExercise 6: Explained Variance Ratio")
    r6 = exercise_6()
    total = 3.0 + 1.0 + 0.5
    expected = np.array([3.0 / total, 1.0 / total, 0.5 / total])
    if r6 is not None and np.allclose(r6, expected) and np.isclose(np.sum(r6), 1.0):
        print("✓ Correct!")
    else:
        print("✗ Incorrect. Expected ratios sum to 1.")

    # Exercise 7
    print("\nExercise 7: PCA from Data Matrix")
    r7 = exercise_7()
    if r7 is not None and r7.shape == (4,) and len(r7) == 4:
        # Data is along line y=2x; first PC should capture all variance
        print("✓ Shape correct! (Values depend on your PCA implementation)")
    else:
        print("✗ Incorrect. Expected 1D array of length 4.")

    # Exercise 8
    print("\nExercise 8: Components for Target Variance")
    r8 = exercise_8()
    # 2.5/4=0.625, (2.5+1)/4=0.875, (2.5+1+0.3)/4=0.95 >= 0.9 -> k=3
    if r8 is not None and r8 == 3:
        print("✓ Correct! Need 3 components for 90% variance.")
    else:
        print("✗ Incorrect. Expected 3 components for 90% variance.")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()
