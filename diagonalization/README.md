# Matrix Diagonalization

**Diagonalization** is the process of decomposing a square matrix A into the form:

```
A = P D P^(-1)
```

where:
- **D** is a diagonal matrix containing the eigenvalues of A
- **P** is an invertible matrix whose columns are the eigenvectors of A
- **P^(-1)** is the inverse of P

## Definition

A square matrix A is **diagonalizable** if there exists an invertible matrix P and a diagonal matrix D such that:

```
A = P D P^(-1)
```

This is equivalent to saying that A has n linearly independent eigenvectors, where n is the dimension of A.

## Key Properties

1. **A = P D P^(-1)**: The fundamental diagonalization equation
2. **A^k = P D^k P^(-1)**: Powers of A can be computed easily using diagonalization
3. **det(A) = product of eigenvalues**: Determinant equals product of diagonal entries of D
4. **trace(A) = sum of eigenvalues**: Trace equals sum of diagonal entries of D

## When is a Matrix Diagonalizable?

A matrix is diagonalizable if:
- It has n linearly independent eigenvectors (where n is the dimension)
- For symmetric matrices: **always diagonalizable** with orthogonal eigenvectors
- For matrices with distinct eigenvalues: **always diagonalizable**

A matrix is **not diagonalizable** (defective) if:
- It has repeated eigenvalues but not enough linearly independent eigenvectors
- Example: [[1, 1], [0, 1]] is not diagonalizable

## In NumPy

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Create diagonal matrix
D = np.diag(eigenvalues)

# Matrix P is the eigenvectors (as columns)
P = eigenvectors
P_inv = np.linalg.inv(P)

# Verify: A = P D P^(-1)
reconstructed = P @ D @ P_inv
```

## Why Diagonalization is Important

1. **Matrix Powers**: Computing A^k is much easier: A^k = P D^k P^(-1)
   - D^k is just raising diagonal entries to power k
   - Avoids repeated matrix multiplications

2. **Matrix Functions**: Can compute functions like e^A, sin(A) using diagonalization

3. **System of Differential Equations**: Diagonalization simplifies solving systems

4. **Principal Component Analysis (PCA)**: Uses diagonalization of covariance matrix

5. **Quantum Mechanics**: Diagonalization of Hamiltonian matrices

6. **Markov Chains**: Finding steady states using diagonalization

7. **Data Analysis**: Dimensionality reduction, feature extraction

8. **Optimization**: Understanding system behavior, stability analysis

## Applications

- **Machine Learning**: PCA for dimensionality reduction
- **Signal Processing**: Filtering, noise reduction
- **Physics**: Quantum mechanics, vibrations analysis
- **Economics**: Input-output models, growth models
- **Engineering**: Control systems, structural analysis

## Running the Examples

```bash
python diagonalization_example.py
python diagonalization_exercise.py
```

