# Eigenvalues and Eigenvectors

**Eigenvalues** and **eigenvectors** are fundamental concepts in linear algebra that reveal important properties of linear transformations.

## Definition

For a square matrix **A**, an **eigenvector** **v** (nonzero vector) and its corresponding **eigenvalue** λ satisfy:

```
Av = λv
```

This means: when matrix **A** acts on eigenvector **v**, the result is simply a scalar multiple of **v** (scaled by λ).

## Key Properties

1. **Sum of eigenvalues = Trace of matrix**: Σλᵢ = trace(A)
2. **Product of eigenvalues = Determinant**: Πλᵢ = det(A)
3. **A and A^T have the same eigenvalues** (but different eigenvectors in general)
4. **Diagonal matrices**: Eigenvalues are the diagonal elements
5. **Identity matrix**: All eigenvalues are 1
6. **Symmetric matrices**: Always have real eigenvalues
7. **Power property**: If λ is eigenvalue of A, then λⁿ is eigenvalue of Aⁿ

## Eigenvalue Decomposition

A matrix can be decomposed as:

```
A = PΛP⁻¹
```

where:
- **P** is the matrix of eigenvectors (as columns)
- **Λ** (Lambda) is the diagonal matrix of eigenvalues
- **P⁻¹** is the inverse of P

## In NumPy

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Note: eigenvectors are returned as columns of the matrix
# eigenvalues[i] corresponds to eigenvectors[:, i]
```

## Why Eigenvalues and Eigenvectors are Important

1. **Principal Component Analysis (PCA)**: Finding directions of maximum variance in data
2. **Google PageRank**: Finding the dominant eigenvector of the web graph
3. **Vibration Analysis**: Natural frequencies and modes of mechanical systems
4. **Quantum Mechanics**: Energy levels and quantum states
5. **Image Processing**: Compression, feature extraction
6. **Machine Learning**: 
   - Dimensionality reduction
   - Clustering algorithms
   - Recommender systems
7. **Differential Equations**: Solving systems of linear differential equations
8. **Stability Analysis**: Determining if systems are stable or unstable
9. **Graph Theory**: Finding communities, centrality measures
10. **Data Science**: Understanding data structure and relationships

## Geometric Interpretation

- **Eigenvectors** point in directions that are preserved by the linear transformation
- **Eigenvalues** tell us how much vectors in those directions are stretched or compressed
- If λ > 1: vectors are stretched
- If 0 < λ < 1: vectors are compressed
- If λ < 0: vectors are flipped and scaled

## Running the Examples

```bash
python eigenvalues_example.py
python eigenvalues_exercise.py
```

