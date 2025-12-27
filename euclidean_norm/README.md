# Euclidean Norm

The **Euclidean norm** (also called L2 norm) measures the length or magnitude of a vector. It's one of the most fundamental concepts in linear algebra and geometry.

## Definition

For a vector **v** = [v₁, v₂, ..., vₙ], the Euclidean norm is:

```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

In 2D, this is the familiar distance formula: `||[x, y]|| = √(x² + y²)`

## Properties

1. **Non-negativity**: ||v|| ≥ 0, and ||v|| = 0 if and only if v is the zero vector
2. **Scalar Multiplication**: ||kv|| = |k| × ||v||
3. **Triangle Inequality**: ||a + b|| ≤ ||a|| + ||b||
4. **Dot Product Relationship**: ||v|| = √(v · v)

## In NumPy

```python
import numpy as np

v = np.array([3, 4])

# Method 1: np.linalg.norm()
norm = np.linalg.norm(v)

# Method 2: Using dot product
norm = np.sqrt(np.dot(v, v))

# Method 3: Manual calculation
norm = np.sqrt(np.sum(v**2))
```

## Unit Vectors

A **unit vector** (normalized vector) has Euclidean norm = 1. To normalize a vector:

```python
v_unit = v / np.linalg.norm(v)
```

## Distance Between Points

The Euclidean distance between two points (vectors) **a** and **b** is:

```
distance = ||a - b||
```

## Matrix Norm (Frobenius Norm)

For a matrix, the **Frobenius norm** is the square root of the sum of squares of all elements:

```
||A||_F = √(Σᵢⱼ aᵢⱼ²)
```

In NumPy, `np.linalg.norm(A)` computes the Frobenius norm for matrices.

## Why Euclidean Norm is Important

1. **Distance Measurement**: Fundamental for measuring distances in space
2. **Vector Magnitude**: Represents the "size" or "length" of a vector
3. **Normalization**: Essential for creating unit vectors (direction without magnitude)
4. **Machine Learning**:
   - **Regularization**: L2 regularization uses Euclidean norm
   - **Loss Functions**: Mean Squared Error uses squared Euclidean norm
   - **Clustering**: K-means uses Euclidean distance
   - **Feature Scaling**: Normalizing features to unit vectors
5. **Computer Graphics**: Calculating distances, normalizing direction vectors
6. **Physics**: Representing magnitudes of forces, velocities, etc.
7. **Optimization**: Many optimization algorithms use Euclidean norm for convergence criteria

## Running the Examples

```bash
python euclidean_norm_example.py
python euclidean_norm_exercise.py
```

