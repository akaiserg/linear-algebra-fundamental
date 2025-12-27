# Dot Product

The **dot product** (also called scalar product or inner product) is an operation that takes two vectors and returns a scalar value. It's one of the most fundamental operations in linear algebra.

## Definition

For two vectors **a** = [a₁, a₂, ..., aₙ] and **b** = [b₁, b₂, ..., bₙ]:

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

## Properties

1. **Commutative**: a · b = b · a
2. **Distributive**: a · (b + c) = a · b + a · c
3. **Scalar Multiplication**: (ka) · b = k(a · b) = a · (kb)
4. **Magnitude**: ||v|| = √(v · v)

## Geometric Interpretation

The dot product has a beautiful geometric meaning:

```
a · b = ||a|| ||b|| cos(θ)
```

where:
- ||a|| and ||b|| are the magnitudes (lengths) of the vectors
- θ is the angle between the vectors

This means:
- If θ = 90° (perpendicular): a · b = 0 (vectors are orthogonal)
- If θ = 0° (parallel, same direction): a · b = ||a|| ||b||
- If θ = 180° (parallel, opposite direction): a · b = -||a|| ||b||

## In NumPy

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Method 1: np.dot()
result = np.dot(a, b)

# Method 2: @ operator
result = a @ b

# Method 3: Element-wise multiplication then sum
result = np.sum(a * b)
```

## Matrix Operations

The dot product extends to matrix operations:

- **Matrix-Vector**: A @ v (each row of A dotted with v)
- **Matrix-Matrix**: A @ B (dot products of rows of A with columns of B)

## Why Dot Product is Important

1. **Projections**: Finding how much one vector points in the direction of another
2. **Orthogonality**: Testing if vectors are perpendicular (dot product = 0)
3. **Similarity**: Measuring how similar two vectors are (used in cosine similarity)
4. **Linear Transformations**: Matrix-vector multiplication uses dot products
5. **Machine Learning**: 
   - Computing weighted sums in neural networks
   - Feature similarity in recommendation systems
   - Distance calculations in clustering
6. **Physics**: Work done by a force, calculating angles between vectors
7. **Computer Graphics**: Lighting calculations, determining visibility

## Running the Examples

```bash
python dot_product_example.py
python dot_product_exercise.py
```

