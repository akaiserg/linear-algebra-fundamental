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

### Matrix Determinant

The **determinant** is a scalar value computed from a square matrix that provides crucial information about the matrix's properties. For a 2×2 matrix A = [[a, b], [c, d]], the determinant is det(A) = ad - bc.

#### Example

Given a 2×2 matrix:
```
A = [[3, 1],
     [2, 4]]
```

The determinant is:
```
det(A) = 3×4 - 1×2 = 12 - 2 = 10
```

In NumPy:
```python
import numpy as np

A = np.array([[3, 1],
              [2, 4]])

det_A = np.linalg.det(A)
# Result: 10.0
```

#### Why Determinant is Useful

1. **Invertibility Test**: A matrix is invertible (has an inverse) if and only if det(A) ≠ 0. If det(A) = 0, the matrix is singular and cannot be inverted.

2. **Volume/Area Scaling**: In 2D, the absolute value of the determinant represents the area scaling factor of the linear transformation. In 3D, it represents volume scaling. This is crucial in geometry and computer graphics.

3. **Linear Independence**: Zero determinant indicates that the rows (or columns) of the matrix are linearly dependent, meaning one can be expressed as a combination of others.

4. **Solving Linear Systems**: Used in Cramer's rule for solving systems of linear equations. Also determines if a system has a unique solution (det ≠ 0) or infinite/no solutions (det = 0).

5. **Eigenvalues**: The product of all eigenvalues of a matrix equals its determinant. This connects determinants to many eigenvalue-based applications.

6. **Matrix Properties**: 
   - det(A^T) = det(A) - determinant of transpose equals original
   - det(AB) = det(A) × det(B) - determinant of product
   - det(kA) = k^n × det(A) - scalar multiplication (n is dimension)

7. **Applications**:
   - **Computer Graphics**: Determining if transformations preserve orientation, calculating areas/volumes
   - **Machine Learning**: Feature selection, checking for multicollinearity in datasets
   - **Physics**: Calculating moments, analyzing stability of systems
   - **Cryptography**: Some encryption algorithms use determinants

See the [determinant example](./determinant/determinant_example.py) for more detailed demonstrations.

### Dot Product

The **dot product** (also called scalar product or inner product) is an operation that takes two vectors and returns a scalar value. For vectors **a** = [a₁, a₂, ..., aₙ] and **b** = [b₁, b₂, ..., bₙ], the dot product is: **a · b** = a₁b₁ + a₂b₂ + ... + aₙbₙ.

#### Example

Given two vectors:
```
a = [1, 2, 3]
b = [4, 5, 6]
```

The dot product is:
```
a · b = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

In NumPy:
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)  # or a @ b
# Result: 32
```

#### Why Dot Product is Useful

1. **Geometric Interpretation**: The dot product equals **a · b = ||a|| ||b|| cos(θ)**, where θ is the angle between vectors. This allows us to:
   - Find angles between vectors
   - Determine if vectors are perpendicular (orthogonal when dot product = 0)
   - Calculate projections of one vector onto another

2. **Vector Magnitude**: The length (magnitude) of a vector can be computed as ||v|| = √(v · v), which is fundamental in distance calculations.

3. **Linear Transformations**: Matrix-vector multiplication (A @ v) is computed using dot products - each row of the matrix is dotted with the vector.

4. **Machine Learning & Data Science**:
   - **Neural Networks**: Computing weighted sums in neurons (activation = weights · inputs + bias)
   - **Similarity Measures**: Cosine similarity uses dot product to measure how similar two vectors are
   - **Recommendation Systems**: Finding similar users/items based on dot product of feature vectors
   - **Clustering**: Distance calculations in K-means and other algorithms

5. **Computer Graphics**:
   - **Lighting Calculations**: Computing how much light hits a surface (dot product of normal vector and light direction)
   - **Visibility Testing**: Determining if objects face the camera
   - **Shading**: Phong and Blinn-Phong shading models use dot products

6. **Signal Processing**: Correlation between signals, filtering operations, and feature extraction.

7. **Physics**: 
   - **Work**: Work done by a force is the dot product of force and displacement vectors
   - **Energy**: Potential energy calculations
   - **Torque**: Angular momentum calculations

8. **Optimization**: Many optimization algorithms (gradient descent, etc.) rely on dot products to compute gradients and update parameters.

See the [dot product example](./dot_product/dot_product_example.py) for more detailed demonstrations.

### Euclidean Norm

The **Euclidean norm** (also called L2 norm) measures the length or magnitude of a vector. For a vector **v** = [v₁, v₂, ..., vₙ], the Euclidean norm is: **||v||** = √(v₁² + v₂² + ... + vₙ²).

#### Example

Given a vector:
```
v = [3, 4]
```

The Euclidean norm is:
```
||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5
```

In NumPy:
```python
import numpy as np

v = np.array([3, 4])
norm = np.linalg.norm(v)
# Result: 5.0
```

#### Why Euclidean Norm is Useful

1. **Distance Measurement**: The Euclidean norm represents the distance from the origin to a point. The distance between two points **a** and **b** is **||a - b||**, which is fundamental in geometry, clustering algorithms, and spatial analysis.

2. **Vector Magnitude**: Provides a measure of the "size" or "length" of a vector, essential for understanding vector quantities in physics and engineering.

3. **Unit Vectors**: Normalizing a vector (dividing by its norm) creates a unit vector with length 1, preserving direction but removing magnitude. This is crucial in computer graphics, physics, and machine learning.

4. **Machine Learning & Data Science**:
   - **L2 Regularization**: Penalizes large weights using the squared Euclidean norm
   - **Loss Functions**: Mean Squared Error (MSE) uses squared Euclidean norm
   - **K-means Clustering**: Uses Euclidean distance to assign points to clusters
   - **Feature Normalization**: Scaling features to unit vectors for better model performance
   - **Similarity Measures**: Many similarity metrics are based on Euclidean distance

5. **Computer Graphics**:
   - **Normalizing Direction Vectors**: Creating unit vectors for lighting calculations
   - **Distance Calculations**: Determining object positions and collisions
   - **Rendering**: Normalizing surface normals for shading

6. **Optimization**: 
   - **Convergence Criteria**: Checking if gradient norm is below threshold
   - **Gradient Descent**: Step sizes often depend on norm calculations
   - **Regularization**: L2 regularization prevents overfitting

7. **Physics**: Representing magnitudes of physical quantities like force, velocity, acceleration, and electric fields.

8. **Matrix Norms**: The Frobenius norm (treating matrix as vector) is used in matrix analysis, optimization, and machine learning (e.g., matrix factorization).

See the [Euclidean norm example](./euclidean_norm/euclidean_norm_example.py) for more detailed demonstrations.

### Linear Combination, Basis, Span, and Rank

These fundamental concepts describe how vectors relate to each other and the spaces they create.

#### Key Concepts

1. **Linear Combination**: A vector formed by scaling and adding other vectors: **c₁v₁ + c₂v₂ + ... + cₙvₙ**

2. **Linear Independence**: Vectors are linearly independent if no vector can be written as a linear combination of the others. Check by: `rank(matrix) = number of vectors`

3. **Span**: The set of all possible linear combinations of vectors. Represents all vectors that can be "reached" using those vectors.

4. **Basis**: A set of vectors that are linearly independent and span the entire space. The standard basis for ℝⁿ has vectors with 1 in one position and 0 elsewhere.

5. **Rank**: The dimension of the column space (span of columns), equal to the number of linearly independent columns.

#### Example

```python
import numpy as np

# Linear combination
v1 = np.array([1, 0])
v2 = np.array([0, 1])
result = 2*v1 + 3*v2  # [2, 3]

# Check linear independence
A = np.column_stack([v1, v2])
rank = np.linalg.matrix_rank(A)  # 2
is_independent = (rank == A.shape[1])  # True

# Matrix rank
A = np.array([[1, 2, 3],
              [0, 1, 2],
              [0, 0, 1]])
rank = np.linalg.matrix_rank(A)  # 3
```

#### Why These Concepts Are Useful

1. **Understanding Vector Spaces**: Basis and span define the structure and dimensionality of vector spaces, fundamental to all linear algebra.

2. **Solving Linear Systems**: Rank determines if systems have unique solutions, infinite solutions, or no solutions. If `rank(A) = rank([A|b])`, the system has solutions.

3. **Dimensionality Analysis**: Rank gives the true dimension of the space spanned by columns, revealing redundancy in data.

4. **Machine Learning & Data Science**:
   - **Feature Space**: Understanding which features are redundant (linearly dependent)
   - **Dimensionality Reduction**: PCA uses rank to find lower-dimensional representations
   - **Overfitting**: Rank relates to model complexity and capacity
   - **Feature Selection**: Identifying linearly independent features

5. **Data Analysis**: 
   - **Multicollinearity**: Detecting when features are linearly dependent
   - **Data Compression**: Rank indicates minimum dimensions needed to represent data
   - **Signal Processing**: Basis functions for representing signals efficiently

6. **Computer Graphics**: Basis vectors define coordinate systems, transformations, and rotations.

7. **Optimization**: Rank constraints in optimization problems, understanding feasible regions.

8. **Cryptography**: Basis vectors used in lattice-based cryptography.

See the [linear combination, basis, span, and rank example](./linear_combination_basis_span_rank/linear_combination_basis_span_rank_example.py) for more detailed demonstrations.
