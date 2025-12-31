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
- SciPy >= 1.10.0 (for null space calculations)

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

### Matrix Inverse

The **inverse** of a square matrix A, denoted as A^(-1), is a matrix such that A × A^(-1) = A^(-1) × A = I, where I is the identity matrix. A matrix is invertible (non-singular) if and only if its determinant is not zero.

#### Example

Given a matrix:
```
A = [[3, 1],
     [2, 4]]
```

The inverse A^(-1) is:
```
A^(-1) = [[ 0.4, -0.1],
          [-0.2,  0.3]]
```

Verification: A × A^(-1) = I

In NumPy:
```python
import numpy as np

A = np.array([[3, 1],
              [2, 4]])

A_inv = np.linalg.inv(A)
# Result: [[ 0.4, -0.1],
#          [-0.2,  0.3]]

# Verify: A @ A_inv should equal identity matrix
identity_check = A @ A_inv
```

#### Why Matrix Inverse is Useful

1. **Solving Linear Systems**: The primary use is solving systems of linear equations Ax = b by computing x = A^(-1) b. However, `np.linalg.solve(A, b)` is preferred for numerical stability.

2. **Change of Basis**: Transforming coordinates between different coordinate systems or bases, essential in computer graphics and robotics.

3. **Matrix Decompositions**: Used in LU decomposition, QR decomposition, and other matrix factorization techniques that are fundamental to numerical linear algebra.

4. **Computer Graphics**:
   - **Transformations**: Inverting transformation matrices to reverse rotations, scaling, or translations
   - **Camera Systems**: Converting between world coordinates and camera coordinates
   - **Animation**: Reversing transformations for undo operations

5. **Machine Learning & Data Science**:
   - **Ridge Regression**: Regularized linear regression uses matrix inverses
   - **Covariance Matrices**: Computing precision matrices (inverse of covariance)
   - **Kalman Filters**: State estimation in control systems and tracking
   - **Gaussian Processes**: Computing posterior distributions

6. **Control Theory**: System analysis, controller design, and stability analysis of dynamic systems.

7. **Cryptography**: Some encryption algorithms use matrix inverses for encoding and decoding.

8. **Economics**: Input-output models, Leontief models, and economic equilibrium calculations.

9. **Key Properties**:
   - **(A^(-1))^(-1) = A** - Double inverse returns original
   - **(A^T)^(-1) = (A^(-1))^T** - Inverse of transpose
   - **(AB)^(-1) = B^(-1) A^(-1)** - Inverse of product (order reverses!)
   - **det(A^(-1)) = 1 / det(A)** - Determinant of inverse

10. **Important Note**: Not all matrices are invertible. A matrix must be square and have a non-zero determinant. Matrices with det(A) = 0 are called **singular** and cannot be inverted.

See the [inverse matrix example](./inverse_matrix/inverse_matrix_example.py) for more detailed demonstrations.

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

### Null Space, Nullity, and Kernel

The **null space** (also called **kernel**) of a matrix A is the set of all vectors **x** such that **Ax = 0**. The **nullity** is the dimension of the null space.

#### Example

Given a matrix:
```
A = [[1, 2],
     [2, 4]]
```

The null space consists of all vectors x such that Ax = 0:
```
x1 + 2x2 = 0  →  x1 = -2x2
```

So the null space is spanned by the vector [-2, 1] (or any scalar multiple).

In NumPy/SciPy:
```python
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 2],
              [2, 4]])

# Find null space
null_space_vectors = null_space(A)
nullity = null_space_vectors.shape[1]  # Dimension of null space
```

#### Why Null Space is Useful

1. **Solving Linear Systems**: Determines if Ax = b has unique solutions
   - If nullity > 0, there are infinitely many solutions (if solution exists)
   - If nullity = 0, solution is unique (if it exists)

2. **Rank-Nullity Theorem**: `rank(A) + nullity(A) = number of columns`. This fundamental relationship connects rank and nullity.

3. **Linear Independence**: Nullity > 0 indicates linearly dependent columns. If nullity = 0, columns are linearly independent.

4. **Eigenvalue Problems**: Null space of (A - λI) gives eigenvectors corresponding to eigenvalue λ.

5. **Machine Learning**:
   - **Regularization**: Understanding overfitting through null space analysis
   - **Feature Selection**: Identifying redundant features (columns in null space)
   - **Dimensionality Reduction**: PCA and other techniques use null space concepts

6. **Signal Processing**: Finding signals that map to zero (null space filtering), noise cancellation.

7. **Control Theory**: Analyzing system controllability and observability, understanding system behavior.

8. **Computer Graphics**: Finding transformations that map to zero, understanding geometric relationships.

9. **Data Analysis**: Identifying redundant information, understanding data structure, detecting multicollinearity.

See the [null space example](./null_space/null_space_example.py) for more detailed demonstrations.

### Linear Equations

Linear equations are fundamental mathematical expressions that can be represented in both **vector form** and **matrix form**. A system of linear equations can be written as **Ax = b**, where A is the coefficient matrix, x is the variable vector, and b is the constant vector.

#### Example

**Vector Form:**
```
3x + 4y = 10
```
In vector form: `[3, 4] · [x, y] = 10`

**Matrix Form:**
```
2x + 3y = 7
4x + 5y = 13
```

In matrix form:
```python
A = [[2, 3],     x = [x,     b = [7,
     [4, 5]]         y]          13]
```

Solving in NumPy:
```python
import numpy as np

A = np.array([[2, 3],
              [4, 5]])
b = np.array([7, 13])

x = np.linalg.solve(A, b)
# Result: [2.0, 1.0]
```

#### Why Linear Equations are Useful

1. **Universal Problem Solver**: Many real-world problems can be modeled as systems of linear equations, from engineering calculations to economic models.

2. **Types of Systems**:
   - **Unique Solution**: When det(A) ≠ 0, use `np.linalg.solve(A, b)`
   - **Overdetermined**: More equations than unknowns - use least squares (`np.linalg.lstsq()`)
   - **Underdetermined**: More unknowns than equations - use pseudo-inverse (`np.linalg.pinv()`)
   - **Homogeneous**: Ax = 0 - always has trivial solution x = 0

3. **Engineering Applications**:
   - **Circuit Analysis**: Solving for currents and voltages in electrical circuits
   - **Structural Mechanics**: Finding forces and stresses in structures
   - **Fluid Dynamics**: Modeling flow and pressure

4. **Machine Learning & Data Science**:
   - **Linear Regression**: Fitting lines/planes to data (solving Ax = b)
   - **Neural Networks**: Forward and backward propagation involve solving linear systems
   - **Optimization**: Many optimization problems reduce to solving linear equations

5. **Computer Graphics**:
   - **Transformations**: Applying rotations, scaling, translations (matrix operations)
   - **3D Rendering**: Projecting 3D coordinates to 2D screen space
   - **Lighting Calculations**: Computing illumination using linear equations

6. **Economics & Finance**:
   - **Supply and Demand**: Modeling market equilibrium
   - **Portfolio Optimization**: Finding optimal asset allocations
   - **Input-Output Models**: Analyzing economic systems

7. **Scientific Computing**:
   - **Differential Equations**: Discretizing and solving PDEs leads to linear systems
   - **Finite Element Method**: Solving partial differential equations
   - **Signal Processing**: Filtering and reconstruction

8. **Numerical Methods**: Linear equations are the foundation for:
   - Gaussian elimination
   - LU decomposition
   - QR decomposition
   - Iterative solvers (for large sparse systems)

See the [linear equations example](./linear_equations/linear_equations_example.py) for more detailed demonstrations.

### Identity Matrix

The **identity matrix** is a square matrix with 1s on the main diagonal and 0s everywhere else. It acts as the multiplicative identity for matrices, meaning that multiplying any matrix by the identity matrix leaves it unchanged.

#### Example

A 3×3 identity matrix:
```
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```

In NumPy:
```python
import numpy as np

# Create identity matrix
I = np.eye(3)  # or np.identity(3)

# Verify: A @ I = A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
result = A @ I  # Result equals A
```

#### Why Identity Matrix is Useful

1. **Multiplicative Identity**: Acts as "1" for matrices - A @ I = I @ A = A for any matrix A. This is fundamental to matrix algebra.

2. **Matrix Inverses**: The defining property of inverse matrices is A @ A^(-1) = A^(-1) @ A = I. Identity is central to understanding and computing inverses.

3. **Linear Transformations**: Represents the "do nothing" transformation in linear algebra - no change to vectors or coordinate systems.

4. **Solving Linear Systems**: Used in Gaussian elimination, matrix factorization, and various algorithms for solving Ax = b.

5. **Regularization in Machine Learning**: 
   - **Ridge Regression**: Uses (A^T A + λI)^(-1) to prevent overfitting
   - **Neural Networks**: Identity matrices used in skip connections (ResNet architecture)
   - **Regularization**: Adding λI to matrices to ensure invertibility

6. **Eigenvalue Problems**: Identity matrix has eigenvalue 1 with multiplicity n, fundamental in eigenvalue decomposition.

7. **Matrix Decomposition**: Used in various factorization techniques like LU decomposition, QR decomposition, and SVD.

8. **Computer Graphics**: Represents no transformation in coordinate systems and transformations.

9. **Control Theory**: Used in state-space representations and system analysis.

10. **Numerical Methods**: Identity matrix is used as starting point in iterative algorithms and as a reference for matrix operations.

See the [identity matrix example](./identity_matrix/identity_matrix_example.py) for more detailed demonstrations.

### Matrix Multiplication

Matrix multiplication is a fundamental operation that combines two matrices to produce a third matrix. For matrices A (m×n) and B (n×p), the product C = AB is an m×p matrix where each element C[i,j] is the dot product of row i of A and column j of B.

#### Example

Given two matrices:
```
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]
```

The product A @ B is:
```
A @ B = [[1×5 + 2×7, 1×6 + 2×8],
         [3×5 + 4×7, 3×6 + 4×8]]
      = [[19, 22],
         [43, 50]]
```

In NumPy:
```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

result = A @ B  # or np.dot(A, B) or np.matmul(A, B)
# Result: [[19, 22],
#         [43, 50]]
```

#### Why Matrix Multiplication is Useful

1. **Linear Transformations**: Representing rotations, scaling, shearing, reflections, and other geometric transformations. Each transformation can be encoded as a matrix, and applying multiple transformations is done through matrix multiplication.

2. **Solving Linear Systems**: Systems of linear equations can be written as Ax = b, where matrix multiplication is fundamental to finding solutions.

3. **Computer Graphics**:
   - **3D Transformations**: Rotating, scaling, and translating 3D objects
   - **Projection**: Converting 3D coordinates to 2D screen coordinates
   - **Animation**: Combining multiple transformations over time

4. **Machine Learning & Deep Learning**:
   - **Neural Networks**: Forward propagation is essentially matrix multiplication (weights @ inputs + bias)
   - **Backpropagation**: Gradient calculations use matrix multiplication
   - **Feature Transformations**: Applying learned transformations to data
   - **Convolutional Layers**: Efficiently implemented using matrix multiplication

5. **Data Processing**:
   - **Feature Engineering**: Combining features through matrix operations
   - **Dimensionality Reduction**: PCA, SVD use matrix multiplication
   - **Data Transformations**: Applying filters and transformations

6. **Quantum Mechanics**: Representing quantum states and operations as matrices, where matrix multiplication represents state evolution.

7. **Economics & Operations Research**:
   - **Input-Output Models**: Analyzing economic relationships
   - **Markov Chains**: State transitions represented as matrix multiplication
   - **Optimization Problems**: Linear programming uses matrix operations

8. **Signal Processing**: Filtering, convolution, and transformations are implemented using matrix multiplication.

9. **Key Properties**:
   - **Associative**: (AB)C = A(BC) - allows efficient computation
   - **Distributive**: A(B + C) = AB + AC - simplifies calculations
   - **NOT Commutative**: AB ≠ BA in general - order matters!
   - **Transpose**: (AB)^T = B^T A^T - order reverses when transposing

See the [matrix multiplication example](./matrix_multiplication/matrix_multiplication_example.py) for more detailed demonstrations.

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

### Cosine Similarity

**Cosine similarity** measures the cosine of the angle between two vectors. It's a scale-invariant measure of orientation (direction) rather than magnitude. For vectors **a** and **b**, cosine similarity is: **cosine_similarity = (a · b) / (||a|| ||b||)** = **cos(θ)**, where θ is the angle between the vectors.

#### Example

Given two vectors:
```
a = [1, 0]
b = [1, 1]
```

The cosine similarity is:
```
cosine_similarity = (1×1 + 0×1) / (1 × √2) = 1/√2 ≈ 0.7071
```

This equals cos(45°) = √2/2 ≈ 0.7071

In NumPy:
```python
import numpy as np

a = np.array([1, 0])
b = np.array([1, 1])

cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Result: 0.7071 (cos(45°))
```

#### Why Cosine Similarity is Useful

1. **Scale-Invariant**: Unlike Euclidean distance, cosine similarity only depends on the angle between vectors, not their magnitudes. This makes it ideal for comparing vectors of different scales.

2. **Information Retrieval & Search**:
   - **Document Similarity**: Finding similar documents based on word frequencies (TF-IDF vectors)
   - **Search Engines**: Ranking search results by relevance
   - **Text Classification**: Grouping similar documents

3. **Recommendation Systems**:
   - **User Similarity**: Finding users with similar preferences
   - **Item Similarity**: Recommending items similar to user's past choices
   - **Collaborative Filtering**: "Users who liked X also liked Y"

4. **Machine Learning & Data Science**:
   - **Feature Similarity**: Measuring how similar features are across samples
   - **Clustering**: Grouping similar data points (K-means with cosine distance)
   - **Classification**: Using similarity as a distance metric
   - **Dimensionality Reduction**: Preserving similarity relationships

5. **Natural Language Processing**:
   - **Word Embeddings**: Comparing word vectors (Word2Vec, GloVe, BERT)
   - **Document Embeddings**: Finding semantically similar documents
   - **Semantic Search**: Finding documents with similar meaning
   - **Sentiment Analysis**: Comparing text sentiment vectors

6. **Computer Vision**:
   - **Image Similarity**: Finding similar images based on feature vectors
   - **Face Recognition**: Comparing face embeddings
   - **Object Matching**: Matching objects across images

7. **Advantages over Euclidean Distance**:
   - **Normalized Range**: Always in [-1, 1], easy to interpret
   - **Direction Focus**: Captures similarity in orientation, not just proximity
   - **Robust to Magnitude**: Works well when vector magnitudes vary significantly

8. **Interpretation**:
   - **1**: Vectors point in the same direction (maximum similarity)
   - **0**: Vectors are perpendicular (no similarity)
   - **-1**: Vectors point in opposite directions (maximum dissimilarity)

See the [cosine similarity example](./cosine_similarity/cosine_similarity_example.py) for more detailed demonstrations.

### Cosine Similarity

**Cosine similarity** measures the cosine of the angle between two vectors, indicating how similar their directions are regardless of their magnitudes. It's calculated as: **cos(θ) = (a · b) / (||a|| ||b||)**.

#### Example

Given two vectors:
```
a = [1, 2]
b = [2, 4]
```

The cosine similarity is:
```
cos(θ) = (1×2 + 2×4) / (√(1²+2²) × √(2²+4²))
       = 10 / (√5 × √20)
       = 10 / 10
       = 1.0
```

In NumPy:
```python
import numpy as np

a = np.array([1, 2])
b = np.array([2, 4])

cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Result: 1.0 (vectors point in the same direction)
```

#### Why Cosine Similarity is Useful

1. **Text Analysis & NLP**:
   - **Document Similarity**: Comparing documents using word frequency vectors (TF-IDF) to find similar articles, papers, or content
   - **Search Engines**: Ranking search results by similarity to the query vector
   - **Recommendation Systems**: Finding similar articles, products, or content based on user preferences
   - **Clustering**: Grouping similar documents together in topic modeling

2. **Machine Learning**:
   - **Feature Similarity**: Measuring similarity between feature vectors in high-dimensional spaces
   - **Recommendation Systems**: User-item similarity in collaborative filtering (e.g., Netflix, Amazon)
   - **Clustering Algorithms**: K-means variants using cosine distance instead of Euclidean distance
   - **Neural Networks**: Attention mechanisms and similarity layers use cosine similarity

3. **Information Retrieval**:
   - **Search**: Finding relevant documents/items based on query similarity
   - **Ranking**: Ordering results by relevance to a query
   - **Deduplication**: Finding duplicate or near-duplicate content

4. **Computer Vision**:
   - **Image Similarity**: Comparing image feature vectors extracted from deep learning models
   - **Face Recognition**: Comparing face embeddings to identify or verify individuals
   - **Object Detection**: Measuring similarity between object features

5. **Natural Language Processing**:
   - **Word Embeddings**: Measuring similarity between word vectors (Word2Vec, GloVe, BERT)
   - **Sentence Embeddings**: Comparing sentence representations for semantic similarity
   - **Semantic Search**: Finding semantically similar text regardless of exact word matches

6. **Key Advantages**:
   - **Scale-Invariant**: Works well with vectors of different magnitudes (only direction matters)
   - **Efficient**: Fast to compute, especially with sparse vectors
   - **Interpretable**: Easy to understand (angle between vectors, range [-1, 1])
   - **Normalized**: Always in [-1, 1], making comparisons meaningful across different vector scales

7. **Relationship to Other Concepts**:
   - Combines **dot product** and **Euclidean norm** in a normalized way
   - Directly related to the angle between vectors: `θ = arccos(cosine_similarity)`
   - When vectors are normalized, cosine similarity equals their dot product

See the [cosine similarity example](./cosine_similarity/cosine_similarity_example.py) for more detailed demonstrations.

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

### Eigenvalues and Eigenvectors

**Eigenvalues** and **eigenvectors** are fundamental concepts that reveal important properties of linear transformations. For a square matrix **A**, an eigenvector **v** (nonzero vector) and its corresponding eigenvalue λ satisfy: **Av = λv**. This means when matrix **A** acts on eigenvector **v**, the result is simply a scalar multiple of **v** (scaled by λ).

#### Example

Given a matrix:
```
A = [[4, 1],
     [2, 3]]
```

The eigenvalues and eigenvectors satisfy:
```
Av = λv
```

In NumPy:
```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues: array of eigenvalues
# eigenvectors: matrix where each column is an eigenvector
```

#### Why Eigenvalues and Eigenvectors are Useful

1. **Principal Component Analysis (PCA)**: Finding directions of maximum variance in data, essential for dimensionality reduction in machine learning and data science.

2. **Google PageRank Algorithm**: The PageRank algorithm finds the dominant eigenvector of the web graph to rank web pages by importance.

3. **Vibration Analysis**: In mechanical engineering, eigenvalues represent natural frequencies and eigenvectors represent vibration modes of structures.

4. **Quantum Mechanics**: Energy levels correspond to eigenvalues, and quantum states correspond to eigenvectors of the Hamiltonian operator.

5. **Image Processing**: Used in compression algorithms, feature extraction, and image recognition systems.

6. **Machine Learning**:
   - **Dimensionality Reduction**: PCA uses eigenvectors to reduce data dimensions
   - **Clustering**: Spectral clustering uses eigenvalues/eigenvectors
   - **Recommender Systems**: Collaborative filtering algorithms
   - **Neural Networks**: Understanding network dynamics and optimization

7. **Differential Equations**: Solving systems of linear differential equations, finding stable and unstable solutions.

8. **Stability Analysis**: In control theory, eigenvalues determine if systems are stable (negative real parts) or unstable (positive real parts).

9. **Graph Theory**: Finding communities in networks, computing centrality measures, analyzing social networks.

10. **Data Science**: Understanding data structure, finding principal directions of variation, detecting anomalies.

11. **Key Properties**:
    - **Sum of eigenvalues = Trace**: Σλᵢ = trace(A) where trace is the sum of diagonal elements
    - **Product of eigenvalues = Determinant**: Πλᵢ = det(A)
    - **A and A^T have the same eigenvalues** (but different eigenvectors in general)
    - **Symmetric matrices always have real eigenvalues**
    - **Eigenvalue decomposition**: A = PΛP⁻¹ where P contains eigenvectors and Λ contains eigenvalues

12. **Geometric Interpretation**: Eigenvectors point in directions preserved by the transformation, while eigenvalues tell us how much vectors in those directions are stretched (λ > 1), compressed (0 < λ < 1), or flipped (λ < 0).

See the [eigenvalues example](./eigenvalues/eigenvalues_example.py) for more detailed demonstrations.
