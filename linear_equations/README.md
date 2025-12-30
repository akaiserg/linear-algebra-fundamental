# Linear Equations

Linear equations are fundamental in mathematics and have wide applications. They can be represented in both **vector form** and **matrix form**.

## Vector Form

A linear equation in vector form:
```
a · x = b
```

where:
- **a** is a coefficient vector
- **x** is the variable vector
- **b** is a scalar constant

**Example:**
```
3x + 4y = 10
```
In vector form: `[3, 4] · [x, y] = 10`

## Matrix Form

A system of linear equations can be written as:
```
Ax = b
```

where:
- **A** is the coefficient matrix (m×n)
- **x** is the variable vector (n×1)
- **b** is the constant vector (m×1)

**Example:**
```
2x + 3y = 7
4x + 5y = 13
```

In matrix form:
```
[[2, 3],     [[x],     [[7],
 [4, 5]]  @   [y]]  =   [13]]
```

## Solving Linear Equations

### 1. Simple Linear Equation
```python
# Solve: 2x + 3 = 11
x = (11 - 3) / 2
```

### 2. System of Equations (Unique Solution)
```python
import numpy as np

A = np.array([[2, 3],
              [4, 5]])
b = np.array([7, 13])

x = np.linalg.solve(A, b)
```

### 3. Using Matrix Inverse
```python
A_inv = np.linalg.inv(A)
x = A_inv @ b
```

**Note:** `np.linalg.solve()` is preferred over `inv()` for numerical stability.

## Types of Systems

### 1. Unique Solution
- **Condition:** det(A) ≠ 0 (A is invertible)
- **Method:** `np.linalg.solve(A, b)`

### 2. Overdetermined System
- **Condition:** More equations than unknowns (m > n)
- **Method:** Least squares - `np.linalg.lstsq(A, b)` or `np.linalg.pinv(A) @ b`
- **Result:** Best approximate solution

### 3. Underdetermined System
- **Condition:** More unknowns than equations (m < n)
- **Method:** Pseudo-inverse - `np.linalg.pinv(A) @ b`
- **Result:** Minimum norm solution (infinite solutions exist)

### 4. Homogeneous System
- **Condition:** b = 0 (Ax = 0)
- **Solutions:**
  - Trivial: x = 0 (always exists)
  - Non-trivial: Exist if det(A) = 0 (A is singular)

## Properties

1. **Existence of Solution:**
   - Unique solution exists if A is invertible (det(A) ≠ 0)
   - Infinite solutions if A is singular and system is consistent
   - No solution if system is inconsistent

2. **Verification:**
   - Check if Ax = b for proposed solution x

3. **Numerical Methods:**
   - Gaussian elimination
   - LU decomposition
   - QR decomposition
   - Iterative methods (for large systems)

## Applications

1. **Engineering:** Circuit analysis, structural mechanics
2. **Economics:** Supply and demand, optimization
3. **Machine Learning:** Linear regression, neural networks
4. **Computer Graphics:** Transformations, 3D rendering
5. **Data Science:** Model fitting, optimization problems
6. **Physics:** Force equilibrium, wave equations

## Running the Examples

```bash
python linear_equations_example.py
python linear_equations_exercise.py
```

