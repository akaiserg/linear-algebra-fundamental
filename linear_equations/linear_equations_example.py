"""
Linear Equations Example

Linear equations can be represented in both vector and matrix forms.
This example demonstrates various ways to solve linear equations.
"""

import numpy as np


def demonstrate_linear_equations():
    """Demonstrate linear equation solving"""
    
    print("=" * 60)
    print("Linear Equations Examples")
    print("=" * 60)
    
    # Example 1: Simple linear equation
    print("\n1. Simple Linear Equation:")
    print("   Solve: 2x + 3 = 11")
    print("   Rearrange: 2x = 11 - 3 = 8")
    print("   Solution: x = 8 / 2 = 4")
    x = (11 - 3) / 2
    print(f"   Result: x = {x}")
    
    # Example 2: System of two equations (matrix form)
    print("\n2. System of Two Linear Equations (Matrix Form):")
    print("   2x + 3y = 7")
    print("   4x + 5y = 13")
    print("\n   In matrix form: Ax = b")
    A = np.array([[2, 3],
                  [4, 5]])
    b = np.array([7, 13])
    print(f"   A = \n{A}")
    print(f"   b = {b}")
    
    x_solution = np.linalg.solve(A, b)
    print(f"   Solution: x = {x_solution}")
    print(f"   Verification: A @ x = {A @ x_solution}")
    
    # Example 3: Vector form
    print("\n3. Vector Form of Linear Equation:")
    print("   Equation: 3x + 4y = 10")
    print("   Vector form: [3, 4] · [x, y] = 10")
    a = np.array([3, 4])
    b_scalar = 10
    print(f"   Vector a = {a}")
    print(f"   Scalar b = {b_scalar}")
    
    # Find a solution
    x_sol = np.array([2, 1])  # 3*2 + 4*1 = 6 + 4 = 10
    print(f"   One solution: {x_sol}")
    print(f"   Verification: {a} · {x_sol} = {np.dot(a, x_sol)}")
    
    # Example 4: Solving using inverse
    print("\n4. Solving Using Matrix Inverse:")
    A = np.array([[2, 1],
                  [1, 3]])
    b = np.array([5, 7])
    print(f"   A = \n{A}")
    print(f"   b = {b}")
    
    A_inv = np.linalg.inv(A)
    x_inv = A_inv @ b
    print(f"   A^(-1) = \n{A_inv}")
    print(f"   Solution: x = A^(-1) @ b = {x_inv}")
    print(f"   Note: np.linalg.solve() is preferred for numerical stability")
    
    # Example 5: Overdetermined system (least squares)
    print("\n5. Overdetermined System (More Equations than Unknowns):")
    print("   When we have more equations than unknowns, use least squares")
    A_over = np.array([[1, 2],
                       [3, 4],
                       [5, 6]])
    b_over = np.array([3, 7, 11])
    print(f"   A (3x2) = \n{A_over}")
    print(f"   b = {b_over}")
    
    x_ls, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
    print(f"   Least squares solution: {x_ls}")
    print(f"   Residuals: {residuals}")
    
    # Example 6: Underdetermined system
    print("\n6. Underdetermined System (More Unknowns than Equations):")
    print("   When we have more unknowns than equations, infinite solutions exist")
    A_under = np.array([[1, 2, 3]])
    b_under = np.array([6])
    print(f"   A (1x3) = {A_under}")
    print(f"   b = {b_under}")
    
    x_under = np.linalg.pinv(A_under) @ b_under
    print(f"   One solution (minimum norm): {x_under}")
    print(f"   Verification: A @ x = {A_under @ x_under}")
    
    # Example 7: Homogeneous system
    print("\n7. Homogeneous System (Ax = 0):")
    A_hom = np.array([[1, 2],
                      [2, 4]])
    b_hom = np.array([0, 0])
    print(f"   A = \n{A_hom}")
    print(f"   b = {b_hom}")
    
    det_A = np.linalg.det(A_hom)
    print(f"   det(A) = {det_A}")
    print("   Since det(A) = 0, non-trivial solutions exist")
    print("   Trivial solution: x = [0, 0]")
    print("   Non-trivial solutions: any vector in the null space")
    
    # Example 8: Types of solutions
    print("\n8. Types of Solutions:")
    print("   - Unique solution: det(A) ≠ 0 (A is invertible)")
    print("   - No solution: System is inconsistent")
    print("   - Infinite solutions: det(A) = 0 (A is singular)")
    
    # Example 9: Checking solution
    print("\n9. Verifying a Solution:")
    A = np.array([[1, 2],
                  [3, 4]])
    b = np.array([5, 11])
    x = np.array([1, 2])
    print(f"   A = \n{A}")
    print(f"   b = {b}")
    print(f"   Proposed solution: x = {x}")
    
    Ax = A @ x
    print(f"   A @ x = {Ax}")
    if np.allclose(Ax, b):
        print("   ✓ Solution is correct!")
    else:
        print("   ✗ Solution is incorrect")
    
    # Example 10: Practical application
    print("\n10. Practical Application:")
    print("    Linear equations are used in:")
    print("    - Engineering: Circuit analysis, structural analysis")
    print("    - Economics: Supply and demand models")
    print("    - Machine Learning: Linear regression, neural networks")
    print("    - Computer Graphics: Transformations, rendering")
    print("    - Data Science: Fitting models, optimization")


if __name__ == "__main__":
    demonstrate_linear_equations()

