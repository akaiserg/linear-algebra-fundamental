"""
Singular Value Decomposition (SVD) Example

SVD decomposes any matrix A into three components:
A = U @ Σ @ V^T

where:
- U: left singular vectors (orthogonal matrix)
- Σ: diagonal matrix of singular values
- V^T: right singular vectors (orthogonal matrix, transposed)
"""

import numpy as np


def demonstrate_svd():
    """Demonstrate SVD operations"""
    
    print("=" * 60)
    print("Singular Value Decomposition (SVD) Examples")
    print("=" * 60)
    
    # Example 1: Basic SVD
    print("\n1. Basic SVD Decomposition:")
    A = np.array([[1, 2],
                  [3, 4]])
    print(f"Matrix A:\n{A}")
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"\nU (left singular vectors):\n{U}")
    print(f"\nS (singular values): {S}")
    print(f"\nV^T (right singular vectors, transposed):\n{Vt}")
    
    # Reconstruct
    S_matrix = np.diag(S)
    reconstructed = U @ S_matrix @ Vt
    print(f"\nReconstructed A = U @ diag(S) @ V^T:\n{reconstructed}")
    print(f"Matches original? {np.allclose(A, reconstructed)}")
    
    # Example 2: Understanding Singular Values
    print("\n2. Singular Values - What They Mean:")
    print("Singular values are always non-negative and in descending order")
    print("They represent the 'importance' or 'strength' of each component")
    
    A2 = np.array([[3, 0],
                   [0, 2]])
    S2 = np.linalg.svd(A2, compute_uv=False)
    print(f"\nFor diagonal matrix A = [[3, 0], [0, 2]]:")
    print(f"Singular values: {S2}")
    print("Note: For diagonal matrices, singular values = absolute values of diagonal")
    
    # Example 3: Rank from Singular Values
    print("\n3. Matrix Rank from Singular Values:")
    A3 = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 6, 9]])  # Rank 1
    S3 = np.linalg.svd(A3, compute_uv=False)
    print(f"Matrix A (rank 1):\n{A3}")
    print(f"Singular values: {S3}")
    print(f"Non-zero singular values (with tolerance 1e-10): {np.sum(S3 > 1e-10)}")
    print(f"Actual rank: {np.linalg.matrix_rank(A3)}")
    
    # Example 4: Low-Rank Approximation
    print("\n4. Low-Rank Approximation:")
    A4 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    print(f"Original matrix A:\n{A4}")
    
    U4, S4, Vt4 = np.linalg.svd(A4, full_matrices=False)
    print(f"\nSingular values: {S4}")
    
    # Rank-1 approximation
    k = 1
    U_k = U4[:, :k]
    S_k = S4[:k]
    Vt_k = Vt4[:k, :]
    A_k = U_k @ np.diag(S_k) @ Vt_k
    print(f"\nRank-{k} approximation:\n{A_k}")
    print(f"Error (Frobenius norm): {np.linalg.norm(A4 - A_k, 'fro'):.4f}")
    
    # Rank-2 approximation
    k = 2
    U_k = U4[:, :k]
    S_k = S4[:k]
    Vt_k = Vt4[:k, :]
    A_k = U_k @ np.diag(S_k) @ Vt_k
    print(f"\nRank-{k} approximation:\n{A_k}")
    print(f"Error (Frobenius norm): {np.linalg.norm(A4 - A_k, 'fro'):.4f}")
    
    # Example 5: Orthogonal Properties
    print("\n5. Orthogonal Properties of U and V:")
    A5 = np.array([[1, 2],
                   [3, 4]])
    U5, S5, Vt5 = np.linalg.svd(A5, full_matrices=False)
    
    print(f"U^T @ U (should be identity):\n{U5.T @ U5}")
    print(f"\nV^T @ V (Vt @ Vt.T, should be identity):\n{Vt5 @ Vt5.T}")
    print("\nBoth U and V are orthogonal matrices!")
    
    # Example 6: SVD of Transpose
    print("\n6. SVD of Transpose:")
    A6 = np.array([[1, 2],
                   [3, 4]])
    S6 = np.linalg.svd(A6, compute_uv=False)
    S6T = np.linalg.svd(A6.T, compute_uv=False)
    
    print(f"Singular values of A: {S6}")
    print(f"Singular values of A^T: {S6T}")
    print(f"Are they equal? {np.allclose(S6, S6T)}")
    print("Note: Singular values of A and A^T are always the same!")
    
    # Example 7: Rectangular Matrix
    print("\n7. SVD for Rectangular Matrix:")
    A7 = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]])  # 2×4 matrix
    print(f"Matrix A (2×4):\n{A7}")
    
    U7, S7, Vt7 = np.linalg.svd(A7, full_matrices=False)
    print(f"\nU shape: {U7.shape} (2×2)")
    print(f"S length: {len(S7)} (min(2,4) = 2)")
    print(f"V^T shape: {Vt7.shape} (2×4)")
    print("\nFor m×n matrix with full_matrices=False:")
    print("  - U: m×min(m,n)")
    print("  - S: min(m,n) singular values")
    print("  - V^T: min(m,n)×n")
    
    # Example 8: Compression Example
    print("\n8. Compression Using SVD:")
    A8 = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])  # 3×4 = 12 elements
    m, n = A8.shape
    original_size = m * n
    
    print(f"Original matrix ({m}×{n}): {original_size} elements")
    
    for k in [1, 2]:
        U8, S8, Vt8 = np.linalg.svd(A8, full_matrices=False)
        compressed_size = m * k + k + k * n
        ratio = original_size / compressed_size
        print(f"\nRank-{k} approximation:")
        print(f"  Storage: U({m}×{k}) + S({k}) + V^T({k}×{n}) = {compressed_size} elements")
        print(f"  Compression ratio: {ratio:.2f}")
        print(f"  {'Compressed' if ratio > 1 else 'Expanded'} by factor of {ratio:.2f}")


if __name__ == "__main__":
    demonstrate_svd()

