"""
Cosine Similarity Example

Cosine similarity measures the cosine of the angle between two vectors.
It ranges from -1 to 1 and indicates how similar two vectors are in direction.
"""

import numpy as np


def demonstrate_cosine_similarity():
    """Demonstrate cosine similarity operations"""
    
    print("=" * 60)
    print("Cosine Similarity Examples")
    print("=" * 60)
    
    # Example 1: Basic cosine similarity
    print("\n1. Basic Cosine Similarity:")
    a = np.array([1, 0])
    b = np.array([1, 1])
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"Formula: (a · b) / (||a|| ||b||) = {np.dot(a, b)} / ({np.linalg.norm(a)} × {np.linalg.norm(b)})")
    print(f"This is cos(45°) = √2/2 ≈ 0.7071")
    
    # Example 2: Same direction (cosine similarity = 1)
    print("\n2. Vectors Pointing in Same Direction:")
    a = np.array([2, 3])
    b = np.array([4, 6])  # b = 2a
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Vector a: {a}")
    print(f"Vector b: {b} (b = 2a, same direction)")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print("When vectors point in the same direction, cosine similarity = 1")
    
    # Example 3: Perpendicular (cosine similarity = 0)
    print("\n3. Perpendicular Vectors:")
    a = np.array([1, 0])
    b = np.array([0, 1])
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print("When vectors are perpendicular, cosine similarity = 0")
    
    # Example 4: Opposite directions (cosine similarity = -1)
    print("\n4. Vectors Pointing in Opposite Directions:")
    a = np.array([2, 3])
    b = np.array([-4, -6])  # b = -2a
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Vector a: {a}")
    print(f"Vector b: {b} (b = -2a, opposite direction)")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print("When vectors point in opposite directions, cosine similarity = -1")
    
    # Example 5: Scale-invariance
    print("\n5. Scale-Invariance Property:")
    a = np.array([1, 2])
    b = np.array([3, 4])
    k = 5
    cosine_sim_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosine_sim_kab = np.dot(k * a, k * b) / (np.linalg.norm(k * a) * np.linalg.norm(k * b))
    print(f"Vectors: a={a}, b={b}, k={k}")
    print(f"Cosine similarity of (a, b): {cosine_sim_ab:.4f}")
    print(f"Cosine similarity of ({k}a, {k}b): {cosine_sim_kab:.4f}")
    print(f"Equal? {np.isclose(cosine_sim_ab, cosine_sim_kab)}")
    print("Cosine similarity only depends on angle, not magnitude!")
    
    # Example 6: Range of values
    print("\n6. Range of Cosine Similarity:")
    print("Cosine similarity always ranges from -1 to 1:")
    print("  -1: Vectors point in opposite directions")
    print("   0: Vectors are perpendicular")
    print("   1: Vectors point in the same direction")
    print("  Values between: Indicate the angle between vectors")
    
    # Example 7: Angle calculation
    print("\n7. Calculating Angle from Cosine Similarity:")
    cosine_sim = 0.5
    cosine_sim_clipped = np.clip(cosine_sim, -1.0, 1.0)
    angle_rad = np.arccos(cosine_sim_clipped)
    angle_deg = np.degrees(angle_rad)
    print(f"Cosine similarity: {cosine_sim}")
    print(f"Angle: {angle_deg:.1f}°")
    print(f"Formula: θ = arccos(cosine_similarity)")
    
    # Example 8: Document similarity
    print("\n8. Document Similarity (Information Retrieval):")
    doc1 = np.array([1, 2, 0, 1, 0])  # Word frequencies
    doc2 = np.array([2, 1, 1, 0, 1])
    cosine_sim = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
    print(f"Document 1 (word frequencies): {doc1}")
    print(f"Document 2 (word frequencies): {doc2}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print("Higher values indicate more similar documents")
    
    # Example 9: Normalized vectors
    print("\n9. Normalized Vectors and Cosine Similarity:")
    a = np.array([3, 4])
    b = np.array([5, 12])
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    cosine_sim = np.dot(a_norm, b_norm) / (np.linalg.norm(a_norm) * np.linalg.norm(b_norm))
    dot_product = np.dot(a_norm, b_norm)
    print(f"Original vectors: a={a}, b={b}")
    print(f"Normalized vectors: a_norm={a_norm}, b_norm={b_norm}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"Dot product of normalized vectors: {dot_product:.4f}")
    print("For normalized vectors, cosine similarity = dot product (since ||a|| = ||b|| = 1)")
    
    # Example 10: Applications
    print("\n10. Applications of Cosine Similarity:")
    print("  - Information Retrieval: Finding similar documents")
    print("  - Recommendation Systems: Finding similar users/items")
    print("  - Machine Learning: Feature similarity, clustering")
    print("  - Natural Language Processing: Word/document embeddings")
    print("  - Computer Vision: Image similarity")
    print("  - Data Mining: Pattern recognition")


if __name__ == "__main__":
    demonstrate_cosine_similarity()
