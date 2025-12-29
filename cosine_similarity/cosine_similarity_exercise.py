"""
Cosine Similarity Exercise

Complete the functions below to practice cosine similarity operations.
Run this file to check your solutions.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Basic Cosine Similarity
    Calculate the cosine similarity between two vectors.
    
    Cosine similarity = (a · b) / (||a|| ||b||)
    
    a = [1, 0]
    b = [1, 1]
    
    Expected result: approximately 0.7071
    (cos(45°) = √2/2 ≈ 0.7071)
    """
    a = np.array([1, 0])
    b = np.array([1, 1])
    
    result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    result = None  # Replace None with your solution
    
    return result


def exercise_2():
    """
    Exercise 2: Cosine Similarity Using np.dot() and np.linalg.norm()
    Calculate cosine similarity using explicit dot product and norms.
    
    a = [3, 4]
    b = [4, 3]
    
    Expected result: 0.96
    (cos(θ) = (3×4 + 4×3) / (5 × 5) = 24/25 = 0.96)
    """
    a = np.array([3, 4])
    b = np.array([4, 3])
    
    result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    return result


def exercise_3():
    """
    Exercise 3: Identical Vectors (Same Direction)
    When two vectors point in the same direction, cosine similarity = 1.
    
    a = [2, 3]
    b = [4, 6]  (b = 2a, same direction)
    
    Return True if cosine similarity equals 1, False otherwise.
    """
    a = np.array([2, 3])
    b = np.array([4, 6])  # b = 2a, same direction
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(similarity)
        
    return np.isclose(similarity, 1.0)


def exercise_4():
    """
    Exercise 4: Orthogonal Vectors (Perpendicular)
    When two vectors are perpendicular, cosine similarity = 0.
    
    a = [1, 0]
    b = [0, 1]
    
    Return True if cosine similarity equals 0, False otherwise.
    """
    a = np.array([1, 0])
    b = np.array([0, 1])

    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(similarity)
    return np.isclose(similarity, 0.0)


def exercise_5():
    """
    Exercise 5: Opposite Vectors
    When two vectors point in opposite directions, cosine similarity = -1.
    
    a = [2, 3]
    b = [-4, -6]  (b = -2a, opposite direction)
    
    Return True if cosine similarity equals -1, False otherwise.
    """
    a = np.array([2, 3])
    b = np.array([-4, -6])  # b = -2a, opposite direction
    
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(similarity)
    return np.isclose(similarity, -1.0)


def exercise_6():
    """
    Exercise 6: Cosine Similarity is Scale-Invariant
    Cosine similarity only depends on the angle, not the magnitudes.
    Verify that cosine similarity of (a, b) equals cosine similarity of (ka, kb).
    
    a = [1, 2]
    b = [3, 4]
    k = 5
    
    Return True if both cosine similarities are equal, False otherwise.
    """
    a = np.array([1, 2])
    b = np.array([3, 4])
    k = 5
    
    similarity_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    similarity_kab = np.dot(k * a, k * b) / (np.linalg.norm(k * a) * np.linalg.norm(k * b))
    print(similarity_ab)
    print(similarity_kab)
    return np.isclose(similarity_ab, similarity_kab)


def exercise_7():
    """
    Exercise 7: Cosine Similarity Range
    Cosine similarity always ranges from -1 to 1.
    
    Verify that for any two non-zero vectors, cosine similarity is in [-1, 1].
    
    a = [1, 2, 3]
    b = [4, 5, 6]
    
    Return True if cosine similarity is between -1 and 1 (inclusive), False otherwise.
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(similarity)
    return -1.0 <= similarity <= 1.0
    

def exercise_8():
    """
    Exercise 8: Angle from Cosine Similarity
    Given cosine similarity, calculate the angle between vectors in degrees.
    
    If cosine similarity = 0.5, what is the angle?
    
    Expected result: 60.0 degrees
    (arccos(0.5) = 60°)
    """
    cosine_similarity = 0.5
    
    result = np.arccos(cosine_similarity)
    print(result)
    return np.degrees(result)


def exercise_9():
    """
    Exercise 9: Cosine Similarity for Text/Document Vectors
    In information retrieval, cosine similarity measures document similarity.
    
    Calculate cosine similarity between two document vectors.
    
    doc1 = [1, 2, 0, 1, 0]  (word frequencies)
    doc2 = [2, 1, 1, 0, 1]
    
    Expected result: approximately 0.617
    """
    doc1 = np.array([1, 2, 0, 1, 0])
    doc2 = np.array([2, 1, 1, 0, 1])
    
    dot_product = np.dot(doc1, doc2)
    norm_doc1 = np.linalg.norm(doc1)
    norm_doc2 = np.linalg.norm(doc2)
    result = dot_product / (norm_doc1 * norm_doc2)
    print(result)
    return result

def exercise_10():
    """
    Exercise 10: Normalized Vectors and Cosine Similarity
    For normalized (unit) vectors, cosine similarity equals the dot product.
    This is because ||a|| = ||b|| = 1, so (a · b) / (1 × 1) = a · b
    
    Normalize two vectors and verify that their cosine similarity equals their dot product.
    
    a = [3, 4]
    b = [5, 12]
    
    Return True if cosine similarity equals dot product for normalized vectors, False otherwise.
    """
    a = np.array([3, 4])
    b = np.array([5, 12])
    
    a_normalized = a / np.linalg.norm(a)
    b_normalized = b / np.linalg.norm(b)
    similarity = np.dot(a_normalized, b_normalized) / (np.linalg.norm(a_normalized) * np.linalg.norm(b_normalized))
    dot_product = np.dot(a_normalized, b_normalized)
    print(similarity)
    print(dot_product)
    return np.isclose(similarity, dot_product)


# ============================================================================
# Test your solutions
# ============================================================================

def test_solutions():
    """Run tests to check your solutions"""
    
    print("=" * 60)
    print("Testing Your Solutions")
    print("=" * 60)
    
    # Test Exercise 1
    print("\nExercise 1: Basic Cosine Similarity")
    result1 = exercise_1()
    expected1 = np.sqrt(2) / 2  # cos(45°) = √2/2
    if result1 is not None and np.isclose(result1, expected1, atol=0.001):
        print(f"✓ Correct! Cosine similarity = {result1:.4f}")
    else:
        print(f"✗ Incorrect. Expected: approximately {expected1:.4f}")
        if result1 is not None:
            print(f"Got: {result1}")
    
    # Test Exercise 2
    print("\nExercise 2: Cosine Similarity Using np.dot() and np.linalg.norm()")
    result2 = exercise_2()
    expected2 = 0.96
    if result2 is not None and np.isclose(result2, expected2, atol=0.01):
        print(f"✓ Correct! Cosine similarity = {result2:.4f}")
    else:
        print(f"✗ Incorrect. Expected: approximately {expected2}")
        if result2 is not None:
            print(f"Got: {result2}")
    
    # Test Exercise 3
    print("\nExercise 3: Identical Vectors (Same Direction)")
    result3 = exercise_3()
    if result3:
        print("✓ Correct! Cosine similarity = 1 for vectors in same direction")
    else:
        print("✗ Incorrect. Vectors in same direction should have cosine similarity = 1.")
    
    # Test Exercise 4
    print("\nExercise 4: Orthogonal Vectors (Perpendicular)")
    result4 = exercise_4()
    if result4:
        print("✓ Correct! Cosine similarity = 0 for perpendicular vectors")
    else:
        print("✗ Incorrect. Perpendicular vectors should have cosine similarity = 0.")
    
    # Test Exercise 5
    print("\nExercise 5: Opposite Vectors")
    result5 = exercise_5()
    if result5:
        print("✓ Correct! Cosine similarity = -1 for opposite vectors")
    else:
        print("✗ Incorrect. Opposite vectors should have cosine similarity = -1.")
    
    # Test Exercise 6
    print("\nExercise 6: Cosine Similarity is Scale-Invariant")
    result6 = exercise_6()
    if result6:
        print("✓ Correct! Cosine similarity is scale-invariant")
    else:
        print("✗ Incorrect. Cosine similarity should be scale-invariant.")
    
    # Test Exercise 7
    print("\nExercise 7: Cosine Similarity Range")
    result7 = exercise_7()
    if result7:
        print("✓ Correct! Cosine similarity is in [-1, 1]")
    else:
        print("✗ Incorrect. Cosine similarity should be between -1 and 1.")
    
    # Test Exercise 8
    print("\nExercise 8: Angle from Cosine Similarity")
    result8 = exercise_8()
    expected8 = 60.0
    if result8 is not None and np.isclose(result8, expected8, atol=0.1):
        print(f"✓ Correct! Angle = {result8}°")
    else:
        print(f"✗ Incorrect. Expected: approximately {expected8}°")
        if result8 is not None:
            print(f"Got: {result8}°")
    
    # Test Exercise 9
    print("\nExercise 9: Cosine Similarity for Text/Document Vectors")
    result9 = exercise_9()
    expected9 = 0.617
    if result9 is not None and np.isclose(result9, expected9, atol=0.01):
        print(f"✓ Correct! Cosine similarity = {result9:.4f}")
    else:
        print(f"✗ Incorrect. Expected: approximately {expected9}")
        if result9 is not None:
            print(f"Got: {result9}")
    
    # Test Exercise 10
    print("\nExercise 10: Normalized Vectors and Cosine Similarity")
    result10 = exercise_10()
    if result10:
        print("✓ Correct! For normalized vectors, cosine similarity = dot product")
    else:
        print("✗ Incorrect. For normalized vectors, cosine similarity should equal dot product.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_solutions()
