"""
Cosine Similarity Exercise - SOLUTIONS

This file contains the solutions to the cosine similarity exercises.
Try to solve them yourself first before looking at the solutions!
"""

import numpy as np


def exercise_1_solution():
    """Solution to Exercise 1: Basic Cosine Similarity"""
    a = np.array([1, 0])
    b = np.array([1, 1])
    
    # Cosine similarity = (a · b) / (||a|| ||b||)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    result = dot_product / (norm_a * norm_b)
    
    # Alternative one-liner:
    # result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    return result


def exercise_2_solution():
    """Solution to Exercise 2: Cosine Similarity Using np.dot() and np.linalg.norm()"""
    a = np.array([3, 4])
    b = np.array([4, 3])
    
    # Calculate using explicit functions
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    result = dot_product / (norm_a * norm_b)
    
    return result


def exercise_3_solution():
    """Solution to Exercise 3: Identical Vectors (Same Direction)"""
    a = np.array([2, 3])
    b = np.array([4, 6])  # b = 2a, same direction
    
    # Calculate cosine similarity
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Check if it equals 1
    result = np.isclose(cosine_sim, 1.0)
    
    return result


def exercise_4_solution():
    """Solution to Exercise 4: Orthogonal Vectors (Perpendicular)"""
    a = np.array([1, 0])
    b = np.array([0, 1])
    
    # Calculate cosine similarity
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Check if it equals 0
    result = np.isclose(cosine_sim, 0.0)
    
    return result


def exercise_5_solution():
    """Solution to Exercise 5: Opposite Vectors"""
    a = np.array([2, 3])
    b = np.array([-4, -6])  # b = -2a, opposite direction
    
    # Calculate cosine similarity
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Check if it equals -1
    result = np.isclose(cosine_sim, -1.0)
    
    return result


def exercise_6_solution():
    """Solution to Exercise 6: Cosine Similarity is Scale-Invariant"""
    a = np.array([1, 2])
    b = np.array([3, 4])
    k = 5
    
    # Calculate cosine similarity of (a, b)
    cosine_sim_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Calculate cosine similarity of (ka, kb)
    cosine_sim_kab = np.dot(k * a, k * b) / (np.linalg.norm(k * a) * np.linalg.norm(k * b))
    
    # Check if they are equal
    result = np.isclose(cosine_sim_ab, cosine_sim_kab)
    
    return result


def exercise_7_solution():
    """Solution to Exercise 7: Cosine Similarity Range"""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # Calculate cosine similarity
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Check if it's in [-1, 1]
    result = -1.0 <= cosine_sim <= 1.0
    
    return result


def exercise_8_solution():
    """Solution to Exercise 8: Angle from Cosine Similarity"""
    cosine_similarity = 0.5
    
    # Clamp to avoid precision issues
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    theta_rad = np.arccos(cosine_similarity)
    result = np.degrees(theta_rad)
    
    return result


def exercise_9_solution():
    """Solution to Exercise 9: Cosine Similarity for Text/Document Vectors"""
    doc1 = np.array([1, 2, 0, 1, 0])
    doc2 = np.array([2, 1, 1, 0, 1])
    
    # Calculate cosine similarity
    dot_product = np.dot(doc1, doc2)
    norm_doc1 = np.linalg.norm(doc1)
    norm_doc2 = np.linalg.norm(doc2)
    result = dot_product / (norm_doc1 * norm_doc2)
    
    return result


def exercise_10_solution():
    """Solution to Exercise 10: Normalized Vectors and Cosine Similarity"""
    a = np.array([3, 4])
    b = np.array([5, 12])
    
    # Normalize vectors
    a_normalized = a / np.linalg.norm(a)
    b_normalized = b / np.linalg.norm(b)
    
    # Calculate cosine similarity
    cosine_sim = np.dot(a_normalized, b_normalized) / (np.linalg.norm(a_normalized) * np.linalg.norm(b_normalized))
    
    # Calculate dot product of normalized vectors
    dot_product = np.dot(a_normalized, b_normalized)
    
    # For normalized vectors, ||a|| = ||b|| = 1, so cosine similarity = dot product
    result = np.isclose(cosine_sim, dot_product)
    
    return result


# Run solutions to verify they work
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise Solutions - Verification")
    print("=" * 60)
    
    print("\nExercise 1 Result:", exercise_1_solution())
    
    print("\nExercise 2 Result:", exercise_2_solution())
    
    print("\nExercise 3 Result:", exercise_3_solution())
    
    print("\nExercise 4 Result:", exercise_4_solution())
    
    print("\nExercise 5 Result:", exercise_5_solution())
    
    print("\nExercise 6 Result:", exercise_6_solution())
    
    print("\nExercise 7 Result:", exercise_7_solution())
    
    print("\nExercise 8 Result:", exercise_8_solution())
    
    print("\nExercise 9 Result:", exercise_9_solution())
    
    print("\nExercise 10 Result:", exercise_10_solution())
