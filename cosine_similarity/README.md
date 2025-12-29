# Cosine Similarity

**Cosine similarity** measures the cosine of the angle between two vectors. It's a measure of orientation (direction) rather than magnitude, making it scale-invariant and widely used in machine learning and information retrieval.

## Definition

For two vectors **a** and **b**, cosine similarity is:

```
cosine_similarity = (a · b) / (||a|| ||b||)
```

This is equivalent to:
```
cosine_similarity = cos(θ)
```

where θ is the angle between the vectors.

## Range and Interpretation

Cosine similarity ranges from **-1 to 1**:

- **1**: Vectors point in the same direction (identical orientation)
- **0**: Vectors are perpendicular (orthogonal)
- **-1**: Vectors point in opposite directions
- **Values between**: Indicate the angle between vectors

## Properties

1. **Scale-Invariant**: Cosine similarity only depends on the angle, not the magnitudes. Multiplying both vectors by the same scalar doesn't change the similarity.

2. **Symmetric**: cosine_similarity(a, b) = cosine_similarity(b, a)

3. **Range**: Always between -1 and 1 (by definition of cosine function)

4. **Normalized Vectors**: For unit vectors (||a|| = ||b|| = 1), cosine similarity equals the dot product.

## In NumPy

```python
import numpy as np

a = np.array([1, 0])
b = np.array([1, 1])

# Method 1: Explicit calculation
cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Method 2: For normalized vectors
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
cosine_sim = np.dot(a_norm, b_norm)  # Since ||a_norm|| = ||b_norm|| = 1
```

## Relationship to Dot Product

Cosine similarity is the dot product divided by the product of magnitudes:

```
cosine_similarity = dot_product / (magnitude_a × magnitude_b)
```

This normalizes the dot product, removing the effect of vector magnitudes.

## Why Cosine Similarity is Important

1. **Information Retrieval**: 
   - Finding similar documents based on word frequencies
   - Search engines use it to rank results
   - Document clustering and classification

2. **Recommendation Systems**:
   - Finding similar users based on preferences
   - Recommending items similar to user's past choices
   - Collaborative filtering

3. **Machine Learning**:
   - **Feature Similarity**: Measuring how similar features are
   - **Clustering**: Grouping similar data points
   - **Classification**: Using similarity as a distance metric
   - **Embeddings**: Comparing word/document embeddings

4. **Natural Language Processing**:
   - Word embeddings similarity (Word2Vec, GloVe)
   - Document similarity
   - Semantic search

5. **Computer Vision**:
   - Image similarity
   - Face recognition
   - Object matching

6. **Data Mining**: Pattern recognition, anomaly detection

7. **Advantages over Euclidean Distance**:
   - **Scale-invariant**: Not affected by vector magnitudes
   - **Focuses on direction**: Captures similarity in orientation
   - **Normalized**: Always in [-1, 1] range, easy to interpret

## Running the Examples

```bash
python cosine_similarity_example.py
python cosine_similarity_exercise.py
```
