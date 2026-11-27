# Import json for deserializing vector strings and numpy for numerical array computation
import json, numpy as np

# Import NearestNeighbors algorithm from scikit-learn for high-dimensional spatial indexing
from sklearn.neighbors import NearestNeighbors

def find_matches(query, cases, threshold=0.08, k=3):
    """
    Finds top-k nearest matching missing persons from a database gallery for a given query face.
    Returns: list of tuples -> [(Case, rms_distance, confidence_percentage)] for all matches within threshold.
    """
    # Filter the input list to only include Case records that have a non-empty embedding
    cases = [c for c in cases if c.embedding]
    
    # If the probe query embedding is None (no face detected) or cases gallery is empty, return empty list
    if query is None or not cases:
        return []
    
    # Deserialize each stored JSON embedding string into a NumPy array, building matrix X of shape (N, 1404)
    X = np.array([json.loads(c.embedding) for c in cases])
    
    # Initialize and fit scikit-learn NearestNeighbors model on matrix X (k neighbors, capped at total case count)
    nn = NearestNeighbors(n_neighbors=min(k, len(cases))).fit(X)
    
    # Query nearest neighbors for the probe vector; returns Euclidean distances and indices of top matches
    dist, idx = nn.kneighbors([query])
    
    # Calculate normalization divisor sqrt(1404) ≈ 37.47 to convert Euclidean distance into Root-Mean-Square (RMS) error
    n = np.sqrt(X.shape[1])
    
    # Initialize list to hold matching candidates that satisfy the threshold
    out = []
    
    # Iterate through the distances and case indices returned by the nearest neighbor search
    for d, i in zip(dist[0], idx[0]):
        # Compute RMS coordinate error per landmark component (distance / sqrt(1404))
        rms = float(d / n)
        
        # Check if the RMS error is within the permissible match threshold (default 0.08)
        if rms <= threshold:
            # Append tuple: (Case object, float RMS distance, rounded confidence score percentage)
            # Confidence formula: 100 * (1 - rms / threshold), where 0% is at threshold cutoff and 100% is exact match
            out.append((cases[i], rms, round(100 * (1 - rms / threshold), 1)))
            
    # Return sorted list of matching cases
    return out
