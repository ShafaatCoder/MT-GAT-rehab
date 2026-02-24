# util/distance_feature.py

import numpy as np

# Define key joint index pairs (from KIMORE layout)
DISTANCE_PAIRS = [
    (16, 32),  # Left shoulder ↔ Right shoulder
    (48, 64),  # Left hip ↔ Right hip
    (0, 80),   # Spine base ↔ Spine shoulder
    (16, 48),  # Left shoulder ↔ Left hip
    (32, 64),  # Right shoulder ↔ Right hip
]

def compute_distances(sequence, pairs=DISTANCE_PAIRS):
    """
    sequence: [timesteps, features] where features = flattened joint data
    Returns: [timesteps, num_distances]
    """
    n_frames = sequence.shape[0]
    distances = np.zeros((n_frames, len(pairs)))
    
    for t in range(n_frames):
        frame = sequence[t]
        for i, (a, b) in enumerate(pairs):
            joint_a = frame[a:a+3]   # x, y, z
            joint_b = frame[b:b+3]
            distances[t, i] = np.linalg.norm(joint_a - joint_b)
    
    return distances