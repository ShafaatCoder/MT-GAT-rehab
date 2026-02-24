import numpy as np

# Define joint triplets for angle computation: (parent, joint, child)
# Indices match KIMORE joint order (0-based, every 4 columns: x,y,z,conf)
JOINT_ANGLE_PAIRS = [
    (16, 20, 24),  # Left Shoulder → Elbow → Wrist
    (32, 36, 40),  # Right Shoulder → Elbow → Wrist
    (48, 52, 56),  # Left Hip → Knee → Ankle
    (64, 68, 72),  # Right Hip → Knee → Ankle
]

def compute_angle(a, b, c):
    """Compute angle at joint b formed by points a-b-c"""
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba, axis=-1, keepdims=True)
    bc_norm = np.linalg.norm(bc, axis=-1, keepdims=True)
    cosine = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return angle  # radians

def extract_joint_angles(sequence):
    """
    sequence: [timesteps, features=100] where features are flattened [x,y,z,conf] for 25 joints
    Returns: [timesteps, num_angles]
    """
    num_timesteps = sequence.shape[0]
    coords = np.zeros((num_timesteps, 25, 3))  # 25 joints, ignore confidence
    for i in range(25):
        coords[:, i, :] = sequence[:, i * 4 : i * 4 + 3]

    angles = []
    for (a, b, c) in JOINT_ANGLE_PAIRS:
        angle = compute_angle(coords[:, a // 4], coords[:, b // 4], coords[:, c // 4])
        angles.append(angle)

    return np.stack(angles, axis=-1)  # shape [timesteps, num_angles]