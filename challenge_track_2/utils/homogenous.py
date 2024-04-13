import numpy as np

def rot(H):
    return H[:3, :3]

def trans(H):
    return H[:3, 3]

def inverse(H):
    H_inv = np.eye(4)
    H_inv[0:3, 0:3] = rot(H).T
    H_inv[0:3, 3] = -rot(H).T @ trans(H)

    return H_inv