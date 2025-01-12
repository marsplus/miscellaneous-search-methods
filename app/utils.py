import random

import numpy as np


def generate_distance_matrix(N: int, seed: int = 42) -> list:
    """
    Generate a symmetric distance matrix for a TSP problem.

    Args:
    - N (int): The number of cities.
    - seed (int): Seed for random number generation (default: 42).

    Returns:
    - dist_matrix (list of list of int): Symmetric N x N distance matrix.
    """
    random.seed(seed)
    dist_matrix = [
        [0 if i == j else random.randint(10, 100) for j in range(N)] for i in range(N)
    ]
    for i in range(N):
        for j in range(i + 1, N):
            dist_matrix[j][i] = dist_matrix[i][j]

    return dist_matrix


def set_all_seeds(seed: int) -> None:
    """
    Set all seeds for reproducibility.

    Args:
        seed: Integer seed for random number generation
    """
    random.seed(seed)
    np.random.seed(seed)
