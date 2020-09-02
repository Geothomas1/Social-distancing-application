import itertools

import numpy as np


def find_min_distance(centers, min_dist=100):
    '''
    return min euclidean distance between predicted anchor boxes
    '''
    comp = list(itertools.combinations(centers, 2))
    critical_distances = {}
    for pts in comp:
        ecdist = np.linalg.norm(np.asarray(pts[0])-np.asarray(pts[1]))
        if ecdist < min_dist:
            critical_distances.update({pts: ecdist})
    return critical_distances
