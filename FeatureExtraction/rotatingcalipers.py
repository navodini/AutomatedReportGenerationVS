# convex hull (Graham scan by x-coordinate) and diameter of a set of points
# David Eppstein, UC Irvine, 7 Mar 2002

from math import sqrt
import numpy as np
import skimage.morphology

def orientation(p, q, r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1])

def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2], U[-1], p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2], L[-1], p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U, L

def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
    between two parallel lines that touch one point each, and yields the sequence
    of pairs of points touched by each pair of lines.'''
    U, L = hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i], L[j]

        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1:
            j -= 1
        elif j == 0:
            i += 1

        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i + 1][1] - U[i][1]) * (L[j][0] - L[j - 1][0]) > \
                (L[j][1] - L[j - 1][1]) * (U[i + 1][0] - U[i][0]):
            i += 1
        else:
            j -= 1

def distance_3d(p1, p2, voxelspacing):
    return np.sum((((np.asarray(p1) - np.asarray(p2)) * voxelspacing) ** 2), axis=0)

def min_max_feret(Points, voxelspacing):
    '''Given a list of 2d points, returns the maximum feret diameter.'''
    squared_distance_per_pair = [
        (((p[0] - q[0]) * voxelspacing[0]) ** 2 + ((p[1] - q[1]) * voxelspacing[1]) ** 2, (p, q))
        for p, q in rotatingCalipers(Points)]
    if len(squared_distance_per_pair) > 0:
        max_feret_sq, max_feret_pair = max(squared_distance_per_pair)
        return sqrt(max_feret_sq), max_feret_pair
    else:
        return 0, [0, 0]

def min_max_feret_2boundaries(Points_extra, Points_intra, voxelspacing):
    '''Given a list of 2d points, returns the minimum and maximum feret diameters.'''
    squared_distance_per_pair = []
    coord_per_pair = []
    for p in Points_intra:
        for q in Points_extra:
            squared_distance_per_pair.append(((p[0]-q[0])*voxelspacing[0])**2 + ((p[1]-q[1])*voxelspacing[1])**2)
            coord_per_pair.append([p, q])

    if len(squared_distance_per_pair) > 0:
        max_feret_sq = max(squared_distance_per_pair)
        idx_max_feret_sq = np.where(squared_distance_per_pair == np.amax(squared_distance_per_pair))[0][0]
        return sqrt(max_feret_sq), coord_per_pair[idx_max_feret_sq]
    else:
        return 0, [0, 0]

def min_max_feret_3D(Points, Points3D, i, voxelspacing):
    '''Given a list of 2d points, returns the minimum and maximum feret diameters.'''
    squared_distance_per_pair = []
    for p, q in rotatingCalipers(Points):
        if len(p) < 3:
            p.append(i)
            if len(q) < 3:
                q.append(i)

        if len(q) < 3:
            q.append(i)
            if len(p) < 3:
                p.append(i)

        squared_distance_per_pair.append((distance_3d(p, q, voxelspacing), (p, q)))
        for point in Points3D:
            if point[2] != i:
                squared_distance_per_pair.append((distance_3d(p, point, voxelspacing), (p, point)))
                squared_distance_per_pair.append((distance_3d(q, point, voxelspacing), (q, point)))

    if len(squared_distance_per_pair) > 0:
        max_feret_sq, max_feret_pair = max(squared_distance_per_pair)
        return sqrt(max_feret_sq), max_feret_pair
    else:
        return 0, [0, 0, 0]

def get_min_max_feret_from_mask(mask_im, voxelspacing):
    """Given a binary mask, calculate the minimum and maximum feret diameter of the foreground object."""
    outline = skimage.segmentation.find_boundaries(mask_im.astype(np.uint8), connectivity=1, mode='inner',
                                                   background=0).astype(np.uint8)
    boundary_points = np.argwhere(outline > 0)
    boundary_point_list = list(map(list, list(boundary_points)))
    return min_max_feret(boundary_point_list, voxelspacing)

def get_min_max_feret_from_boundary(mask_intra, mask_extra, voxelspacing):
    """Given binary masks for intra and extra boundaries, calculate the minimum and maximum feret diameter."""
    outline_extra = skimage.segmentation.find_boundaries(mask_extra.astype(np.uint8), connectivity=1, mode='extra',
                                                         background=0).astype(np.uint8)
    outline_intra = skimage.segmentation.find_boundaries(mask_intra.astype(np.uint8), connectivity=1, mode='inner',
                                                         background=0).astype(np.uint8)
    outline_extra[outline_intra == 1] = 0
    outline_intra[outline_extra == 1] = 0
    boundary_points_extra = np.argwhere(outline_extra > 0)
    boundary_points_intra = np.argwhere(outline_intra > 0)
    boundary_point_list_extra = list(map(list, list(boundary_points_extra)))
    boundary_point_list_intra = list(map(list, list(boundary_points_intra)))
    return min_max_feret_2boundaries(boundary_point_list_extra, boundary_point_list_intra, voxelspacing)

def get_min_max_feret_from_mask_3D(mask_im_slice, mask_im, i, voxelspacing):
    """Given a binary mask slice and a 3D mask, calculate the minimum and maximum feret diameter of the foreground object."""
    outline_slice = skimage.segmentation.find_boundaries(mask_im_slice, connectivity=1, mode='inner',
                                                         background=0).astype(np.uint8)
    outline_3D = skimage.segmentation.find_boundaries(mask_im, connectivity=1, mode='inner', background=0).astype(
        np.uint8)
    boundary_points_slice = np.argwhere(outline_slice > 0)
    boundary_points_3D = np.argwhere(outline_3D > 0)
    boundary_point_list = list(map(list, list(boundary_points_slice)))
    boundary_point_list_3D = list(map(list, list(boundary_points_3D)))
    return min_max_feret_3D(boundary_point_list, boundary_point_list_3D, i, voxelspacing)
