from math import sqrt
import numpy as np
import skimage.morphology
from sympy import Line, Point
import copy
import random
import rotatingcalipers


def longest_3D(mask_, voxelspacing, case='extra'):
    # Obtain the longest feret diameter in 3D along with the corresponding coordinates
    mask = copy.deepcopy(mask_)
    if case == 'extra':
        mask[mask == 2] = 0
    elif case == 'whole':
        mask[mask > 0] = 1

    # Calculate feret diameters for each slice in the z-direction
    max_all = [rotatingcalipers.get_min_max_feret_from_mask_3D(mask[:, :, i], mask, i, voxelspacing) for i in
               range(mask.shape[2])]

    # Find the index of the maximum feret diameter
    idx = np.argmax([item[0] for item in max_all])
    print(round(max_all[idx][0], 3), max_all[idx][1])
    return round(max_all[idx][0], 3), max_all[idx][1]


def longest_axial(mask_, voxelspacing, id_, case='extra'):
    # Obtain the longest feret diameter in a specific axial plane along with the corresponding coordinates
    mask = copy.deepcopy(mask_)
    if case == 'extra':
        mask[mask == 2] = 0
    elif case == 'whole':
        mask[mask > 0] = 1
    elif case == 'intra':
        mask[mask == 1] = 0
        mask[mask == 2] = 1

    # Calculate feret diameter for the specified axial plane
    max_all = rotatingcalipers.get_min_max_feret_from_mask(mask[:, :, id_], voxelspacing)
    list_coord = np.array(max_all[1])
    list_coord[:, 0].fill(id_)

    return round(max_all[0], 3), list_coord.tolist()


def longest_IAM_axial(mask_, voxelspacing):
    # Obtain the longest feret diameter in the IAM (Internal Acoustic Meatus) axial plane
    mask_extra = copy.deepcopy(mask_)
    mask_intra = copy.deepcopy(mask_)
    mask_extra[mask_extra == 2] = 0
    mask_intra[mask_intra == 1] = 0
    mask_intra[mask_intra == 2] = 1

    max_all_slices = []
    max_all_coordpair_slices = []
    index_list = []

    # Find positive planes with intra-class presence
    positive_planes = np.where([mask_intra[:, :, ix].any() > 0 for ix in range(mask_intra.shape[2])])[0]
    for id_ in positive_planes:
        # Calculate feret diameter for each positive intra-class axial plane
        max_all = rotatingcalipers.get_min_max_feret_from_boundary(mask_intra[:, :, id_], mask_extra[:, :, id_],
                                                                   voxelspacing)
        max_all_coordpair = np.array(max_all[1])
        if max_all[0] > 0:
            max_all_coordpair_ = [(np.append(x, id_)).tolist() for x in max_all_coordpair]
            max_all_slices.append(max_all[0])
            max_all_coordpair_slices.append(max_all_coordpair_)
            index_list.append(id_)

    # Find the index of the axial plane with the maximum feret diameter
    idx_slice = np.argmax(max_all_slices)
    IAM_coords = max_all_coordpair_slices[idx_slice]
    idx_plane = index_list[idx_slice]

    # Check if intra-class is present in the volume
    if mask_intra.any() == 1:
        return round(max_all_slices[idx_slice], 3), IAM_coords, idx_plane
    else:
        return None, None, None


def return_Max_AxialPlane(mask_, voxelspacing, case='extra'):
    # Obtain the axial plane with the maximum feret diameter in the entire volume
    mask = copy.deepcopy(mask_)
    if case == 'extra':
        mask[mask == 2] = 0
    elif case == 'whole':
        mask[mask > 0] = 1

    list_dist_plane = []
    list_coords_plane = []
    list_idx = []

    # Find positive planes in the volume
    positive_planes = np.where([mask_[:, :, ix].any() > 0 for ix in range(mask_.shape[2])])[0]
    for id_ in positive_planes:
        # Calculate feret diameter for each positive axial plane
        max_all = rotatingcalipers.get_min_max_feret_from_mask(mask[:, :, id_], voxelspacing)
        max_all_coordpair = np.array(max_all[1])
        if max_all[0] > 0:
            max_all_coordpair_ = [(np.append(x, id_)).tolist() for x in max_all_coordpair]
            list_dist_plane.append(max_all[0])
            list_coords_plane.append(max_all_coordpair_)
            list_idx.append(id_)

    # Check if positive axial planes are present
    if list_dist_plane:
        return round(max(list_dist_plane), 3), list_coords_plane[np.argmax(list_dist_plane)], list_idx[
            np.argmax(list_dist_plane)]
    else:
        return None, None, None


def return_parallel_dist(plane_im_, mask_, voxelspacing, plane_idx, case='extra'):
    # Return the parallel distance between two boundaries in a specified axial plane
    mask = copy.deepcopy(mask_)
    plane_im = copy.deepcopy(plane_im_)

    if case == 'extra':
        mask = (mask_ == 1).astype(int)
        mode_ = 'inner'
    elif case == 'intra':
        mask = (mask_ == 2).astype(int)
        mode_ = 'inner'

    # Find boundary points in the specified axial plane
    boundary_points = np.argwhere(plane_im[:, :, plane_idx] == 1)
    list_dist = []
    list_dist_coords = []

    # Check if there are enough boundary points
    if len(boundary_points) > 2:
        count = 0
        while len(list_dist) == 0 and count < len(boundary_points):
            # Select a reference point (position 1) and find the farthest point (position 2)
            nopos = boundary_points.tolist().copy()
            pos = nopos[count]
            count += 1
            nopos.remove(pos)
            distance_metric = lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2
            farthest = max(nopos, key=distance_metric)
            line = Line(Point(pos), Point(farthest))

            # Create binary mask for the specified axial plane
            T2_im_idx = mask[:, :, plane_idx]
            outline_idx = skimage.segmentation.find_boundaries(T2_im_idx, connectivity=1, mode=mode_, background=0).astype(
                np.uint8)
            boundary_points_idx = np.argwhere(outline_idx > 0)
            boundary_points_idx_list = boundary_points_idx.tolist()

            # Iterate through boundary points to find pairs with parallel distances
            for ix in boundary_points_idx_list:
                l2 = line.parallel_line(ix)
                noix = boundary_points_idx_list.copy()
                noix.remove(ix)

                for ix2 in noix:
                    if l2.contains(ix2):
                        line_parallel = Line(Point(ix), Point(ix2))

                        if line.is_parallel(line_parallel):
                            dist = sqrt(
                                ((ix[0] - ix2[0]) * voxelspacing[0]) ** 2 + ((ix[1] - ix2[1]) * voxelspacing[1]) ** 2)
                            list_dist.append(dist)
                            ix_ = np.append(ix, plane_idx)
                            ix2_ = np.append(ix2, plane_idx)
                            list_dist_coords.append([ix_, ix2_])

    # Check if parallel distances are found
    if list_dist:
        print(list_dist_coords[np.argmax(list_dist)])
        return round(max(list_dist), 3), list_dist_coords[np.argmax(list_dist)]
    else:
        return None, None


def return_perpend_dist(plane_im_, mask_, voxelspacing, idx):
    # Return the perpendicular distance between two boundaries in a specified axial plane
    mask = copy.deepcopy(mask_)
    plane_im = copy.deepcopy(plane_im_)
    mask[mask == 2] = 0

    # Find boundary points in the specified axial plane
    boundary_points = np.argwhere(plane_im[:, :, idx] == 1)
    list_dist = []
    list_dist_coords = []
    # Check if there are enough boundary points
    if len(boundary_points) > 2:
        count = 0
        while len(list_dist) == 0 and count < len(boundary_points):
            # Select a reference point (position 1) and find the farthest point (position 2)
            nopos = boundary_points.tolist().copy()
            pos = nopos[count]
            count += 1
            nopos.remove(pos)
            distance_metric = lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2
            farthest = max(nopos, key=distance_metric)
            line = Line(Point(pos), Point(farthest))

            # Randomly sample a point to create a perpendicular line
            random_points = np.array([random.choice(boundary_points)])
            # Create binary mask for the specified axial plane
            T2_im_idx = mask[:, :, idx]
            outline_idx = skimage.segmentation.find_boundaries(T2_im_idx, connectivity=1, mode='inner',
                                                               background=0).astype(np.uint8)
            boundary_points_idx = np.argwhere(outline_idx > 0)
            boundary_points_idx_list = boundary_points_idx.tolist()
            lineperp = line.perpendicular_line(random_points[0])
            # Iterate through boundary points to find pairs with perpendicular distances
            for ix in boundary_points_idx_list:
                l2 = lineperp.parallel_line(ix)
                noix = boundary_points_idx_list.copy()
                noix.remove(ix)
                for ix2 in noix:
                    if l2.contains(ix2):
                        line_parallel = Line(Point(ix), Point(ix2))
                        if lineperp.is_parallel(line_parallel):
                            dist = sqrt(
                                ((ix[0] - ix2[0]) * voxelspacing[0]) ** 2 + ((ix[1] - ix2[1]) * voxelspacing[1]) ** 2)
                            list_dist.append(dist)
                            ix_ = np.append(ix, idx)
                            ix2_ = np.append(ix2, idx)
                            list_dist_coords.append([ix_, ix2_])

    # Check if perpendicular distances are found
    if list_dist:
        print(list_dist_coords[np.argmax(list_dist)])
        return round(max(list_dist), 3), list_dist_coords[np.argmax(list_dist)]
    else:
        return None, None
