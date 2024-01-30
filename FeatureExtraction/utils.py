import numpy as np
import torch
import cupy
from cupyx.scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
import copy

def get_boundingbox(img_, voxelspacing):
    """
    Get the bounding box of the binary image.

    Parameters:
        img_ (numpy.ndarray): Binary image.
        voxelspacing (tuple): Voxel spacing in each dimension.

    Returns:
        list: Bounding box coordinates.
    """
    img = copy.deepcopy(img_)
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    # Extend the bounding box by 10 voxels in each dimension
    list_b = [[rmin, rmax], [cmin, cmax], [zmin, zmax]]
    list_bb_l = [item[0] - (10 / voxelspacing[0]) for item in list_b]
    list_bb_u = [item[1] + (10 / voxelspacing[1]) for item in list_b]
    list_bb_lm = [max(0, n) for n in list_bb_l]
    list_bb_um = [min(img.shape[0], list_bb_u[0]), min(img.shape[1], list_bb_u[1]), min(img.shape[2], list_bb_u[2])]

    return [[list_bb_lm[0], list_bb_lm[1], list_bb_lm[2]], [list_bb_um[0], list_bb_um[1], list_bb_um[2]]]

def get_boundingbox_extended(img_, voxelspacing):
    """
    Get the extended bounding box of the binary image.

    Parameters:
        img_ (numpy.ndarray): Binary image.
        voxelspacing (tuple): Voxel spacing in each dimension.

    Returns:
        list: Extended bounding box coordinates.
    """
    img = copy.deepcopy(img_)
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    # Use the entire range along the first dimension
    rmin = 0
    rmax = img.shape[0]

    list_b = [[rmin, rmax], [cmin, cmax], [zmin, zmax]]
    list_bb_l = [item[0] - (12 / voxelspacing[0]) for item in list_b]
    list_bb_u = [item[1] + (12 / voxelspacing[1]) for item in list_b]
    list_bb_lm = [max(0, n) for n in list_bb_l]
    list_bb_um = [min(img.shape[0], list_bb_u[0]), min(img.shape[1], list_bb_u[1]), min(img.shape[2], list_bb_u[2])]

    return [[list_bb_lm[0], list_bb_lm[1], list_bb_lm[2]], [list_bb_um[0], list_bb_um[1], list_bb_um[2]]]

def _get_boundaries(vol, connectivity=1, background=0, mode='inner'):
    """
    Get the boundaries of the binary image.

    Parameters:
        vol (cupy.ndarray): Binary image.
        connectivity (int): Connectivity for binary operations.
        background (int): Background value.
        mode (str): Mode for boundary extraction ('inner' or 'outer').

    Returns:
        cupy.ndarray: Binary image containing boundaries.
    """
    ndim = vol.ndim
    footprint = generate_binary_structure(ndim, connectivity)
    boundaries = (binary_dilation(vol, footprint) != binary_erosion(vol, footprint))

    if mode == 'inner':
        foreground_image = (vol != background)
        boundaries &= foreground_image
    elif mode == 'outer':
        max_label = np.iinfo(vol.dtype).max
        background_image = (vol == background)
        footprint = generate_binary_structure(ndim, ndim)
        inverted_background = cupy.array(vol, copy=True)
        inverted_background[background_image] = max_label
        adjacent_objects = ((binary_dilation(vol, footprint) != binary_erosion(inverted_background, footprint)) & ~background_image)
        boundaries &= (background_image | adjacent_objects)

    return boundaries

def _get_plane(vol, connectivity=1):
    """
    Get a binary image representing the plane between two classes in the volume.

    Parameters:
        vol (cupy.ndarray): Volumetric data with two classes (1 and 2).
        connectivity (int): Connectivity for binary operations.

    Returns:
        torch.Tensor: Binary image representing the plane.
    """
    vol_in = cupy.zeros_like(vol)
    vol_out = cupy.zeros_like(vol)
    vol_in[vol == 2] = 1
    vol_out[vol == 1] = 1
    vol_in = vol_in.astype(int)
    vol_out = vol_out.astype(int)

    # Get boundaries for each class
    bound_in = _get_boundaries(vol_in, connectivity=1, background=0, mode='outer').astype(cupy.uint8)
    bound_out = _get_boundaries(vol_out, connectivity=1, background=0, mode='inner').astype(cupy.uint8)

    # Find the plane between the two classes
    vol_ref = cupy.logical_and(bound_in == 1, bound_out == 1).astype(cupy.uint8)
    vol_ref = torch.tensor(vol_ref, device='cuda:0')

    return vol_ref