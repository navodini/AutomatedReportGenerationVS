import numpy as np
import copy

def volume_mm(mask_, voxelspacing, case='intra'):
    """
    Calculate the volume of a binary mask in cubic millimeters.

    Parameters:
        mask_ (numpy.ndarray): Binary mask.
        voxelspacing (tuple): Voxel spacing in each dimension.
        case (str): Case type ('extra', 'whole', 'intra').

    Returns:
        float: Volume of the binary mask in cubic millimeters.
    """
    mask = copy.deepcopy(mask_)

    # Adjust the mask based on the specified case
    if case == 'extra':
        mask[mask == 2] = 0
    elif case == 'whole':
        mask[mask > 0] = 1
    elif case == 'intra':
        mask[mask == 1] = 0
        mask[mask > 0] = 1

    # Calculate the voxel volume and the total volume of the mask
    voxel = np.prod(voxelspacing)
    vol = round(voxel * np.sum(mask), 3)

    return vol