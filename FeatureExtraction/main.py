import os
import SimpleITK
import nibabel as nib
import numpy as np
import cupy
import glob
import json
from numpyencoder import NumpyEncoder
from tqdm import tqdm
from datetime import date
import nilearn.image as image
import utils
import VolumeMeasurements
import LinearMeasurements
from radiomics import shape

subjects_path  = '../SampleCase/MRI/'
list_files = sorted(glob.glob(subjects_path + '/*/*/*_seg.nii.gz'))
json_dir = '../SampleCase/features_json/'

if not os.path.exists(json_dir):
    os.makedirs(json_dir)

Patient_id_init = 0

for file_path in tqdm(list_files):
    Patient_id = os.path.dirname(file_path).split('/')[3]
    Time_point = os.path.dirname(file_path).split('/')[4]
    MRI_modality = os.path.basename(file_path).split('_')[0]
    Spliseg = nib.load(file_path)
    sitk_mask = SimpleITK.ReadImage(file_path)
    orig_mask = Spliseg.get_fdata()
    spliseg_im = image.largest_connected_component_img(Spliseg).get_fdata()
    spliseg_im[orig_mask == 2] = 2
    voxelspacing = Spliseg.header['pixdim'][1:4]
    splitseg_cp = cupy.asarray(spliseg_im)
    splitplane_im = utils._get_plane(splitseg_cp).cpu().numpy()

    positive_planes = [spliseg_im[:,:,ix].any() > 0 for ix in range(spliseg_im.shape[2])].count(True)

    if np.count_nonzero(spliseg_im) < 10:
        plane_idx_max = np.where([spliseg_im[:,:,ix].any() > 0 for ix in range(spliseg_im.shape[2])])[0]
        markupPoint = [spliseg_im.shape[0]/2, spliseg_im.shape[1]/2, plane_idx_max[0]]
        coords_exax = [[(spliseg_im.shape[0]/2)+3, (spliseg_im.shape[1]/2)-4, plane_idx_max[0]], [(spliseg_im.shape[0]/2)-1, (spliseg_im.shape[1]/2)+2, plane_idx_max[0]]]
        single_voxel_vol = (spliseg_im == 2).astype(int)
        boundary_bb = utils.get_boundingbox(spliseg_im, voxelspacing)
        boundary_bb_extended = utils.get_boundingbox_extended(spliseg_im, voxelspacing)
#         boundary_bb = None
#         boundary_bb_extended = None
        max_3D, coord_3D = None, None
        max_exax, plane_idx = None, None
        max_ax, coord_maxax = None, None
        max_IAM, coords_IAM = None, None
        vol_extra, vol_intra, vol_whole = 0.0, 0.0, 0.0
    else:
        vol_extra = VolumeMeasurements.volume_mm(spliseg_im,voxelspacing,case='extra')
        vol_intra = VolumeMeasurements.volume_mm(spliseg_im,voxelspacing,case='intra')
        vol_whole = VolumeMeasurements.volume_mm(spliseg_im,voxelspacing,case='whole')

        print(vol_intra,vol_extra,vol_whole)
        if vol_extra > 0 and vol_intra > 0:
            boundary_bb = utils.get_boundingbox(spliseg_im, voxelspacing)
            boundary_bb_extended = utils.get_boundingbox_extended(spliseg_im, voxelspacing)
            max_pyradiomics = shape.RadiomicsShape(sitk_mask,sitk_mask)
            max_IAM, coords_IAM, plane_idx = LinearMeasurements.longest_IAM_axial(spliseg_im,voxelspacing)

            max_para_extra, coords_para_extra = LinearMeasurements.return_parallel_dist(splitplane_im, spliseg_im, voxelspacing, plane_idx, case= 'extra')
            max_para_intra, coords_para_intra = LinearMeasurements.return_parallel_dist(splitplane_im,spliseg_im,voxelspacing, plane_idx, case= 'intra')
            max_perpend, coords_perpend = LinearMeasurements.return_perpend_dist(splitplane_im,spliseg_im,voxelspacing,plane_idx)
            if (max_para_intra <= max_para_extra):
                if max_perpend >= 2:
                    max_exax, coords_exax, plane_idx_max = LinearMeasurements.return_Max_AxialPlane(spliseg_im, voxelspacing, case = 'extra')
                    max_ax, coord_maxax = LinearMeasurements.longest_axial(spliseg_im,voxelspacing,plane_idx_max, case = 'whole')
                    max_3D, coord_3D = LinearMeasurements.longest_3D(spliseg_im, voxelspacing, case='extra')
                    markupPoint = [np.random.randint(spliseg_im.shape[0],size=1)[0],np.random.randint(spliseg_im.shape[1],size=1)[0],plane_idx_max]
                elif max_perpend < 2:
                    max_exax, coords_exax, plane_idx_max = None, None, None
                    max_ax, coord_maxax, plane_idx_max = LinearMeasurements.return_Max_AxialPlane(spliseg_im,
                                                                                                  voxelspacing,
                                                                                                  case='whole')
                    max_3D, coord_3D = LinearMeasurements.longest_3D(spliseg_im, voxelspacing, case='whole')
                    markupPoint = [np.random.randint(spliseg_im.shape[0], size=1)[0],
                                   np.random.randint(spliseg_im.shape[1], size=1)[0], plane_idx_max]
            else:
                max_exax, coords_exax, plane_idx_max = None, None, None
                max_ax, coord_maxax, plane_idx_max = LinearMeasurements.return_Max_AxialPlane(spliseg_im, voxelspacing, case = 'whole')
                max_3D, coord_3D = LinearMeasurements.longest_3D(spliseg_im, voxelspacing, case='whole')
                markupPoint = [np.random.randint(spliseg_im.shape[0],size=1)[0],np.random.randint(spliseg_im.shape[1],size=1)[0],plane_idx_max]
        elif vol_intra == 0 and vol_extra > 0:
            boundary_bb = utils.get_boundingbox(spliseg_im, voxelspacing)
            boundary_bb_extended = utils.get_boundingbox_extended(spliseg_im, voxelspacing)

            max_3D, coord_3D = LinearMeasurements.longest_3D(spliseg_im, voxelspacing, case='extra')

            max_exax, coords_exax, plane_idx_max = LinearMeasurements.return_Max_AxialPlane(spliseg_im, voxelspacing, case = 'extra')
            max_ax, coord_maxax = max_exax, coords_exax
            max_IAM, coords_IAM = max_exax, coords_exax
            markupPoint = [np.random.randint(spliseg_im.shape[0], size=1)[0],
                           np.random.randint(spliseg_im.shape[1], size=1)[0], plane_idx_max]
            max_para_extra, coords_para_extra = None, None
            max_para_intra, coords_para_intra = None, None
            max_perpend, coords_perpend = None, None
        elif vol_extra == 0 and vol_intra > 0:
            boundary_bb = utils.get_boundingbox(spliseg_im, voxelspacing)
            boundary_bb_extended = utils.get_boundingbox_extended(spliseg_im, voxelspacing)

            max_3D, coord_3D = LinearMeasurements.longest_3D(spliseg_im, voxelspacing, case='whole')

            max_ax, coord_maxax, plane_idx_max = LinearMeasurements.return_Max_AxialPlane(spliseg_im, voxelspacing, case = 'whole')
            max_exax, coords_exax = None, None
            max_IAM, coords_IAM = max_ax, coord_maxax
            markupPoint = [np.random.randint(spliseg_im.shape[0],size=1)[0],np.random.randint(spliseg_im.shape[1],size=1)[0],plane_idx_max]
            max_para_extra, coords_para_extra = None, None
            max_para_intra, coords_para_intra = None, None
            max_perpend, coords_perpend = None, None
        else:
            boundary_bb = None
            boundary_bb_extended = None
            max_3D, coord_3D = None, None
            max_exax, coords_exax, plane_idx = None, None, None
            max_ax, coord_maxax = None, None
            max_IAM, coords_IAM = None, None
            markupPoint = None
            max_para_extra, coords_para_extra = None, None
            max_para_intra, coords_para_intra = None, None
            max_perpend, coords_perpend = None, None
    dictionary = {
    "ID": Patient_id,
    "Time_point": Time_point,
    "MRI_modality":MRI_modality,
    "Positive_planes":positive_planes,
    "boundary_bb":boundary_bb,
    "boundary_bb_extended":boundary_bb_extended,
    "max3dDiameter": max_3D,
    "max3dCoordinates":coord_3D,
    "2dFeaturePlane":plane_idx_max,
    "markupPoint":markupPoint,
    "maxAxialDiameter": max_ax,
    "maxAxialDiameterCoordinates":coord_maxax,
    "maxExtraMeatalDiameter": max_exax,
    "maxExtraMeatalDiameterCoordinates":coords_exax,
    "maxIAMDiameter": max_IAM,
    "maxIAMDiameterCoordinates":coords_IAM,
    "VolumeExtra":vol_extra,
    "VolumeIntra":vol_intra,
    "VolumeWhole":vol_whole,
    "maxParaIntraDiameter":max_para_intra,
    "maxParaIntraDiameterCoordinates":coords_para_intra,
    "maxParaExtraDiameter":max_para_extra,
    "maxParaExtraDiameterCoordinates":coords_para_extra,
    "maxPerpendExtraDiameter": max_perpend,
    "maxPerpendExtraDiameterCoordinates":coords_perpend
    }
    with open(os.path.join(json_dir,'%s_%s_features.json'%(Patient_id,Time_point)), 'w') as f:
        json.dump(dictionary, f, indent=4, sort_keys=False,
                  separators=(', ', ': '), ensure_ascii=False,
                  cls=NumpyEncoder)
