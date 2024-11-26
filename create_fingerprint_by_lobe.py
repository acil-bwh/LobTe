"""
File: create_fingerprint_by_lobe.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from numba import njit, prange


@njit(parallel=True)
def get_fpz_pctl(zi, step=10):
    n = 100 // step + 1
    delta = step * .1
    ret_val = np.zeros((zi.shape[-1], n))
    for i in prange(n):
        ii = 10 * i * delta
        for j in prange(zi.shape[-1]):
            ret_val[j, i] = np.percentile(zi[:, j], ii)
    return ret_val


def get_patches_from_subject(I0, Im, stride=4, nlung_th=32 * 20):
    patch = np.lib.stride_tricks.sliding_window_view(I0, (32, 32), axis=(1, 2))
    x_data = patch[:, ::stride, ::stride]

    # Lung Mask
    m_patch = np.lib.stride_tricks.sliding_window_view(Im, (32, 32),
                                                       axis=(1, 2))
    m_data = m_patch[:, ::stride, ::stride]

    # Take regions with enough lung tissue
    nlung = m_data.sum(axis=(-2, -1))
    id_full = np.where(nlung >= nlung_th)
    x_data = x_data[id_full]
    return x_data, id_full


def predict_fingerprint(I0, Imask, model, batch_size=600):
    if I0.dtype != np.float64:
        I0 = I0.astype(np.float64)

    x_data, lm_data, id_full = utils.tools.get_patches_from_subject(I0,
                                                                    Imask,
                                                                    stride=4)
    # Patch Normalization
    #  fname = os.path.join(local_data_path, 'norm_patches.npz')
    #  norm = np.load(fname)
    # Data Normalization
    norm = {'mean': -778.8, 'std': 305.1}

    x_data -= norm['mean']
    x_data /= norm['std']

    z0 = model['enc'].predict(x_data[..., np.newaxis],
                              batch_size=batch_size,
                              verbose=2)

    step = 10
    fpz0 = get_fpz_pctl(z0, step)
    return z0, fpz0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=int,
                        default=0)
    parser.add_argument('-st',
                        '--study',
                        help="Study folder",
                        default='',
                        type=str)
    parser.add_argument('-df',
                        '--data_folder',
                        help="Data folder",
                        default='',
                        type=str)
    parser.add_argument('-sf',
                        '--save_folder',
                        help="Save folder",
                        default='',
                        type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    import tensorflow as tf
    import sys
    import utils

    lobe_codes = utils.tools.get_lobe_codes()

    sid = args.study.split('_')[0]
    data_folder = os.path.join(args.data_folder, sid, args.study)
    base_name = os.path.join(data_folder, args.study)
    ctname = base_name + '.nrrd'
    lname = base_name + '_lungLobeLabelMap.nrrd'

    if not os.path.isfile(ctname):
        print('File not found: %s' % ctname)
        sys.exit(1)
    if not os.path.isfile(lname):
        print('File not found: %s' % lname)
        sys.exit(1)

    ct = sitk.ReadImage(ctname)
    lmap = sitk.ReadImage(lname)
    I = sitk.GetArrayViewFromImage(ct)
    Im = sitk.GetArrayViewFromImage(lmap)

    # Load models
    model_path = 'models'
    m_name = 'dense_ae_hu_gt3_pretrain_pred_EmphProgression_balnew_daug_v10_st-lossV9_2Di_d0_z200_n300_swish_e100_lrAE1E-05_lrENN1E-05_lrNN1E-04_lrZ1E-04_l3_0_cn4'

    m_enc = os.path.join(model_path, m_name, m_name + '_enc')

    model = tf.keras.models.load_model(m_enc)

    save_folder = os.path.join(args.save_folder, sid, args.study)
    fpname = os.path.join(save_folder,
                          args.study + '_gt3_fingerprint_pctl_%s.npy')

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    for key, val in lobe_codes.items():
        Imi = (Im == key).astype(np.uint16)
        z, fpz = predict_fingerprint(I, Imi, model)
        fname_zi = fpname % val['Name']
        np.save(fname_zi, fpz)
