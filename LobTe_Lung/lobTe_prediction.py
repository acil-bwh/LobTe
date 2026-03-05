"""
File: lobTe_prediction.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

if __name__ == '__main__':
    import os
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
    parser.add_argument('-cv',
                        '--cross_val',
                        help="K-Folding cross validation (0: no CV)",
                        type=int,
                        default=5)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    import tensorflow as tf
    import sys
    import utils
    import numpy as np
    import nn.schedules

    lobe_codes = utils.tools.get_lobe_codes()

    sid = args.study.split('_')[0]
    data_folder = os.path.join(args.data_folder, sid, args.study)
    base_name = os.path.join(data_folder, args.study)

    fplobes = []
    for key, val in lobe_codes.items():
        fpname = base_name + '_gt3_fingerprint_pctl_%s.npy'
        fpname_li = fpname % val['Name']
        if os.path.isfile(fpname_li):
            fpi = np.load(fpname_li)
            fplobes.append(fpi)
        else:
            print('File not found: %s' % fpname_li)
            sys.exit(1)

    fplobes = np.stack(fplobes, axis=0)
    # Fingerprint normalization (measured on the training dataset)
    norm = {'mean': 0.0186028, 'std': 2.86856385}

    data = (fplobes - norm['mean']) / norm['std']
    data = data[np.newaxis, ..., np.newaxis]

    # Load models
    models_path = 'models'
    oname = 'Change_Adj_Density_plethy_P1_P2'
    mname = 'FP_LobeTransformer_%s' % oname + '%s'
    mversion = 'FP_LobeTransformer_nh8_nl1_dff32_df5_dpr1_dm32_e1000_lrNone_drop2.5E-01'
    mfolder = os.path.join(models_path, 'FP_models', mversion, mname)

    custom_objects = {
        'TransformerSchedule': nn.schedules.TransformerSchedule,
    }

    yp = []
    if args.cross_val > 0:
        for i in args.cross_val:
            mfolderi = os.path.join(mfolder % '_%i' % i, mname % '')

            model = tf.keras.models.load_model(mfolderi,
                                               custom_objects=custom_objects)
            ypi = model.predict(data)
            yp.append(np.squeeze(ypi))
    else:
        mfolderi = os.path.join(mfolder % '', mname % '')
        model = tf.keras.models.load_model(mfolderi,
                                           custom_objects=custom_objects)
        ypi = model.predict(data)
        yp.append(np.squeeze(ypi))

    print('%s prediction: %s=%f' % (args.study, oname, np.mean(yp)))
