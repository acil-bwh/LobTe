"""
File: train_LobTe.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf


def read_data(data_path, outcome):
    # Fingerprint normalization (measured on the training dataset)
    norm = {'mean': 0.0186028, 'std': 2.86856385}

    dfname = 'fingerprint_copdgene_lobe_5DF_train_v9_z0.npz'
    deep_phenotypes = np.load('%s/%s' % (data_path, dfname), allow_pickle=True)

    lobe_names = [
        'RightSuperiorLobe', 'RightMiddleLobe', 'RightInferiorLobe',
        'LeftSuperiorLobe', 'LeftInferiorLobe'
    ]

    data = np.stack([deep_phenotypes[k] for k in lobe_names], axis=1)
    data = (data - norm['mean']) / norm['std']

    # Adding the channel axis
    X_data = data[..., np.newaxis]
    y_data = deep_phenotypes[outcome]

    return X_data, y_data


def trainLobTe(X_train,
               X_val,
               y_train,
               y_val,
               oname,
               nlayers,
               dmodel,
               d_dp_rep,
               nheads,
               dff,
               deepf,
               epochs,
               batch_size,
               dropout_rate=.25,
               patch_size=(300, 11),
               save_suffix='',
               save_model=True,
               models_path=''):
    import nn.schedules
    import nn.lobte

    model_name = 'FP_LobeTransformer_%s' % oname
    input_shape = X_train.shape[1:]
    model = nn.lobte.create_LobTe(input_shape,
                                  nlayers,
                                  dmodel,
                                  d_dp_rep,
                                  nheads,
                                  dff,
                                  deepf,
                                  oname,
                                  dropout_rate=dropout_rate,
                                  patch_size=patch_size,
                                  name=model_name)
    model.summary()

    # 1/4 of the epochs were a epoch has X_train.shape[0] / batch_size steps
    warmup_steps = (epochs // 4) * (X_train.shape[0] // batch_size)

    learning_rate = nn.schedules.TransformerSchedule(dmodel,
                                                     warmup_steps=warmup_steps)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    loss = {oname: tf.keras.losses.MeanSquaredError()}
    model.compile(optimizer=optimizer, loss=loss)
    # At the end of the training get the best model. Patient = epochs
    callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=int(epochs / 2),
                                                 restore_best_weights=True,
                                                 mode='min')

    callbacks = [tf.keras.callbacks.TerminateOnNaN(), callback1]

    suffix = 'nh%i_nl%i_dff%i_df%i_dpr%i_dm%i_e%i_lrNone_drop%.1E' % (
        nheads, nlayers, dff, deepf, d_dp_rep, dmodel, epochs, dropout_rate)

    save_folder = os.path.join(models_path, 'FP_models',
                               'FP_LobeTransformer_%s' % suffix, model_name)
    save_folder += save_suffix
    print('Saving files in: %s' % save_folder)

    hi = model.fit(X_train,
                   y_train,
                   epochs=epochs,
                   validation_data=(X_val, y_val),
                   callbacks=callbacks,
                   batch_size=batch_size,
                   verbose=1)
    # Saving models
    history = hi.history
    # Add the best epoc
    history['EarlyStopping'] = {
        callback1.monitor: callback1.best,
        'best_epoch': callback1.best_epoch
    }

    if save_model:
        print('Saving files in: {}'.format(save_folder))
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        save_model_filepath = save_folder + '/' + model.name
        model.save(save_model_filepath, save_format='tf')
        np.save(save_model_filepath + '_hist.npy', history)

    return model, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=int,
                        default=0)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=1000,
                        help="Number of Epochs")
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument('-cv',
                        '--cross_val',
                        help="K-Folding cross validation (0: no CV)",
                        type=int,
                        default=5)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    local_data_path = '/data'
    models_path = 'models'
    save_suffix = ''

    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) backend random seed
    # 3) `python` random seed
    tf.keras.utils.set_random_seed(100)

    # If using TensorFlow, this will make GPU ops as deterministic as possible,
    # but it will affect the overall performance, so be mindful of that.
    tf.config.experimental.enable_op_determinism()

    X_data, y_data = read_data(local_data_path)

    oname = 'Change_Adj_Density_plethy_P1_P2'
    nlayers = 1
    dmodel = 32
    d_dp_rep = 1
    nheads = 8
    dff = 32
    deepf = 5
    dropout_rate = 0.25
    patch_size = (300, 11)

    if args.cross_val > 0:
        from sklearn.model_selection import KFold
        cv_suffix = ''
        kf = KFold(n_splits=args.cross_val, shuffle=True, random_state=40)
        for i, (train_idx, val_idx) in enumerate(kf.split(X_data, y_data)):
            cv_suffix = '_%i' % i

            X_train = X_data[train_idx]
            y_train = y_data[train_idx]

            X_val = X_data[val_idx]
            y_val = y_data[val_idx]

            # Train and Save the model
            model, history = trainLobTe(X_train,
                                        X_val,
                                        y_train,
                                        y_val,
                                        oname,
                                        nlayers,
                                        dmodel,
                                        d_dp_rep,
                                        nheads,
                                        dff,
                                        deepf,
                                        args.lr,
                                        args.epochs,
                                        args.batch_size,
                                        dropout_rate=dropout_rate,
                                        patch_size=patch_size,
                                        models_path=models_path,
                                        save_suffix=save_suffix + cv_suffix)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_data,
                                                          y_data,
                                                          test_size=0.3,
                                                          random_state=42)

        model, history = trainLobTe(X_train,
                                    X_val,
                                    y_train,
                                    y_val,
                                    oname,
                                    nlayers,
                                    dmodel,
                                    d_dp_rep,
                                    nheads,
                                    dff,
                                    deepf,
                                    args.lr,
                                    args.epochs,
                                    args.batch_size,
                                    dropout_rate=dropout_rate,
                                    patch_size=patch_size,
                                    models_path=models_path,
                                    save_suffix=save_suffix)
