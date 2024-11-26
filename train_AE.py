"""
File: train_AE.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
    Pre-train the local autoencoder to predict regions at 5-year
"""


def read_copdgene_p1p2(data_path):
    # Data Normalization
    mean = -778.8
    std = 305.1

    # Read the data Huge file
    fname_db = 'COPDGene_emphysema_progression_DB_train_32'
    fname = os.path.join(data_path, fname_db + '.h5')

    hdf = h5py.File(fname, 'r')

    lm_class = 'lm1'

    fname = os.path.join(data_path,
                         fname_db + '_' + lm_class + '_classification.npy')
    lm1_label = np.load(fname).astype(np.int16)

    # delta == 1 : Emph progression
    delta_hu = np.load(os.path.join(data_path, fname_db + '_HU.npy'))
    emph_prog = delta_hu == 1

    n_subjects = hdf['ct1_patches'].shape[0]

    ids_fname = os.path.join(
        data_path, 'COPDGene_emphysema_progression_DB_train_id_subjects.npy')
    ids = np.load(ids_fname)

    # Training with the 70% of the data, the 30% is used as validation
    n_train = int(.7 * n_subjects)
    ids_train = ids[:n_train]
    ids_val = ids[n_train:]
    ids_train = np.sort(ids_train)
    ids_val = np.sort(ids_val)

    lm1l_train = lm1_label[ids_train]
    lm1l_val = lm1_label[ids_val]
    # Exclude undefined patches
    ids_ok_train = np.where(lm1l_train > 0)
    ids_ok_val = np.where(lm1l_val > 0)

    label_train = lm1l_train[ids_ok_train]
    label_val = lm1l_val[ids_ok_val]

    emph_prog_train = emph_prog[ids_train]
    emph_prog_val = emph_prog[ids_val]
    emph_prog_train = emph_prog_train[ids_ok_train]
    emph_prog_val = emph_prog_val[ids_ok_val]

    key = 'ct1_patches'

    ct = hdf[key][:]
    ct_train = ct[ids_train]
    ct_val = ct[ids_val]
    # Free memory
    del ct
    x_train = ct_train[ids_ok_train].astype(np.float32)
    x_val = ct_val[ids_ok_val].astype(np.float32)

    lm = hdf['lm1_patches'][:]
    lm_train = lm[ids_train]
    lm_val = lm[ids_val]
    # Free memory
    del lm
    lm_train = lm_train[ids_ok_train]
    lm_val = lm_val[ids_ok_val]

    # Data Normalization
    # Train data
    x_train -= mean
    x_train /= std
    # Val data
    x_val -= mean
    x_val /= std

    ct = hdf['ct2_patches'][:]
    ct_train = ct[ids_train]
    ct_val = ct[ids_val]
    # Free memory
    del ct

    y_train = ct_train[ids_ok_train].astype(np.float32)
    y_val = ct_val[ids_ok_val].astype(np.float32)
    # Free memory
    del ct_train
    del ct_val

    # Data Normalization
    # Train data
    y_train -= mean
    y_train /= std
    # Val data
    y_val -= mean
    y_val /= std

    data_train = (x_train, y_train, label_train, lm_train, emph_prog_train)
    data_val = (x_val, y_val, label_val, lm_val, emph_prog_val)
    return data_train, data_val


if __name__ == '__main__':
    import os
    import h5py
    from pathlib import Path
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=int,
                        default=0)
    parser.add_argument('-v',
                        '--verbose',
                        help="Verbose level",
                        type=int,
                        default=2)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=100,
                        help="Number of Epochs")
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=20,
                        help="Batch size")

    parser.add_argument('-lr_ae', '--lr_ae', type=float, default=5e-5)
    parser.add_argument('-da',
                        '--daugmention',
                        action='store_true',
                        help="Data Augmentation")

    # Read data and exclude Undefine patches
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # TF - Keras
    import tensorflow as tf
    # Custom
    import nn
    import utils

    local_data_path = '/data'
    models_path = 'models'

    subtype_weights = utils.tools.get_emph_subtype_weights()

    @tf.function
    def compute_ae_loss(y_batch, lm_batch, ep_batch, y_pred):

        lung_mask = tf.cast(y_batch < 0, tf.float32)
        ep = tf.cast(ep_batch, tf.float32)

        # Data Normalization
        mean = -778.8
        std = 305.1

        emph_th = (-950 - mean) / std
        emph_mask = tf.cast(y_batch <= emph_th, tf.float32)
        # Penalize emphysema progression lung tissue
        ae_loss = 100 * ae_loss_fn(y_batch * ep, y_pred * ep)
        # Focuse on the Emphysema
        ae_loss += 100 * ae_loss_fn(y_batch * emph_mask, y_pred * emph_mask)
        # Focuse on lung tissue
        ae_loss += 10 * ae_loss_fn(y_batch * lung_mask, y_pred * lung_mask)
        ae_loss += ae_loss_fn(y_batch, y_pred)

        # Subtype customization
        for code, weight in subtype_weights.items():
            mask = tf.cast(lm_batch == code, tf.float32)
            ae_loss += 10 * weight * ae_loss_fn(y_batch * mask, y_pred * mask)

        return ae_loss

    @tf.function
    def train_step_ae(x_batch, y_batch, lm_batch, ep_batch):

        # ---------------Train the AE -------------
        # Step 1: Train reconstruction
        with tf.GradientTape() as tape:
            y_pred = ae(x_batch, training=True)
            # Penalize emphysema progression lung tissue
            ae_loss = compute_ae_loss(y_batch, lm_batch, ep_batch, y_pred)
            # add to the loss other losses in the layer such as regularization
            ae_loss += sum(ae.losses)
        grads = tape.gradient(ae_loss, ae.trainable_variables)
        ae_optimizer.apply_gradients(zip(grads, ae.trainable_variables))
        # Update the metric
        for m in ae_metrics:
            m.update_state(y_batch, y_pred)

        return ae_loss

    @tf.function
    def train_step(batch):
        """
        patches: tensor data
        """
        x_batch, y_batch, yl_batch, lm_batch, ep_batch = batch
        # ------------ Step 1: Train the Reconstruction ---------
        ae_loss = train_step_ae(x_batch, y_batch, lm_batch, ep_batch)

        return ae_loss

    @tf.function
    def test_step_ae(x_batch, y_batch, lm_batch, ep_batch):
        y_pred = ae(x_batch, training=False)
        ae_loss_val = compute_ae_loss(y_batch, lm_batch, ep_batch, y_pred)
        # Update the metric
        for m in ae_metrics:
            m.update_state(y_batch, y_pred)

        return ae_loss_val

    def train(train, val, epochs=1000, batch_size=30, verbose=2):
        hist = {
            ae_loss_fn.name: [],
            'val_' + ae_loss_fn.name: [],
        }

        for m in ae_metrics:
            hist[m.name] = []
            hist['val_' + m.name] = []

        x_train, y_train, yl_train, lm_train, ep_train = train
        x_val, y_val, yl_val, lm_val, ep_val = val

        npatches = len(x_train)
        indexes = np.arange(npatches)
        niter_train = int(npatches / batch_size)
        npatches_val = len(x_val)
        niter_val = int(npatches_val / batch_size)

        for epoch in range(epochs):
            np.random.shuffle(indexes)
            print('Epoch {}/{}'.format(epoch + 1, epochs), flush=True)
            prog_bar = tf.keras.utils.Progbar(niter_train,
                                              width=15,
                                              verbose=verbose)
            for step in range(niter_train):
                ids = indexes[step * batch_size:(step + 1) * batch_size]
                x_batch = x_train[ids]
                y_batch = y_train[ids]
                yl_batch = yl_train[ids]
                if lm_train is None:
                    lm_batch = None
                else:
                    lm_batch = lm_train[ids]

                if ep_train is None:
                    ep_batch = None
                else:
                    ep_batch = ep_train[ids]

                # Data Augmentation
                if args.daugmention:
                    flip_0 = np.random.rand()
                    flip_1 = np.random.rand()
                    transpose = np.random.rand()
                    if flip_0 > .5:
                        # How many samples we will flip
                        n = np.random.randint(x_batch.shape[0])
                        # Which ones
                        s2f = np.r_[0:len(x_batch)]
                        np.random.shuffle(s2f)
                        s2f = s2f[:n]
                        x_batch[s2f, :, :] = x_batch[s2f, ::-1, :]
                        y_batch[s2f, :, :] = y_batch[s2f, ::-1, :]

                        if lm_batch is not None:
                            lm_batch[s2f, :, :] = lm_batch[s2f, ::-1, :]
                        if ep_batch is not None:
                            ep_batch[s2f, :, :] = ep_batch[s2f, ::-1, :]

                    if flip_1 > .5:
                        # How many samples we will flip
                        n = np.random.randint(x_batch.shape[0])
                        # Which ones
                        s2f = np.r_[0:len(x_batch)]
                        np.random.shuffle(s2f)
                        s2f = s2f[:n]
                        x_batch[s2f, :, :] = x_batch[s2f, :, ::-1]
                        y_batch[s2f, :, :] = y_batch[s2f, :, ::-1]
                        if lm_batch is not None:
                            lm_batch[s2f, :, :] = lm_batch[s2f, :, ::-1]
                        if ep_batch is not None:
                            ep_batch[s2f, :, :] = ep_batch[s2f, :, ::-1]
                    if transpose > .5:
                        # How many samples we will flip
                        n = np.random.randint(x_batch.shape[0])
                        # Which ones
                        s2f = np.r_[0:len(x_batch)]
                        np.random.shuffle(s2f)
                        s2f = s2f[:n]
                        x_batch[s2f, :, :] = np.transpose(x_batch[s2f, :, :],
                                                          axes=(0, 2, 1))
                        y_batch[s2f, :, :] = np.transpose(y_batch[s2f, :, :],
                                                          axes=(0, 2, 1))
                        if lm_batch is not None:
                            lm_batch[s2f, :, :] = np.transpose(
                                lm_batch[s2f, :, :], axes=(0, 2, 1))
                        if ep_batch is not None:
                            ep_batch[s2f, :, :] = np.transpose(
                                ep_batch[s2f, :, :], axes=(0, 2, 1))

                x_batch = x_batch[..., np.newaxis]
                y_batch = y_batch[..., np.newaxis]
                if lm_batch is not None:
                    lm_batch = lm_batch[..., np.newaxis]
                if ep_batch is not None:
                    ep_batch = ep_batch[..., np.newaxis]

                train_batch = (x_batch, y_batch, yl_batch, lm_batch, ep_batch)
                ae_loss = train_step(train_batch)
                ae_loss = ae_loss.numpy()
                log_values = [(ae_loss_fn.name, ae_loss)]

                # Metrics
                for m in ae_metrics:
                    val = m.result().numpy()
                    log_values.append((m.name, val))

                prog_bar.update(step, values=log_values)

            # Reset the metrics
            for m in ae_metrics:
                m.reset_states()

            # Loss and metric reconstruction on the validation dataset
            ae_loss_val = 0
            for step_val in range(niter_val):
                # Sequential order of validation data there is no need to shuffle
                ids = np.r_[step_val * batch_size:(step_val + 1) * batch_size]
                if lm_batch is not None:
                    lm_val_batch = lm_val[ids][..., np.newaxis]
                else:
                    lm_val_batch = None

                if ep_batch is not None:
                    ep_val_batch = ep_val[ids][..., np.newaxis]
                else:
                    ep_val_batch = None

                ct_loss_val = test_step_ae(x_val[ids][..., np.newaxis],
                                           y_val[ids][..., np.newaxis],
                                           lm_val_batch, ep_val_batch)
                ae_loss_val += ct_loss_val.numpy()
            ae_loss_val /= niter_val
            log_values = [('val_' + ae_loss_fn.name, ae_loss_val)]
            for m in ae_metrics:
                val = m.result().numpy()
                log_values.append(('val_' + m.name, val))
            # Progress Bar update
            prog_bar.update(niter_train, values=log_values)
            # Reset the metrics
            for m in ae_metrics:
                m.reset_states()

            # Save history
            logged_values = prog_bar._values
            for k in logged_values:
                hist[k].append(logged_values[k][0] / logged_values[k][1])

        return hist

    args.gpus = tf.config.list_physical_devices('GPU')

    data_train, data_val = read_copdgene_p1p2(local_data_path)
    #  data_train = (x_train, y_train, label_train, lm_train, emph_prog_train)
    #  data_val = (x_val, y_val, label_val, lm_val, emph_prog_val)

    input_shape = data_train[0].shape[1:] + (1, )

    latent_dim = 300
    m_name = 'ae_pred_2D_st-loss_d0_z300_e20_swish_v10_lrAE5E-05_0_cn4'
    ae = nn.local_models.ae(input_shape, latent_dim, activation='swish')

    save_folder = os.path.join(models_path, m_name)
    print('Saving files in: {}'.format(save_folder), flush=True)
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    ae_loss_fn = tf.keras.losses.MeanAbsoluteError(name='ae_mae_loss')
    print('AE Loss: {}'.format(ae_loss_fn.name), flush=True)
    ae_metrics = [
        tf.keras.metrics.MeanSquaredError(name='mse'),
    ]
    ae_optimizer = tf.keras.optimizers.Adam(lr=args.lr_ae)

    hist = train(data_train,
                 data_val,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 verbose=args.verbose)

    ae.save(save_folder + '/' + m_name + '_ae', save_format='tf')
    np.save(save_folder + '/' + m_name + '_hist.npy', hist)

    print('Done !', flush=True)
