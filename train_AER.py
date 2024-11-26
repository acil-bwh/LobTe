"""
File: train_AER.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
    Train the local model to regress %Emph using a pretrained AE
"""


def read_data(data_path):
    # Data Normalization
    mean = -778.8
    std = 305.1

    fname_db = 'COPDGene_emphysema_progression_DB_train_32'
    fname = os.path.join(data_path, fname_db + '.h5')

    hdf = h5py.File(fname, 'r')

    dim = hdf['ct1_patches'].shape

    #  nclass = 2
    fname = os.path.join(data_path, fname_db + '_HU_2D_GT3.npy')
    classification = np.load(fname)

    lm_name = 'lm1'

    fname = os.path.join(data_path,
                         fname_db + '_' + lm_name + '_classification.npy')
    subtype_label = np.load(fname).astype(np.int16)

    n_subjects = dim[0]

    ids_fname = os.path.join(
        data_path, 'COPDGene_emphysema_progression_DB_train_id_subjects.npy')
    ids = np.load(ids_fname)

    # Training with the 70% of the data, the 30% is used as validation
    n_train = int(.7 * n_subjects)
    ids_train = ids[:n_train]
    ids_val = ids[n_train:]
    ids_train = np.sort(ids_train)
    ids_val = np.sort(ids_val)

    subtype_train = subtype_label[ids_train]
    subtype_val = subtype_label[ids_val]

    cl_train = classification[ids_train]
    cl_val = classification[ids_val]

    # NOTE: Probar si esto de excluir los indefinidos es lo que baja el
    # rendimiento respecto a la version OLD
    # Exclude undefined patches
    ids_ok_train = np.where(subtype_train > 0)
    ids_ok_val = np.where(subtype_val > 0)

    label_train = subtype_train[ids_ok_train]
    label_val = subtype_val[ids_ok_val]

    cl_train = cl_train[ids_ok_train]
    cl_val = cl_val[ids_ok_val]

    key = 'ct1_patches'
    ct = hdf[key][:]
    ct_train = ct[ids_train]
    ct_val = ct[ids_val]
    # Free memory
    del ct
    x_train = ct_train[ids_ok_train].astype(np.float32)
    x_val = ct_val[ids_ok_val].astype(np.float32)

    # Train data
    x_train -= mean
    x_train /= std
    # Val data
    x_val -= mean
    x_val /= std

    # Lung mask by subtype
    lm = hdf['lm1_patches'][:]
    lm_train = lm[ids_train]
    lm_val = lm[ids_val]
    # Free memory
    del lm
    lm_train = lm_train[ids_ok_train]
    lm_val = lm_val[ids_ok_val]

    # delta == 1 : Emph progression
    delta_hu = np.load(os.path.join(data_path, fname_db + '_HU.npy'))
    emph_prog = delta_hu == 1
    emph_prog_train = emph_prog[ids_train]
    emph_prog_val = emph_prog[ids_val]
    emph_prog_train = emph_prog_train[ids_ok_train]
    emph_prog_val = emph_prog_val[ids_ok_val]

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

    yl_train = cl_train
    yl_val = cl_val

    data_train = (x_train, y_train, yl_train, label_train, lm_train,
                  emph_prog_train)
    data_val = (x_val, y_val, yl_val, label_val, lm_val, emph_prog_val)
    return data_train, data_val


if __name__ == '__main__':
    import os
    from pathlib import Path
    import h5py
    import argparse
    import numpy as np

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
                        default=200,
                        help="Batch size")

    parser.add_argument('-lr_ae', '--lr_ae', type=float, default=1e-5)
    parser.add_argument('-lr_enc_nn', '--lr_enc_nn', type=float, default=1e-5)
    parser.add_argument('-lr_nn', '--lr_nn', type=float, default=1e-4)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    #  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    local_data_path = '/data'
    models_path = 'models'

    import tensorflow as tf

    import nn
    import utils
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
        # Penalize where predict emphysema and there is no Emph. Prog. So fail
        # to predict EP.
        emph_pred = tf.cast(y_pred <= emph_th, tf.float32)  # * (1 - ep)
        ae_loss += 100 * ae_loss_fn(y_batch * emph_pred, y_pred * emph_pred)

        # Subtype customization
        for code, weight in subtype_weights.items():
            mask = tf.cast(lm_batch == code, tf.float32)
            ae_loss += 10 * weight * ae_loss_fn(y_batch * mask, y_pred * mask)

        return ae_loss

    @tf.function
    def train_step_ae(x_batch, y_batch, lm_batch, ep_batch):
        """
        patches: tensor data
        """
        # ---------------Train the AE -------------
        # Step 1: Train reconstruction
        if args.lr_ae == 0:
            return 0.

        with tf.GradientTape() as tape:
            y_pred = dec(enc(x_batch, training=True), training=True)
            ae_loss = compute_ae_loss(y_batch, lm_batch, ep_batch, y_pred)
            #  ae_loss = ae_loss_fn(y_batch, y_pred)
            # add to the loss other losses in the layer such as regularization
            ae_loss += sum(enc.losses) + sum(dec.losses)
        grads = tape.gradient(
            ae_loss, [enc.trainable_variables, dec.trainable_variables])
        optimizer_ae.apply_gradients(zip(grads[0], enc.trainable_variables))
        optimizer_ae.apply_gradients(zip(grads[1], dec.trainable_variables))

        return ae_loss

    @tf.function
    def train_step_nn(x_batch, label_batch):
        with tf.GradientTape() as tape:
            label_pred = model(enc(x_batch, training=False), training=True)
            nn_loss = nn_loss_fn(label_batch, label_pred)
            nn_loss += sum(model.losses)
        grads = tape.gradient(nn_loss, model.trainable_variables)
        optimizer_nn.apply_gradients(zip(grads, model.trainable_variables))
        # Update the metric
        for m in nn_metrics:
            m.update_state(label_batch, label_pred)
        return nn_loss

    @tf.function
    def train_step_enc_nn(x_batch, label_batch):

        if args.lr_enc_nn == 0:
            return 0.

        with tf.GradientTape() as tape:
            label_pred = model(enc(x_batch, training=True), training=False)
            enc_nn_loss = nn_loss_fn(label_batch, label_pred)
            enc_nn_loss += sum(enc.losses)
        grads = tape.gradient(enc_nn_loss, enc.trainable_variables)
        optimizer_enc_nn.apply_gradients(zip(grads, enc.trainable_variables))
        # Update the metric
        for m in enc_nn_metrics:
            m.update_state(label_batch, label_pred)
        return enc_nn_loss

    @tf.function
    def train_step(batch):
        """
        patches: tensor data
        """
        x_batch, y_batch, yl_batch, label_batch, lm_batch, ep_batch = batch
        # ------------ Step 1: Train the Reconstruction ---------
        ae_loss = train_step_ae(x_batch, y_batch, lm_batch, ep_batch)
        # ------------ Step 2: Train the MLP or Classifaction
        nn_loss = train_step_nn(x_batch, yl_batch)
        # ------------ Step 3: Train the Encoder according to the Classification
        enc_nn_loss = train_step_enc_nn(x_batch, yl_batch)
        return (ae_loss, nn_loss, enc_nn_loss)

    @tf.function
    def test_step(x_batch, y_batch, label_batch, lm_batch, ep_batch):
        y_pred = dec(enc(x_batch, training=False), training=False)

        ae_loss_val = ae_loss_fn(y_batch, y_pred)
        label_pred = model(enc(x_batch, training=False), training=False)
        nn_loss_val = nn_loss_fn(label_batch, label_pred)

        # Update the metric
        for m in nn_metrics:
            m.update_state(label_batch, label_pred)
        return ae_loss_val, nn_loss_val

    def train(train_data,
              epochs=1000,
              batch_size=30,
              validation_data=None,
              balance=True,
              daugmention=True,
              monitor=None,
              mode='min',
              save_best_only=True,
              verbose=2):

        x_train, y_train, yl_train, label_train, lm_train, ep_train = train_data
        x_val, y_val, yl_val, label_val, lm_val, ep_val = validation_data

        hist = {
            ae_loss_fn.name: [],
            nn_loss_fn.name: [],
            'enc_' + nn_loss_fn.name: [],
        }
        if validation_data is not None:
            hist['val_' + ae_loss_fn.name] = []
            hist['val_' + nn_loss_fn.name] = []

        for m in enc_nn_metrics:
            hist[m.name] = []

        for m in nn_metrics:
            hist[m.name] = []
            if validation_data is not None:
                hist['val_' + m.name] = []

        if balance:
            th_prob = 0.25
            ids1 = yl_train > th_prob
            ids0 = ~ids1
            x0_train = x_train[ids0]
            x1_train = x_train[ids1]
            yl0_train = yl_train[ids0]
            yl1_train = yl_train[ids1]
            lm0_train = lm_train[ids0]
            lm1_train = lm_train[ids1]
            ep0_train = ep_train[ids0]
            ep1_train = ep_train[ids1]

            n_class = [len(yl0_train), len(yl1_train)]
            n_balance = np.min(n_class)
            npatches = 2 * n_balance
        else:
            npatches = len(x_train)
        #  npatches = len(x_train)
        indexes = np.arange(npatches)
        niter_train = int(npatches / batch_size)

        if validation_data is not None:
            npatches_val = len(validation_data[0])
            niter_val = int(npatches_val / batch_size)
        if mode.lower() == 'max':
            monitor_best = -np.inf
        else:
            monitor_best = np.inf

        for epoch in range(epochs):
            if balance:
                if len(yl0_train) > n_balance:
                    ids_balance = np.random.choice(len(yl0_train),
                                                   size=n_balance,
                                                   replace=False)
                    x0 = x0_train[ids_balance]
                    yl0 = yl0_train[ids_balance]
                    lm0 = lm0_train[ids_balance]
                    ep0 = ep0_train[ids_balance]
                    x_train = np.concatenate([x0, x1_train], axis=0)
                    yl_train = np.concatenate([yl0, yl1_train], axis=0)
                    lm_train = np.concatenate([lm0, lm1_train], axis=0)
                    ep_train = np.concatenate([ep0, ep1_train], axis=0)
                else:
                    ids_balance = np.random.choice(len(yl1_train),
                                                   size=n_balance,
                                                   replace=False)
                    x1 = x1_train[ids_balance]
                    yl1 = yl1_train[ids_balance]
                    lm1 = lm1_train[ids_balance]
                    ep1 = ep1_train[ids_balance]
                    x_train = np.concatenate([x0_train, x1], axis=0)
                    yl_train = np.concatenate([yl0_train, yl1], axis=0)
                    lm_train = np.concatenate([lm0_train, lm1], axis=0)
                    ep_train = np.concatenate([ep0_train, ep1], axis=0)
            np.random.shuffle(indexes)
            print('Epoch {}/{}'.format(epoch + 1, epochs), flush=True)
            if validation_data is not None:
                prog_bar = tf.keras.utils.Progbar(niter_train,
                                                  width=15,
                                                  verbose=verbose)
            else:
                prog_bar = tf.keras.utils.Progbar(niter_train - 1,
                                                  width=15,
                                                  verbose=verbose)
            for step in range(niter_train):
                ids = indexes[step * batch_size:(step + 1) * batch_size]
                x_batch = x_train[ids]
                y_batch = y_train[ids]
                lm_batch = lm_train[ids]
                ep_batch = ep_train[ids]
                # Data Augmentation
                if daugmention:
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
                        lm_batch[s2f, :, :] = lm_batch[s2f, ::-1, :]
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
                        lm_batch[s2f, :, :] = lm_batch[s2f, :, ::-1]
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
                        lm_batch[s2f, :, :] = np.transpose(lm_batch[s2f, :, :],
                                                           axes=(0, 2, 1))
                        ep_batch[s2f, :, :] = np.transpose(ep_batch[s2f, :, :],
                                                           axes=(0, 2, 1))

                x_batch = x_batch[..., np.newaxis]
                y_batch = y_batch[..., np.newaxis]
                lm_batch = lm_batch[..., np.newaxis]
                ep_batch = ep_batch[..., np.newaxis]
                yl_batch = yl_train[ids]
                label_batch = label_train[ids]

                train_batch = (x_batch, y_batch, yl_batch, label_batch,
                               lm_batch, ep_batch)

                all_losses = train_step(train_batch)

                ae_loss, nn_loss, enc_nn_loss = all_losses

                # Loss and metrics returns tensors
                ae_loss = ae_loss.numpy()
                nn_loss = nn_loss.numpy()
                enc_nn_loss = enc_nn_loss.numpy()

                # Accuracy
                log_values = [
                    (ae_loss_fn.name, ae_loss),
                    (nn_loss_fn.name, nn_loss),
                    ('enc_' + nn_loss_fn.name, enc_nn_loss),
                ]

                for m in enc_nn_metrics:
                    val = m.result().numpy()
                    log_values.append((m.name, val))
                for m in nn_metrics:
                    val = m.result().numpy()
                    log_values.append((m.name, val))

                prog_bar.update(step, values=log_values)

            # Reset the metrics
            for m in enc_nn_metrics:
                m.reset_states()

            for m in nn_metrics:
                m.reset_states()

            # Loss and metric reconstruction on the validation dataset
            if validation_data is not None:
                ae_loss_val = 0
                nn_loss_val = 0
                for step_val in range(niter_val):
                    ids = np.r_[step_val * batch_size:(step_val + 1) *
                                batch_size]

                    loss_val = test_step(x_val[ids][..., np.newaxis],
                                         y_val[ids][..., np.newaxis],
                                         yl_val[ids], lm_val[ids][...,
                                                                  np.newaxis],
                                         ep_val[ids][..., np.newaxis])

                    ae_loss, nn_loss = loss_val
                    # Loss and metrics returns tensors
                    ae_loss_val += ae_loss.numpy()
                    nn_loss_val += nn_loss.numpy()

                ae_loss_val /= niter_val
                nn_loss_val /= niter_val
                # Progress Bar update
                log_values = [
                    ('val_' + ae_loss_fn.name, ae_loss_val),
                    ('val_' + nn_loss_fn.name, nn_loss_val),
                ]

                for m in nn_metrics:
                    val = m.result().numpy()
                    log_values.append(('val_' + m.name, val))

                prog_bar.update(niter_train, values=log_values)
                # Reset the metrics
                for m in nn_metrics:
                    m.reset_states()

            # Save history
            logged_values = prog_bar._values
            for k in logged_values:
                hist[k].append(logged_values[k][0] / logged_values[k][1])

            # Save model if monitor is not None
            if monitor is not None:
                val = hist[monitor][-1]
                if save_best_only:
                    if mode.lower() == 'max' and val > monitor_best:
                        save_models(hist)
                        monitor_best = val
                        print('Saving model monitor: {}'.format(monitor_best))
                    elif mode.lower() == 'min' and monitor_best > val:
                        save_models(hist)
                        monitor_best = val
                        print('Saving model monitor: {}'.format(monitor_best))

        return hist

    def save_models(hist, e=None):
        if e is None:
            np.save(save_folder + '/' + m_name + '_hist.npy', hist)
            e = ''
        else:
            e = '-{}'.format(e)

        enc.save(save_folder + '/' + m_name + '_enc' + e, save_format='tf')
        dec.save(save_folder + '/' + m_name + '_dec' + e, save_format='tf')
        model.save(save_folder + '/' + m_name + '_nn' + e, save_format='tf')
        model_z.save(save_folder + '/' + m_name + '_nn_z' + e,
                     save_format='tf')

    args.gpus = tf.config.list_physical_devices('GPU')

    m_name = 'dense_ae_hu_gt3_pretrain_pred_EmphProgression_balnew_daug_v10_st-lossV9_2Di_d0_z200_n300_swish_e100_lrAE1E-05_lrENN1E-05_lrNN1E-04_lrZ1E-04_l3_0_cn4'
    nlayers = 3
    latent_dim = 300
    nclass = 2
    neurons = 200

    data_train, data_val = read_data(local_data_path)
    #  data_.. = (x_.., y_.., yl_.., label_.., lm_.., emph_prog_..)
    x_train = data_train[0]
    yl_train = data_train[2]

    input_shape = x_train.shape[1:] + (1, )

    # Count the number of patches with a substancial number of pixels belinging to
    # the class. A full patch with probability 1 will have 32**2, we will consider
    # 32**2/4 as a potential class for progresion. So if the probability is bigger
    # thatn 25% then counts as progression
    th_prob = 0.25
    n_li = (yl_train > th_prob).sum()
    n_total = np.prod(yl_train.shape)
    n_class = [n_total - n_li, n_li]

    print('Unbalanced classes: {}'.format(n_class))
    trained_model = 'ae_pred_2D_st-loss_d0_z300_e20_swish_v10_lrAE5E-05_0_cn4'
    trained_model_folder = trained_model

    print('Pretrain Model: ' + trained_model_folder)

    ae_path = os.path.join(models_path, trained_model_folder,
                           trained_model + '_ae')
    ae = tf.keras.models.load_model(ae_path)
    enc = ae.get_layer('Encoder')
    dec = ae.get_layer('Decoder')

    input_dim = int(enc.output.shape[1])
    model_z, model = nn.local_models.dense_model(
        input_dim,
        latent_dim,
        nclass - 1,  # For two calsses we use one neuron
        neurons,
        nlayers=nlayers,
        activation='swish')

    save_folder = os.path.join(models_path, m_name)
    print('Saving files in: {}'.format(save_folder))
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create the Optimizers and Losses
    optimizer_enc_nn = tf.keras.optimizers.Adam(lr=args.lr_enc_nn)
    optimizer_ae = tf.keras.optimizers.Adam(lr=args.lr_ae)
    optimizer_nn = tf.keras.optimizers.Adam(lr=args.lr_nn)
    optimizer_zreg = tf.keras.optimizers.Adam(lr=args.lr_zreg)
    nn_loss_fn = tf.keras.losses.MeanAbsoluteError(name='nn_mae_loss')
    ae_loss_fn = tf.keras.losses.MeanAbsoluteError(name='ae_mae_loss')
    print('AE Loss: {}'.format(ae_loss_fn.name))

    enc_nn_metrics = []

    nn_metrics = []

    if (len(x_train) < 500) and (args.batch_size >= 500):
        bs = 20
    elif (len(x_train) < 1000) and (args.batch_size >= 1000):
        bs = 50
    else:
        bs = args.batch_size

    if args.batch_size != bs:
        args.batch_size = bs
        print('Changing batch size to {}'.format(bs))

    hist = train(data_train,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 validation_data=data_val,
                 monitor='val_nn_mae_loss',
                 mode='min',
                 save_best_only=True,
                 balance=True,
                 daugmention=True,
                 verbose=args.verbose)

    print('Done !')
