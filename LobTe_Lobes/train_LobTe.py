"""
File: train_LobTe.py
Author: Ariel Hernán Curiale
Github: https://gitlab.com/Curiale
Description: Script to train LobTe using simple ViT
"""

import numpy as np
import torch
import torch.nn.utils
import torch.nn.functional as F

from optim.lr_scheduler import TransformerScheduler

from utils.progbar import Progbar

from nn.simple_lobte import SimpleLobTe
import data.loader
import nn.losses
import nn.metrics
import nn.callbacks


def data_to_device(x, device):
    """
    Send the data to the device, if the data is a numpy or a dict it is first
    converted to torch tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x).to(device)
    elif isinstance(x, list):
        x = torch.tensor(x).to(device)
    elif isinstance(x, dict):
        x = {k: torch.tensor(x[k]).to(device) for k in x}
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    return x


def train_step(
    model,
    device,
    x_batch,
    y_batch,
    optimizer,
    loss,
    metrics,
    scheduler,
    scaler,
):

    total_loss = 0
    loss_val = {k: 0 for k in loss}
    x = data_to_device(x_batch, device)
    y = data_to_device(y_batch, device)
    optimizer.zero_grad()

    if scaler is None:
        # Using float32 (no autocast and Scaler needed)
        # ------------------------------------
        y_pred = model(x)
        for k in y:
            loss_val[k] = loss[k](y_pred[k].squeeze(), y[k].squeeze())
            total_loss += loss_val[k]

        total_loss.backward()
        # Clip the gradient just to avoid gradient explotion. It usually helps
        # to train the model
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    else:
        # Mixed precision float16 and float32
        # ------------------------------------
        # Runs the forward pass with autocasting.
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            y_pred = model(x)
            for k in y:
                loss_val[k] = loss[k](y_pred[k].squeeze(), y[k].squeeze())
                total_loss += loss_val[k]

        # Scales loss.  Calls backward() on scaled loss to create scaled
        # gradients
        scaler.scale(total_loss).backward()

        # Clip the gradient just to avoid gradient explotion. It usually helps
        # to train the model
        scaler.unscale_(optimizer)
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # scaler.step() first unscales the gradients of the optimizer's
        # assigned params.  If these gradients do not contain infs or NaNs,
        # a optimizer.step() is then called, otherwise, optimizer.step() is
        # skipped.
        scaler.step(optimizer)

        scale = scaler.get_scale()  # Check if the optimizer.step() was skipped
        # Updates the scale for next iteration.
        scaler.update()
        skip_lr_sched = scale > scaler.get_scale()
        if scheduler is not None and not skip_lr_sched:
            scheduler.step()

    # Update the metric
    for k in metrics:
        metrics[k].update_state(y_pred[k].squeeze(), y[k].squeeze())

    # NOTE: Use item to return a CPU value avoiding OOM. Returning a tensor
    # will keep the tensor in GPU alive and will end in OOM
    loss_val = {k: loss_val[k].item() for k in loss_val}
    return loss_val


def train(
    info_outcomes,
    model,
    device,
    train_data,
    val_data,
    optimizer,
    loss,
    metrics={},
    batch_size=32,
    epochs=100,
    scheduler=None,
    mixed_precision=True,
    verbose=1,
):

    x, y = train_data

    # Create the data sampler.
    train_sampler = data.loader.RandomBatchSampler(
        x,
        y,
        batch_size,
        drop_last=False,
        boostrap=True,
    )

    hist = {}
    if len(y) > 1:
        for k in y:
            hist["loss_%s" % info_outcomes[k]["label"]] = []
            hist["val_loss_%s" % info_outcomes[k]["label"]] = []

    hist["loss"] = []
    hist["val_loss"] = []

    for k in metrics:
        hist["%s_%s" % (metrics[k].name, info_outcomes[k]["label"])] = []
        hist["val_%s_%s" % (metrics[k].name, info_outcomes[k]["label"])] = []

    first_k = list(y.keys())[0]

    early_stopping = nn.callbacks.EarlyStopping(
        model, patience=int(epochs / 2), mode="min", restore_best_weights=True
    )
    early_stopping.on_train_start()

    if mixed_precision:
        # Mixed precision float16 and float32
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.amp.GradScaler("cuda")
    else:
        # Using float32 (no autocast and Scaler needed)
        # ------------------------------------
        scaler = None

    for epoch in range(1, epochs + 1):
        print("Epoch {}/{}".format(epoch, epochs), flush=True)
        prog_bar = Progbar(
            train_sampler.n_batches + 1, width=15, verbose=verbose
        )
        train_sampler.reset()
        # 1) Train loop
        model.train()
        for step, (idx, x_batch, y_batch) in enumerate(train_sampler):
            loss_val = train_step(
                model,
                device,
                x_batch,
                y_batch,
                optimizer,
                loss,
                metrics,
                scheduler,
                scaler,
            )

            if len(y) > 1:
                total_loss = 0
                log_values = []
                for k in y:
                    total_loss += loss_val[k]
                    loss_label = "loss_%s" % info_outcomes[k]["label"]
                    log_values.append((loss_label, loss_val[k]))
                log_values = [("loss", total_loss)] + log_values
            else:
                log_values = [("loss", loss_val[first_k])]

            # Metrics: If we update in the progbar we need to return the last
            # value of the metric because the progbar is going to take the mean
            # value
            for k in metrics:
                mlabel = "%s_%s" % (metrics[k].name, info_outcomes[k]["label"])
                log_values.append((mlabel, metrics[k].val))

            prog_bar.update(step + 1, values=log_values)

        # Reset the metrics
        for k in metrics:
            metrics[k].reset_states()

        # 2) Validation loop
        val_loss = test(model, device, val_data, batch_size, loss, metrics)

        if len(y) > 1:
            total_loss = 0
            log_values = []
            for k in y:
                total_loss += val_loss[k]
                loss_label = "val_loss_%s" % info_outcomes[k]["label"]
                log_values.append((loss_label, val_loss[k]))
            log_values = [("val_loss", total_loss)] + log_values
        else:
            log_values = [("val_loss", val_loss[first_k])]

        for k in metrics:
            mlabel = "%s_%s" % (metrics[k].name, info_outcomes[k]["label"])
            log_values.append(("val_" + mlabel, metrics[k].result()))
        # Progress Bar update
        prog_bar.update(train_sampler.n_batches + 1, values=log_values)

        # Reset the metrics
        for k in metrics:
            metrics[k].reset_states()

        # Save history
        logged_values = prog_bar._values
        for k in logged_values:
            hist[k].append(logged_values[k][0] / logged_values[k][1])

        # Check if we should stop
        lv = np.sum([val_loss[k] for k in val_loss])

        early_stopping.on_epoch_end(epoch - 1, lv)
        if early_stopping.stop_training:
            break

    early_stopping.on_train_end()
    hist["EarlyStopping"] = {
        "val_loss": early_stopping.best_loss,
        "best_epoch": early_stopping.best_epoch,
    }

    return hist


def test(model, device, val_data, batch_size, loss, metrics):

    x_val, y_val = val_data
    val_sampler = data.loader.RandomBatchSampler(
        x_val, y_val, batch_size, drop_last=False
    )
    val_loss = {k: 0 for k in y_val}
    model.eval()
    # Disable Autograd (context no_grad) for a faster validation and avoid OOM
    with torch.no_grad():
        for step, (idx, x_batch, y_batch) in enumerate(val_sampler):
            x = data_to_device(x_batch, device)
            y = data_to_device(y_batch, device)
            y_pred = model(x.float())
            # NOTE: Use item to detach the tensor and avoid GPU OOM
            for k in y_val:
                val_loss[k] += loss[k](
                    y_pred[k].squeeze(), y[k].squeeze()
                ).item()
                # Update the metric
                metrics[k].update_state(y_pred[k].squeeze(), y[k].squeeze())

        for k in y_val:
            val_loss[k] /= val_sampler.n_batches
    return val_loss


def main():
    import os
    from pathlib import Path

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    models_path = "models/"
    epochs = 1000
    batch_size = 1024

    # Control reproducibility
    torch.manual_seed(10)
    np.random.seed(10)
    import random

    random.seed(0)

    if torch.cuda.is_available():
        print("CUDA device found !")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS device found !")
        device = torch.device("mps")
    else:
        print("Running on CPU !")
        device = torch.device("cpu")

    # Load the train and validation data
    # Train/val: [ndarray, dict(array)]
    train_data, val_data = None, None  # <create and define the data loader>
    # Lobe fingerprints should be normalized, our model was trained using
    # norm = {"mean": -0.0344326, "std": 4.6108916}

    m_name = "LobTe_nh8_nl1_dff32_df5_dpr1_dm32_e1000_lrNone_drop2.5E-01"
    dim = 32
    depth = 1
    heads = 8
    mlp_dim = 32

    mm_name = (
        "LobTe_Change_Adj_Density_Lobes_P1_P2_"
        + "AER-TensorFlow_Dens_MultiLobe"
    )

    save_folder = os.path.join(models_path, m_name, mm_name)
    print("Saving files in: %s" % save_folder, flush=True)
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # Call the GC to release memory
    # gc.collect()

    in_channels = train_data[0].shape[1]
    nlobes = train_data[0].shape[2]
    image_size = train_data[0].shape[3:]

    # outcomes = {k: train_data[1][k].ndim for k in train_data[1]}
    outcomes = {
        "Change_Adj_Density_P1P2_RightSuperiorLobe": 1,
        "Change_Adj_Density_P1P2_RightMiddleLobe": 1,
        "Change_Adj_Density_P1P2_RightInferiorLobe": 1,
        "Change_Adj_Density_P1P2_LeftSuperiorLobe": 1,
        "Change_Adj_Density_P1P2_LeftInferiorLobe": 1,
    }
    info_outcomes = {
        "Change_Adj_Density_P1P2_RightSuperiorLobe": {"label": "DALD RUL"},
        "Change_Adj_Density_P1P2_RightMiddleLobe": {"label": "DALD RML"},
        "Change_Adj_Density_P1P2_RightInferiorLobe": {"label": "DALD RLL"},
        "Change_Adj_Density_P1P2_LeftSuperiorLobe": {"label": "DALD LUL"},
        "Change_Adj_Density_P1P2_LeftInferiorLobe": {"label": "DALD LLL"},
    }

    patch_size = image_size

    model = SimpleLobTe(
        image_size,
        patch_size,
        outcomes,
        dim,
        depth,
        heads,
        mlp_dim,
        deepf_dim=5,
        in_channels=in_channels,
        lobes=nlobes,
        pool="mean",
        dropout_rate=0.25,
    ).to(device)

    metrics = {k: nn.metrics.Metric("mae", F.l1_loss) for k in outcomes}

    bias_penalty = 0.2
    loss = {
        k: nn.losses.Loss("huber", F.huber_loss, bias_penalty=bias_penalty)
        for k in outcomes
    }
    nsteps = (epochs // 5) * len(train_data[0]) // batch_size
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-09,
        weight_decay=1e-5,
        amsgrad=True,
    )

    lr_mul = 1.0
    d_model = dim
    scheduler = TransformerScheduler(optimizer, lr_mul, d_model, nsteps)

    hist = train(
        info_outcomes,
        model,
        device,
        train_data,
        val_data,
        optimizer,
        loss,
        metrics=metrics,
        epochs=epochs,
        scheduler=scheduler,
        batch_size=batch_size,
        verbose=2,
    )

    torch.save(model.state_dict(), os.path.join(save_folder, mm_name + ".pt"))
    np.save(os.path.join(save_folder, mm_name + "_hist.npy"), hist)

    # Print the loss and metrics of the model saved
    ide = hist["EarlyStopping"]["best_epoch"]
    msg = "Epoch %i: " % ide
    for k in hist:
        if k != "EarlyStopping":
            msg += "%s: %.4f - " % (k, hist[k][ide])
    print("Restoring best model:")
    print(msg)

    print("Saving files in: %s" % save_folder, flush=True)
    print("Done !", flush=True)


if __name__ == "__main__":
    main()
