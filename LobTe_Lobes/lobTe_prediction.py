"""
File: lobTe_prediction.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import os
import torch
import numpy as np

from nn.simple_lobte import SimpleLobTe


def load_LobTe(device=0):
    models_path = "models/"

    if isinstance(device, int):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        device = torch.device("cuda")

    m_name = "LobTe_nh8_nl1_dff32_df5_dpr1_dm32_e1000_lrNone_drop2.5E-01"
    mm_name = (
        "LobTe_Change_Adj_Density_Lobes_P1_P2_"
        + "AER-TensorFlow_Dens_MultiLobe"
    )
    mfile = os.path.join(models_path, m_name, mm_name, mm_name + ".pt")

    image_size = (300, 11)
    patch_size = (300, 11)
    outcomes = {}
    dim = 32
    depth = 1
    heads = 8
    mlp_dim = 32
    nlobes = 5

    model = SimpleLobTe(
        image_size,
        patch_size,
        outcomes,
        dim,
        depth,
        heads,
        mlp_dim,
        deepf_dim=5,
        in_channels=1,
        lobes=nlobes,
        pool="mean",
        dropout_rate=0.25,
    ).to(device)

    print("Loading weights %s" % mfile)

    model.load_state_dict(torch.load(mfile, weights_only=True))
    model.eval()
    return model, m_name, mm_name


if __name__ == "__main__":

    ndevice = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ndevice)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    fplobes = None  # shape: (5, 300, 11)

    # Fingerprint normalization for our model is:
    norm = {"mean": -0.0344326, "std": 4.6108916}

    x_data = (fplobes - norm["mean"]) / norm["std"]
    x_data = x_data[np.newaxis, np.newaxis, ...]

    # Load models
    lobte, m_name, mm_name = load_LobTe(device=ndevice)
    device = next(lobte.parameters()).device

    x = torch.tensor(x_data).to(device)
    y_pred = lobte(x.float())

    # Outcomes where normalized
    onorm = {"mean": -0.083200775, "std": 15.708397}
    dpred = {}

    for k in y_pred:
        if k == "deepf":
            deepf_pred = y_pred[k].numpy(force=True)
            for i in range(deepf_pred.shape[1]):
                dpred["DF_%i" % i] = deepf_pred[:, i]
        else:
            dpred[k] = (
                y_pred[k].numpy(force=True) * onorm["std"] + onorm["mean"]
            )
