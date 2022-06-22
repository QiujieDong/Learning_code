import os

import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt

def visualization_WKS():
    data_path = 'data/0001.obj'
    mesh = trimesh.load_mesh(data_path, process=False)
    mesh: trimesh.Trimesh
    #
    eig_vector = torch.from_numpy(np.load(os.path.join('data', '0001_eigen.npy'))).float()
    eig_value = torch.from_numpy(np.load(os.path.join('data', '0001_eigenValues.npy'))).float()

    N = 100
    wks_variance = N * 0.05

    log_E = np.log(torch.maximum(eig_value, torch.tensor(1e-6))).T
    e = torch.from_numpy(np.linspace(log_E[1], torch.max(log_E) / 1.02, num=N))
    sigma = torch.tensor((e[1] - e[0]) * wks_variance)

    WKS = []
    WKS_norm = []
    C_all = []

    for i in range(N):
        wks = torch.sum(torch.square(eig_vector) * torch.exp(- np.square(e[i] - log_E) / (2 * torch.square(sigma))),
                        dim=1)
        C = torch.sum(torch.exp(- torch.square(e[i] - log_E) / (2 * torch.square(sigma))))

        wks_norm = C * wks

        WKS.append(wks)
        C_all.append(C)
        WKS_norm.append(wks_norm)

    k = 0
    color_wks_norm = (WKS_norm[k] - WKS_norm[k].min()) / (WKS_norm[k].max() - WKS_norm[k].min())

    colors = plt.get_cmap("viridis")(color_wks_norm)
    mesh.visual.vertex_colors = colors[:, :3]
    mesh.show()


def visualization_HKS():
    data_path = 'data/0001.obj'
    mesh = trimesh.load_mesh(data_path, process=False)
    mesh: trimesh.Trimesh

    eig_vector = torch.from_numpy(np.load(os.path.join('data', '0001_eigen.npy'))).float()
    eig_value = torch.from_numpy(np.load(os.path.join('data', '0001_eigenValues.npy'))).float()

    # get time interval
    t_min = 4 * np.log(10) / eig_value.max()
    t_max = 4 * np.log(10) / np.sort(eig_value)[1]

    ts = np.linspace(t_min, t_max, num=100)
    hkss = (eig_vector[:, :, None] ** 2) * np.exp(-eig_value[None, :, None] * ts.flatten()[None, None, :])

    hks = torch.sum(hkss, dim=1)

    k = 1
    colors = plt.get_cmap("viridis")(((hks[:, k] - hks[:, k].min()) / (hks[:, k].max() - hks[:, k].min())))

    mesh.visual.vertex_colors = colors[:, :3]
    mesh.show()
