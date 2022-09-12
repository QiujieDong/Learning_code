import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP


class HumanbodySegDataset(Dataset):
    """Human segmentation dataset from Maron et al (not the remeshed version from subsequent work)"""

    def __init__(self, root_dir, train, k_eig=128, use_cache=True, op_cache_dir=None):

        self.is_train = train  # bool
        self.k_eig = k_eig
        self.root_dir = root_dir
        self.data_path = self.root_dir
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 8

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-face labels!!

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            load_cache = train_cache if self.is_train else test_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list = torch.load(
                    load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        self.mesh_path_list = []
        self.labels_list = []

        subset_data_path = os.path.join(self.data_path, 'train' if self.is_train else 'test')
        labels_path = os.path.join(self.data_path, 'segs')  # per_faces label path

        # compute the num of classes as the weight of loss
        # class_num = torch.zeros(self.args.num_classes)
        for f in sorted(os.listdir(subset_data_path)):
            mesh_path = os.path.join(subset_data_path, f)
            if self.is_train:
                label_path = os.path.join(labels_path, f[:-4] + ".eseg")
                labels = np.loadtxt(label_path).astype(int)
            else:
                label_path = os.path.join(labels_path, 'shrec_' + f[:-4] + "_full.txt")
                labels = np.loadtxt(label_path).astype(int) - 1

            # class_num += torch.as_tensor([np.sum(labels == i) for i in range(self.args.num_classes)])

            self.mesh_path_list.append(mesh_path)
            self.labels_list.append(labels)

        # Load the actual files
        for iFile in range(len(self.mesh_path_list)):
            print("loading mesh " + str(self.mesh_path_list[iFile]))

            verts, faces = pp3d.read_mesh(self.mesh_path_list[iFile])

            if self.is_train:
                label_path = os.path.join(labels_path, f[:-4] + ".eseg")
                labels = np.loadtxt(label_path).astype(int)
            else:
                label_path = os.path.join(labels_path, 'shrec_' + f[:-4] + "_full.txt")
                labels = np.loadtxt(label_path).astype(int) - 1

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            labels = torch.tensor(np.ascontiguousarray(labels))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)

        for ind, labels in enumerate(self.labels_list):
            self.labels_list[ind] = labels

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(
            self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list,
                        self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list),
                       load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[
            idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], \
               self.labels_list[idx]
