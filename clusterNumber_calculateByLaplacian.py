from scipy.sparse.linalg import eigs
import numpy as np
import igl
import matplotlib.pyplot as plt
import trimesh
from sklearn.cluster import KMeans

if __name__ == '__main__':
    # mesh = trimesh.load_mesh('E:\\dataset\\asset\\famous_model\\famous_sparse\\03_meshes\\horse_face20k.ply')
    # mesh = trimesh.load_mesh("E:\\dataset\\sig17_seg_benchmark\\meshes\\train\\faust\\tr_reg_000.ply")
    mesh = trimesh.load_mesh("E:\\dataset\\sig17_seg_benchmark\\meshes\\train\\scape\\mesh044.off")
    mesh: trimesh.Trimesh
    cot = igl.cotmatrix(mesh.vertices, mesh.faces).toarray()
    cot = -cot
    value, vector = np.linalg.eigh(cot)
    # value = np.sort(value)[::-1]
    value = np.sort(value)
    # value_norm = (value) / (value.max())
    # plt.plot(value_norm[:10])
    plt.plot(value[:10])
    plt.show()

    # model = KMeans(n_clusters=6, random_state=0)
    # model.fit(mesh.vertices)
    # labels = model.labels_
    # plt.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], c=labels)
    # plt.show()
