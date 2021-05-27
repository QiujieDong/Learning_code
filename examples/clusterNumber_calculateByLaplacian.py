from scipy.sparse.linalg import eigs
import numpy as np
import igl
import matplotlib.pyplot as plt
import trimesh
from sklearn.cluster import KMeans
import open3d as o3d

if __name__ == '__main__':
    # mesh = trimesh.load_mesh('E:\\dataset\\asset\\famous_model\\famous_sparse\\03_meshes\\horse_face20k.ply')
    # mesh = trimesh.load_mesh("E:\\dataset\\sig17_seg_benchmark\\meshes\\train\\faust\\tr_reg_000.ply")
    # mesh = trimesh.load_mesh("E:\\dataset\\sig17_seg_benchmark\\meshes\\train\\scape\\mesh044.off")
    mesh = trimesh.load_mesh("E:\\dataset\\sig17_seg_benchmark\\meshes\\train\\MIT_animation\\meshes_jumping\\meshes\\mesh_0000.obj")
    mesh: trimesh.Trimesh
    cot = igl.cotmatrix(mesh.vertices, mesh.faces).toarray()
    cot = -cot
    value, vector = np.linalg.eigh(cot)

    value_ind = np.argsort(value)
    print(value_ind)
    print(vector[2])
 

    # 可视化特征值变化信息
    # value = np.sort(value)[::-1]
    # value = np.sort(value)
    # value_norm = (value) / (value.max())
    # plt.plot(value_norm[:10])
    # plt.plot(value[:10])
    # plt.show()


    # 可视化mesh上的顶点的分类
    # model = KMeans(n_clusters=6, random_state=0)
    # model.fit(vector)
    # labels = model.labels_
    # # plt.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], c=labels)
    # # plt.show()
    
    # max_label = labels.max()
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    
    # pts = o3d.geometry.PointCloud()
    # pts.points = o3d.utility.Vector3dVector(mesh.vertices)
    # pts.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # o3d.visualization.draw_geometries([pts])

