import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.sparse import csc_matrix

import trimesh

import matplotlib as mpl
import matplotlib.cm as cm

import igl
import tqdm
import pyrender


def uniform_matrix(mesh: trimesh.Trimesh):
    D = np.diag(np.array([1. / len(vv) for vv in mesh.vertex_neighbors]))
    return D


def cot_matrix(mesh: trimesh.Trimesh):
    # get face vertex
    face_vert = mesh.vertices[mesh.faces]
    v0, v1, v2 = face_vert[:, 0], face_vert[:, 1], face_vert[:, 2]

    # get square edge length
    A = np.linalg.norm(v1 - v2, axis=1)
    B = np.linalg.norm(v0 - v2, axis=1)
    C = np.linalg.norm(v0 - v1, axis=1)
    A2, B2, C2 = A * A, B * B, C * C
    l2 = np.stack((A2, B2, C2), axis=1)

    # get area
    area = mesh.area_faces
    # compute cot
    cota = np.true_divide(l2[:, 1] + l2[:, 2] - l2[:, 0], area) / 4.0
    cotb = np.true_divide(l2[:, 2] + l2[:, 0] - l2[:, 1], area) / 4.0
    cotc = np.true_divide(l2[:, 0] + l2[:, 1] - l2[:, 2], area) / 4.0
    cot = np.stack((cota, cotb, cotc), axis=1).reshape(-1)
    cot[cot < 1e-7] = 0

    # get L
    ii = mesh.faces[:, [1, 2, 0]]
    jj = mesh.faces[:, [2, 0, 1]]
    idx = np.stack((ii, jj), axis=0).reshape((2, -1))
    L = csc_matrix((cot, idx), shape=(mesh.vertices.shape[0], mesh.vertices.shape[0])).toarray()
    L += L.T
    for i in range(L.shape[0]):  # get diag
        L[i, i] = -np.sum(L[i, mesh.vertex_neighbors[i]])
    return L


def mass_matrix(mesh: trimesh.Trimesh, type='BARYCENTRIC'):
    if type == 'BARYCENTRIC':
        face_area = mesh.area_faces
        barycentric_area_per_face = face_area / 3.
        M = np.zeros(shape=(mesh.vertices.shape[0],))
        Minv = np.zeros(shape=(mesh.vertices.shape[0],))
        for i, v_near_face in enumerate(mesh.vertex_faces):
            M[i] = np.mean(barycentric_area_per_face[v_near_face[v_near_face != -1]])
            Minv[i] = 1. / M[i]
        return np.diag(M), np.diag(Minv)


def mean_curvature(mesh: trimesh.Trimesh, lap: np.ndarray):
    H = 0.5 * np.linalg.norm(lap, axis=1)
    return H


def gaussian_curvature(mesh: trimesh.Trimesh, Minv: np.ndarray):
    K = np.dot(Minv, mesh.vertex_defects)
    return K


def jet(D: np.ndarray, cmap=cm.viridis):
    norm = mpl.colors.Normalize(vmin=D.min(), vmax=D.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(D)
    return colors


def show_curvature_by_uniform_lap(mesh: trimesh.Trimesh, type='mean'):
    D = uniform_matrix(mesh)

    uniform_lap = np.dot(D, mesh.vertices)

    M, Minv = mass_matrix(mesh)
    H = mean_curvature(mesh, uniform_lap)
    K = gaussian_curvature(mesh, Minv)

    # jet
    colors = None
    if type == 'mean':
        colors = jet(H)[:, :3]
    else:
        colors = jet(K)[:, :3]

    mesh.visual.vertex_colors = colors

    mesh.show()
    # scene = pyrender.Scene(ambient_light=0.5*np.array([1.0, 1.0, 1.0, 1.0]))
    # scene = trimesh.Scene()
    # scene.add(pyrender.Mesh.from_trimesh(mesh))
    # v = pyrender.Viewer(scene, use_raymond_lighting=True)
    # scene.show()


def ellipsoid_normal_curvature(t=None):
    def epllipsoid(u, v, a=1, b=1, c=1):
        return np.array([a * np.cos(u) * np.sin(v), b * np.sin(u) * np.sin(v), c * np.cos(v)])

    def grad_u(u, v, a=1, b=1, c=1):
        return np.array([-a * np.sin(u) * np.sin(v), -b * np.cos(u) * np.sin(v), 0])

    def grad_v(u, v, a=1, b=1, c=1):
        return np.array([a * np.cos(u) * np.cos(v), -b * np.sin(u) * np.cos(v), -c * np.sin(v)])

    def jacobi(u, v, a=1, b=1, c=1):
        return np.stack((grad_u(u, v, a, b, c), grad_v(u, v, a, b, c)), axis=1)

    def normal(u, v, a, b, c):
        n = np.cross(grad_u(u, v, a, b, c), grad_v(u, v, a, b, c))
        n = n / (np.linalg.norm(n) + 1e10)
        return n

    def first_form(u, v, a=1, b=1, c=1):
        x_u = grad_u(u, v, a, b, c)
        x_v = grad_v(u, v, a, b, c)
        return np.array([[np.dot(x_u, x_u), np.dot(x_u, x_v)], [np.dot(x_u, x_v), np.dot(x_v, x_v)]])

    def grad_uu(u, v, a=1, b=1, c=1):
        return np.array([-a * np.cos(u) * np.sin(v), b * np.sin(u) * np.sin(v), 0])

    def grad_uv(u, v, a=1, b=1, c=1):
        return np.array([-a * np.cos(u) * np.cos(v), -b * np.cos(u) * np.cos(v), 0])

    def grad_vv(u, v, a=1, b=1, c=1):
        return np.array([-a * np.cos(u) * np.sin(v), b * np.sin(u) * np.sin(v), -c * np.cos(v)])

    def second_form(u, v, a=1, b=1, c=1):
        x_uu = grad_uu(u, v, a, b, c)
        x_uv = grad_uv(u, v, a, b, c)
        x_vv = grad_vv(u, v, a, b, c)
        n = normal(u, v, a, b, c)
        return np.array([[np.dot(x_uu, n), np.dot(x_uv, n)], [np.dot(x_uv, n), np.dot(x_vv, n)]])

    def normal_curvature(u, v, t, a=1, b=1, c=1):
        t_overline = t
        first = first_form(u, v, a, b, c)
        second = second_form(u, v, a, b, c)
        cur = np.dot(np.dot(t_overline, second), t_overline) / (np.dot(np.dot(t_overline, first), t_overline) + 1e10)
        return cur

    # a=1, b=1, c=1
    pt = (1, 0, 0)  # (a, 0, 0)
    u = 0
    v = np.pi / 2.
    if t is None:
        t = np.array([1, 1])
    cur = normal_curvature(u, v, t)
    print(cur)
    return cur


def show_curvature_by_cot_lap(mesh: trimesh.Trimesh, type='mean'):
    C = cot_matrix(mesh)
    M, Minv = mass_matrix(mesh)
    cot_lap_vertices = np.dot(Minv, np.dot(C, mesh.vertices))

    H = mean_curvature(mesh, cot_lap_vertices)
    K = gaussian_curvature(mesh, Minv)

    # jet
    colors = None
    if type == 'mean':
        colors = jet(H)[:, :3]
    else:
        colors = jet(K)[:, :3]

    mesh.visual.vertex_colors = colors

    mesh.show(smooth=False)


def reconstruction(mesh: trimesh.Trimesh, lap_type='cot', k=5):
    C = None
    if lap_type == 'cot':
        C = cot_matrix(mesh)
    else:
        C = uniform_matrix(mesh)

    def top_k_smallest(mat, k):
        eig_value, eig_vector = np.linalg.eig(mat)
        sorted_indices = np.argsort(eig_value)
        return eig_value[sorted_indices[:k]], eig_vector[sorted_indices[:k]]

    eig_value, eig_vector = top_k_smallest(C, k)  # get all eigen vector
    x = np.sum([np.dot(mesh.vertices[:, 0], e_i) * e_i for e_i in eig_vector], axis=0)
    y = np.sum([np.dot(mesh.vertices[:, 1], e_i) * e_i for e_i in eig_vector], axis=0)
    z = np.sum([np.dot(mesh.vertices[:, 2], e_i) * e_i for e_i in eig_vector], axis=0)
    rec_mesh = mesh
    rec_mesh.vertices = np.stack((x, y, z), axis=1)
    rec_mesh.show()
    return rec_mesh


def lap_smooth_explicit(mesh: trimesh.Trimesh, t, lambda_coffe, h):
    C = cot_matrix(mesh)

    t = t  # iter num
    lambda_coffe = lambda_coffe  # lambda
    h = h  # time step
    mesh.show()
    for i in tqdm.trange(t):
        M, Minv = mass_matrix(mesh)
        L = 0.5 * np.dot(Minv, C)
        I = np.identity(mesh.vertices.shape[0])
        mesh.vertices = np.dot(I + h * lambda_coffe * L, mesh.vertices)
    mesh.show()


def lap_smooth_implicit(mesh: trimesh.Trimesh, t, lambda_coffe, h):
    C = cot_matrix(mesh)

    t = t  # iter num
    lambda_coffe = lambda_coffe  # lambda
    h = h  # time step
    mesh.show()
    for i in tqdm.trange(t):
        M, Minv = mass_matrix(mesh)
        M = sp.csc_matrix(M)
        A = sp.csc_matrix(M - h * lambda_coffe * C)
        b = M * mesh.vertices
        mesh.vertices = spl.spsolve(A, b)
    mesh.show()


if __name__ == '__main__':
    def Q1():
        lilium_s = trimesh.load('cw2_meshes/curvatures/lilium_s.obj', process=False)
        show_curvature_by_cot_lap(lilium_s, type='mean')
        show_curvature_by_cot_lap(plane, type='gauss')


    def Q2():
        ellipsoid_normal_curvature()


    def Q3():
        lilium_s = trimesh.load('0006.obj', process=False)
        plane = trimesh.load('0006.obj')
        # Q1 mean
        show_curvature_by_cot_lap(lilium_s, type='mean')
        show_curvature_by_cot_lap(plane, type='mean')


    def Q4():
        armadillo = trimesh.load("C:\\Users\\dqj12\\Desktop\\armadillo.obj")
        rec_mesh = reconstruction(armadillo, k=5)
        rec_mesh = reconstruction(armadillo, k=15)
        rec_mesh = reconstruction(armadillo, k=340)
        rec_mesh = reconstruction(armadillo, k=361)


    def Q5():
        fandisk_ns = trimesh.load('cw2_meshes/smoothing/fandisk_ns.obj')
        plane = trimesh.load('cw2_meshes/smoothing/plane_ns.obj')
        lap_smooth_explicit(fandisk_ns, t=100, lambda_coffe=0.01, h=1e-3)
        lap_smooth_explicit(plane, t=50, lambda_coffe=0.01, h=1e-5)
        lap_smooth_explicit(plane, t=10, lambda_coffe=0.01, h=1)


    def Q6():
        fandisk_ns = trimesh.load('cw2_meshes/smoothing/fandisk_ns.obj')
        plane = trimesh.load('cw2_meshes/smoothing/plane_ns.obj')
        lap_smooth_implicit(fandisk_ns, t=1, lambda_coffe=0.01, h=1)
        lap_smooth_implicit(plane, t=1, lambda_coffe=0.01, h=0.1)
        lap_smooth_implicit(plane, t=10, lambda_coffe=0.01, h=1)


    def Q7():
        bunny = trimesh.load('cw2_meshes/smoothing/bunny.obj')
        bunny: trimesh.Trimesh
        # 0.001 control noisy
        bunny.vertices += 0.001 * np.random.normal(loc=0.0, scale=1.0, size=(bunny.vertices.shape[0], 3))
        # oversmooth  parameter
        # lap_smooth_implicit(bunny, t=1, lambda_coffe=0.01, h=0.1)
        lap_smooth_implicit(bunny, t=10, lambda_coffe=0.01, h=0.00001)


    Q1()
