# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# MIT License

# Copyright (c) 2019 Rana Hanocka

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from heapq import heappop, heapify
from threading import Thread
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "MeshCNNClassifier",
    "compute_face_normals_and_areas",
    "extract_meshcnn_features",
]


def compute_face_normals_for_mesh(mesh):
    r"""Compute face normals for an input kaolin.rep.TriangleMesh object.
    计算输入的三角mesh的face单位法向

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.

    Returns:
        face_normals (torch.Tensor): Tensor containing face normals for
            each triangle in the mesh (shape: :math:`(M, 3)`), where :math:`M`
            is the number of faces (triangles) in the mesh.

    function:
        torch.cross: 计算叉积
        norm: 计算范数，p=2为二范数，指空间上两个向量的直线距离，用来计算向量长度；dim=-1在最后一个维度上操作，这里就是在行上计算范数
        [..., None]：...表示相应维度的所有数据，None表示在列维度上所有数据再增加一个维度，所有数据在这一维（列维度）中
    """
    face_normals = torch.cross(
        mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
        mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]],
    )
    face_normals = face_normals / face_normals.norm(p=2, dim=-1)[..., None]  # 计算单位法向
    return face_normals


def compute_face_normals_and_areas(mesh):
    r"""Compute face normals and areas for an input kaolin.rep.TriangleMesh object.
    计算三角face的单位法向与面积

    Args:
       mesh (kaolin.rep.TriangleMesh): A triangle mesh object.

    Returns:
        face_normals (torch.Tensor): Tensor containing face normals for
            each triangle in the mesh (shape: :math:`(M, 3)`), where :math:`M`
            is the number of faces (triangles) in the mesh.
        face_areas (torch.Tensor): Tensor containing areas for each triangle
            in the mesh (shape: :math:`(M, 1)`), where :math:`M` is the number
            of faces (triangles) in the mesh.
    """
    face_normals = torch.cross(
        mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
        mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]],
    )
    face_normal_lengths = face_normals.norm(p=2, dim=-1)
    face_normals = face_normals / face_normal_lengths[..., None]  # 单位法向
    # Recall: area of a triangle defined by vectors a and b is 0.5 * norm(cross(a, b))
    face_areas = 0.5 * face_normal_lengths  # 使用向量叉积求三角形面积求法
    return face_normals, face_areas


def is_two_manifold(mesh):
    """Returns whether the current mesh is 2-manifold. Assumes that adjacency info
    for the mesh is enabled.
    在有mesh的邻接信息的前提下，判断当前mesh是否是二流形

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object (assumes adjacency
            info is enabled).

    ef: Edge-Face neighbourhood tensor
    流形表面：一条边只能对应两个面
    """
    return (mesh.ef.shape[-1] == 2) and (mesh.ef.min() >= 0)


def build_gemm_representation(mesh, face_areas):
    r"""Build a GeMM-suitable representation for the current mesh.
    构建edge的相邻边索引矩阵，以及定义edge面积并计算此面积
    general matrix multiplication (GEMM,广义矩阵乘法): by expanding (or unwrapping) the image into a column matrix.

    The GeMM representation contains the following attributes:
        gemm_edges: tensor of four 1-ring neighbours per edge (E, 4)
        sides: tensor of indices (in the range [0, 3]) indicating the index of an edge
            in the gemm_edges entry of the 4 neighbouring edges.
        Eg. edge i => gemm_edges[gemm_edges[i], sides[i]] = [i, i, i, i]
        gemm_edges[i]为edge i的1-ring的四条边，sides[i]为edge i在其1-ring的四条边的1-ring的索引

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh that is 2-manifold.
        face_areas (torch.Tensor): Areas of each triangle in the mesh
            (shape: :math:`(F)`, where :math:`F` is the number of faces).

    """
    # Retain first four neighbours for each edge (Needed esp if using newer
    # adjacency computation code).
    mesh.gemm_edges = mesh.ee[..., :4]  # ee为相邻edge矩阵。取前四个，也就是对于流形表面一个edge有四个1-ring邻居
    # Compute the "sides" tensor
    mesh.sides = torch.zeros_like(mesh.gemm_edges)  # 初始化sides矩阵与gemm-edges矩阵是same size的全零矩阵

    # TODO: Vectorize this!
    for i in range(mesh.gemm_edges.shape[-2]):  # i in range[0, the_number_of_edge]
        for j in range(mesh.gemm_edges.shape[-1]):  # 正常情况下 j in range[0,3]
            nbr = mesh.gemm_edges[i, j]
            # torch.nonzero(): returns a 2-D tensor where each row is the index for a nonzero value.
            ind = torch.nonzero(mesh.gemm_edges[nbr] == i)
            mesh.sides[i, j] = ind  # edge i为在相邻的第edge j中是第ind (range[0,3] )条边，以此构成sides矩阵

    # Average area of all faces neighbouring an edge (normalized by the overall area of
    # the mesh). Weirdly, MeshCNN, computes averages by dividing by 3 (as opposed to
    # dividing by 2), and hence, we adopt their convention.
    mesh.edge_areas = face_areas[mesh.ef].sum(dim=-1) / (3 * face_areas.sum())
    # 这里定义edge_areas：edge相邻face面积之和的平均值除以mesh的所有face面积之和。这里除以的是3，而不是2，是因为在MeshCNN就是这样，所以遵照原文章。


def get_edge_points_vectorized(mesh):
    r"""Get the edge points (a, b, c, d, e) as defined in Fig. 4 of the MeshCNN
    paper: https://arxiv.org/pdf/1809.05910.pdf.
    edge e的四条edge(a,b,c,d),逆时针顺序可以是(a,b,c,d)或(c,d,a,b),所以为保持卷积不变性，需要进行处理

    下面这个set_edge_lengths函数的输入参数edge_points就是由此函数输出

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.

    Returns:
        (torch.Tensor): Tensor containing "edge points" of the mesh, as per
            MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E` is the
            total number of edges in the mesh.
    """

    a = mesh.edges  # 所有边的信息，边是由两顶点（vertices）组成
    b = mesh.edges[mesh.gemm_edges[:, 0]]  # 与a相邻的第一个edge,输出为两个顶点坐标（因为两个顶点确定一条edge）
    c = mesh.edges[mesh.gemm_edges[:, 1]]
    d = mesh.edges[mesh.gemm_edges[:, 2]]
    e = mesh.edges[mesh.gemm_edges[:, 3]]

    v1 = torch.zeros(a.shape[0]).bool().to(a.device)  # a.shape[0]为edge数目，zeros生成零向量，然后转为bool形式，也就是全为False的向量。
    v2 = torch.zeros_like(v1)
    v3 = torch.zeros_like(v1)

    """
    这个地方比较特殊，所以写个例子记录一下。
    In: x = torch.tensor([1,2,3])
        y = torch.tensor([3,4,3])
        z = torch.tensor([6,7,8])
        print((x[2] == y[0]) +  (x[2] == y[0])) # Out: tensor(True)
        print((x[2] == y[0]) +  (x[2] == y[1])) # Out: tensor(True)
        print((x[2] == y[1]) +  (x[2] == y[1])) # Out: tensor(False)
        print(((x[2] == y[0]) +  (x[2] == y[0])).long()) # Out: tensor(1)
        print(((x[2] == y[0]) +  (x[2] == y[1])).long()) # Out: tensor(1)
        print(((x[2] == y[0]) +  (x[2] == y[1])).long()) # Out: tensor(0)
        torch.stack((x,y,z), dim=0) # Out: tensor([[1, 2, 3],[3, 4, 3],[6, 7, 8]])
        torch.stack((x,y,z), dim=1) # Out: tensor([[1, 3, 6],[2, 4, 7],[3, 3, 8]])
    """
    # 一条edge是由两个顶点组成的，那么a[:, 1]就是所有edge的第一个顶点的索引
    # a_in_b：若a与b共同点是a中第1列的点，则a_in_b=tensor(True)，若共同点是第0列的点，那么a_in_b=tensor(False)
    # 而tensor(True).long()=tensor(1),正好对应于第一列。同理tensor(False).long()=tensor(0),对应第0列。
    a_in_b = (a[:, 1] == b[:, 0]) + (a[:, 1] == b[:, 1])  # tensor(True) or tensor(False)。
    not_a_in_b = ~a_in_b  # ~按位取反，这里是True或者False。如果是数字，例如5,按位取反为 -(5+1)=-6。
    a_in_b = a_in_b.long()  # 输出为tensor(1)或者tensor(0)
    not_a_in_b = not_a_in_b.long()
    b_in_c = ((b[:, 1] == c[:, 0]) + (b[:, 1] == c[:, 1])).long()
    d_in_e = ((d[:, 1] == e[:, 0]) + (d[:, 1] == e[:, 1])).long()

    arange = torch.arange(mesh.edges.shape[0]).to(a.device)

    # torch.stack:沿维度dim连接一系列向量,每一行（dim=0）或者每一列(dim=1)组成一个向量
    # stack后，每一个向量为edge_points的顺序就定了：[a与b共同点，a与b不共同点，b与c共同点，d与e共同点]，全部为点的索引。
    return torch.stack(
        (
            a[arange, a_in_b],
            a[arange, not_a_in_b],
            b[arange, b_in_c],
            d[arange, d_in_e],
        ),
        dim=-1,
    )


def set_edge_lengths(mesh, edge_points):
    r"""Set edge lengths for each of the edge points. 
    计算edge的长度
    Args:
       mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
       edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
            edge_points为上面这个函数get_edge_points_vectorized输出
    """
    # 用两向量相减求edge_lengths，edge_points[:, 0]为上面get_edge_points_vectorized中a与b的共同点，edge_points[:, 1]为a与b的不共同点
    # 也就是说edge_points[:, 0]与edge_points[:, 1]是edge a的两个顶点。
    mesh.edge_lengths = (
            mesh.vertices[edge_points[:, 0]] - mesh.vertices[edge_points[:, 1]]
    ).norm(p=2, dim=1)


def compute_normals_from_gemm(mesh, edge_points, side, eps=1e-1):
    r"""Compute vertex normals from the GeMM representation.
    计算edge相邻面的法向
    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
        side (int): Side of the edge used in computing normals (0, 1, 2, or 3
            following MeshCNN convention).
        eps (float): A small number, for numerical stability (default: 1e-1,
            following MeshCNN implementation).

    Returns:
        (torch.Tensor): Face normals for each vertex on the chosen side of the
            edge (shape: :math:`(E, 2)`, where :math:`E` is the number of edges
            in the mesh).
    """
    # 下面计算二面角的函数compute_dihedral_angles时，side分别为0与3，这样a与b就分别为两个三角形的两条边。
    a = (
            mesh.vertices[edge_points[:, side // 2 + 2]]  # //为整除运算符，取商的整数部分
            - mesh.vertices[edge_points[:, side // 2]]
    )
    b = (
            mesh.vertices[edge_points[:, 1 - side // 2]]
            - mesh.vertices[edge_points[:, side // 2 + 2]]
    )
    normals = torch.cross(a, b)
    return normals / (normals.norm(p=2, dim=-1)[:, None] + eps)  # 单位法向


def compute_dihedral_angles(mesh, edge_points):
    r"""Compute dihedral angle features for each edge. 
    计算二面角
    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.

    Returns:
        (torch.Tensor): Dihedral angle features for each edge in the mesh
            (shape: :math:`(E, 4)`, where :math:`E` is the number of edges
            in the mesh).

    torch.clamp(input, min, max): 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量。
    """
    a = compute_normals_from_gemm(mesh, edge_points, 0)
    b = compute_normals_from_gemm(mesh, edge_points, 3)
    dot = (a * b).sum(dim=-1).clamp(-1, 1)  # 这里(a * b).sum(dim=-1)=x_1x_2+y_1y_2+z_1z_2,因为角度在0-180之间，那么对应余弦值为[-1,1]
    # 如果两个法向量一个指向二面角内部另一个指向二面角外部，则二面角的大小就是θ。如果两个法向量同时指向二面角内部或外部，则二面角的大小为π-θ。
    return math.pi - torch.acos(dot)


def compute_opposite_angles(mesh, edge_points, side, eps=1e-1):
    r"""Compute opposite angle features for each edge.
    计算文章中的inner angles（也就是两个内角）
    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
        side (int): Side of the edge used in computing normals (0, 1, 2, or 3
            following MeshCNN convention).
        eps (float): A small number, for numerical stability (default: 1e-1,
            following MeshCNN implementation).

    Returns:
        (torch.Tensor): Opposite angle features on the chosen side of the
            edge (shape: :math:`(E, 2)`, where :math:`E` is the number of edges
            in the mesh).
    """
    a = (
            mesh.vertices[edge_points[:, side // 2]]  # a与b为例如原文图四中同一边的两条边
            - mesh.vertices[edge_points[:, side // 2 + 2]]
    )
    b = (
            mesh.vertices[edge_points[:, 1 - side // 2]]
            - mesh.vertices[edge_points[:, side // 2 + 2]]
    )
    a = a / (a.norm(p=2, dim=-1)[:, None] + eps)
    b = b / (b.norm(p=2, dim=-1)[:, None] + eps)
    dot = (a * b).sum(dim=-1).clamp(-1, 1)  # 向量内积：a.b=|a||b|cosθ
    return torch.acos(dot)  # 计算得出两向量夹角


def compute_symmetric_opposite_angles(mesh, edge_points):
    r"""Compute symmetric opposite angle features for each edge.
    计算得到两个inner angles，并stack起来
    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.

    Returns:
        (torch.Tensor): Symmetric opposite angle features for each edge in the mesh
            (shape: :math:`(E, 4)`, where :math:`E` is the number of edges
            in the mesh).
    """
    a = compute_opposite_angles(mesh, edge_points, 0)
    b = compute_opposite_angles(mesh, edge_points, 3)
    angles = torch.stack((a, b), dim=0)  # stack将a,b放到了一个维度上（这里是dim=0上，a与b两个向量成为同一行），angles[0]=a,angles[1]=b
    val, _ = torch.sort(angles, dim=0)
    return val  # angles是有两个内角角度，这个sort后，val进行了排序，对于两个内角，小的内角在val[0]中，大的内角在val[1]中


def compute_edgelength_ratios(mesh, edge_points, side, eps=1e-1):
    r"""Compute edge-length ratio features for each edge.
    edge-length的定义是：edge的长度与其相邻面到edge的垂线的比率
    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
        side (int): Side of the edge used in computing normals (0, 1, 2, or 3
            following MeshCNN convention).
        eps (float): A small number, for numerical stability (default: 1e-1,
            following MeshCNN implementation).

    Returns:
        (torch.Tensor): Edge-length ratio features on the chosen side of the
            edge (shape: :math:`(E, 2)`, where :math:`E` is the number of edges
            in the mesh).
    """
    # 上面函数有计算edge长度的函数set_edge_lengths
    edge_lengths = (
            mesh.vertices[edge_points[:, side // 2]]
            - mesh.vertices[edge_points[:, 1 - side // 2]]
    ).norm(p=2, dim=-1)
    o = mesh.vertices[edge_points[:, side // 2 + 2]]
    a = mesh.vertices[edge_points[:, side // 2]]
    b = mesh.vertices[edge_points[:, 1 - side // 2]]
    ab = b - a
    projection_length = (ab * (o - a)).sum(dim=-1) / (ab.norm(p=2, dim=-1) + eps)  # 这个是edge一条临边在edge上的投影
    closest_point = a + (projection_length / edge_lengths)[:, None] * ab  # 通过比例计算得到在edge上的垂足
    d = (o - closest_point).norm(p=2, dim=-1)  # 垂足与顶点连线就是edge相邻面顶点与edge之间垂线段
    return d / edge_lengths


def compute_symmetric_edgelength_ratios(mesh, edge_points):
    r"""Compute symmetric edge-length ratio features for each edge.
    这个与上面计算两个内角的函数性质一样。得到两个edge-length ration
    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.

    Returns:
        (torch.Tensor): Symmetric edge-length ratio features for each edge in the mesh
            (shape: :math:`(E, 4)`, where :math:`E` is the number of edges
            in the mesh).
    """
    ratios_a = compute_edgelength_ratios(mesh, edge_points, 0)
    ratios_b = compute_edgelength_ratios(mesh, edge_points, 3)
    ratios = torch.stack((ratios_a, ratios_b), dim=0)
    val, _ = torch.sort(ratios, dim=0)
    return val


def extract_meshcnn_features(mesh, edge_points):
    r"""Extract the various features used by MeshCNN.

    Args:
        mesh (kaolin.rep.TriangleMesh): Input (2-manifold) triangle mesh.
        edge_points (torch.Tensor): Computed edge points from the input
            triangle mesh (following MeshCNN convention).

    torch.unsqueeze(input, dim, out=None): 起作用是在指定维度上扩展一个维度
    例如：x = torch.Tensor([1, 2, 3, 4])
        print(x.size())  # torch.Size([4])
        print(torch.unsqueeze(x, 0))  # tensor([[1., 2., 3., 4.]])
        print(torch.unsqueeze(x, 0).size())  # torch.Size([1, 4])
    """
    dihedral_angles = compute_dihedral_angles(mesh, edge_points).unsqueeze(0)
    symmetric_opposite_angles = compute_symmetric_opposite_angles(mesh, edge_points)
    symmetric_edgelength_ratios = compute_symmetric_edgelength_ratios(mesh, edge_points)
    mesh.features = torch.cat(
        (dihedral_angles, symmetric_opposite_angles, symmetric_edgelength_ratios), dim=0
    )  # torch.cat()将两个tensor拼接起来，这里按dim=0拼接，形成每一个edge的特征向量


class MeshCNNConv(torch.nn.Module):
    r"""Implements the MeshCNN convolution operator. Recall that convolution is performed on the 1-ring
    neighbours of each (non-manifold) edge in the mesh.
    实现MeshCNN的卷积操作

    Args:
        in_channels (int): number of channels (features) in the input. 输入的特征数
        out_channels (int): number of channels (features) in the output.
        kernel_size (int): kernel size of the filter.滤波器的核size
        bias (bool, Optional): whether or not to use a bias term (default: True). 是否使用偏差的bool判断，并不是偏差项

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            bias: Optional[bool] = True,
    ):
        super(MeshCNNConv, self).__init__()  # 其父类为torch.nn.module
        self.conv = torch.nn.Conv2d(  # 2D卷积的一些参数
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),  # non-square kernels
            bias=bias,
        )
        self.kernel_size = kernel_size

    def __call__(self, edge_features: torch.Tensor, meshes: Iterable):
        r"""Calls forward when invoked.
        #__call__():使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
        当调用MeshCNNConv函数时，直接调用__call__()函数，也就是返回forward()

        Args:
            edge_features (torch.Tensor): input features of the mesh (shape: :math:`(B, F, E)`), where :math:`B` is
                the batchsize, :math:`F` is the number of features per edge, and :math:`E` is the number of edges
                in the input mesh.
            meshes (list[kaolin.rep.TriangleMesh]): list of TriangleMesh objects. Length of the list must be equal
                to the batchsize :math:`B` of `edge_features`.
        """
        return self.forward(edge_features, meshes)

    def forward(self, x, meshes):
        r"""Implements forward pass of the MeshCNN convolution operator.
        前向传播
        Args:
            x (torch.Tensor): input features of the mesh (shape: :math:`(B, F, E)`), where :math:`B` is
                the batchsize, :math:`F` is the number of features per edge, and :math:`E` is the number of edges
                in the input mesh.
            meshes (list[kaolin.rep.TriangleMesh]): list of TriangleMesh objects. Length of the list must be equal
                to the batchsize :math:`B` of `x`.
        """
        x = x.squeeze(-1)  # squeeze(x, dim),删除指定维度的单维度条目，若dim=None,则删除所有单维度条目
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in meshes], 0)  # 在指定维度（dim=0）上级联tensor
        # MeshCNN "trick": Build a "neighbourhood map" and apply 2D convolution.
        G = self.create_gemm(x, G)
        return self.conv(G)  # 在__init__中定义的conv2D

    def flatten_gemm_inds(self, Gi: torch.Tensor):
        r"""Flattens the indices of the gemm representation. 展平gemm表示的索引，也就是1-ring neighbor的相关操作
        """
        B, NE, NN = Gi.shape  # Gi的三个维度
        NE += 1
        batch_n = torch.floor(
            torch.arange(B * NE, device=Gi.device, dtype=torch.float) / NE
        ).view(B, NE)  # 指定shape为B行，NE列
        add_fac = batch_n * NE
        add_fac = add_fac.view(B, NE, 1)
        add_fac = add_fac.repeat(1, 1, NN)
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_gemm(self, x: torch.Tensor, Gi: torch.Tensor):
        r"""Gathers edge features (x) from within the 1-ring neighbours (Gi) and applies symmetric pooling for order
        invariance. Returns a "neighbourhood map" that we can use 2D convolution on.
        为了保证顺序的不变性，也就是论文中的公式2
        Args:
            x (torch.Tensor):
            Gi (torch.Tensor):
        """
        Gishape = Gi.shape
        # Zero-pad the first row of every sample in the batch.
        # TODO: Can replace by torch.nn.functional.pad()
        padding = torch.zeros(  # padding操作
            (x.shape[0], x.shape[1], 1),
            requires_grad=True,
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1

        # Flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()

        outdims = x.shape
        x = x.permute(0, 2, 1).contiguous()  # permute将tensor维度换位，contiguous:一般与transpose，permute，view搭配使用,不然容易报错
        x = x.view(outdims[0] * outdims[2], outdims[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)  # index_select从指定维度选取相应索引的数据
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # Perform "symmetrization" (ops defined in paper) for the convolution to be "equivariant".
        x1 = f[:, :, :, 1] + f[:, :, :, 3]
        x2 = f[:, :, :, 2] + f[:, :, :, 4]
        x3 = (f[:, :, :, 1] - f[:, :, :, 3]).abs()
        x4 = (f[:, :, :, 2] - f[:, :, :, 4]).abs()
        return torch.stack((f[:, :, :, 0], x1, x2, x3, x4), dim=3)

    def pad_gemm(self, mesh, desired_size: int, device: torch.device):
        r"""Extracts the 1-ring neighbours (four per edge), adds the edge itself to the list, and pads to
        `desired_size`.
        padding去这期望的size
        Args:
            mesh (kaolin.rep.TriangleMesh): Mesh to convolve over.
            desired_size (int): Desired size to pad to.

        """
        # padded_gemm = torch.tensor(mesh.gemm_edges, device=device, dtype=torch.float, requires_grad=True)
        padded_gemm = mesh.gemm_edges.clone().float()  # clone返回一个完全相同tensor,不共享内存，但是会留在计算图中。
        padded_gemm.requires_grad = True
        # TODO: Revisit when batching is implemented, to update `mesh.edges.shape[1]`.
        num_edges = mesh.edges.shape[-2]
        padded_gemm = torch.cat(
            (
                torch.arange(num_edges, device=device, dtype=torch.float).unsqueeze(-1),
                padded_gemm,
            ),
            dim=1,
        )
        padded_gemm = F.pad(
            padded_gemm, (0, 0, 0, desired_size - num_edges), "constant", 0
        )  # 在padded_gemm[-1]这个维度上padding0行0列，在在padded_gemm[-2]这个维度上padding0行(desired_size - num_edges)列
        return padded_gemm.unsqueeze(0)  # 在指定维度上增加尺寸为1的维度


class MeshCNNUnpool(torch.nn.Module):
    r"""Implements the MeshCNN unpooling operator.
    实现unpooling操作
    Args:
        unroll_target (int): number of target edges to unroll to. 也就是原来edge的数目

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, unroll_target: int):
        super(MeshCNNUnpool, self).__init__()
        self.unroll_target = unroll_target

    def __call__(self, features: torch.Tensor, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group: torch.Tensor, unroll_start: int):
        start, end = group.shape
        padding_rows = unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols != 0:
            padding = torch.nn.ConstantPad2d((0, padding_cols, 0, padding_rows),
                                             0)  # 使用常量值0填充 (pad_left, right, top, bottom)
            group = padding(group)
        return group

    def pad_occurrences(self, occurrences: torch.Tensor):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = torch.nn.ConstantPad1d((0, padding), 1)  # 用1填充
            occurrences = padding(occurrences)
        return occurrences

    def forward(self, features: torch.Tensor, meshes):
        r"""Implements forward pass of the MeshCNN convolution operator.

        Args:
            x (torch.Tensor): input features of the mesh (shape: :math:`(B, F, E)`), where :math:`B` is
                the batchsize, :math:`F` is the number of features per edge, and :math:`E` is the number of edges
                in the input mesh.
            meshes (list[kaolin.rep.TriangleMesh]): list of TriangleMesh objects. Length of the list must be equal
                to the batchsize :math:`B` of `x`.

        Returns:
            (torch.Tensor): output features, at the target unpooled size
                (shape: :math:`(B, F, \text{self.unroll_target})`)
        """
        B, F, E = features.shape
        groups = [self.pad_groups(mesh.get_groups(), E) for mesh in meshes]
        unroll_mat = torch.cat(groups, dim=0).view(B, E, -1)
        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(B, 1, -1)
        occurrences = occurrences.expand(unroll_mat.shape)
        unroll_mat = unroll_mat / occurrences
        urnoll_mat = unroll_mat.to(features)
        for mesh in meshes:
            mesh.unroll_gemm()
        return torch.matmul(features, unroll_mat)  # matmul两个矩阵向量的乘积


class MeshUnion:
    r"""Implements the MeshCNN "union" operator.

    Args:
        num_edges (int): number of edges to attach (i.e., perform "union").
        device (torch.device): device on which tensors reside.

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, num_edges: int, device: torch.device):
        self.groups = torch.eye(num_edges, device=device)  # eye生成对角线全1，其余部分全0的二维数组.
        self.rebuild_features = self.rebuild_features_average

    def union(self, source, target):
        self.groups[target, :] = self.groups[target, :] + self.groups[source, :]

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sum(self.groups, 0)

    def get_groups(self, mask):
        self.groups = self.groups.clamp(0, 1)
        return self.groups[mask, :]

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        faces_edges = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(faces_edges.shape)
        faces_edges = faces_edges / occurrences
        padding = target_edges - faces_edges.shape[1]
        if padding > 0:
            padding = torch.nn.ConstantPad2d((0, padding, 0, 0), 0)
            faces_edges = padding(faces_edges)
        return faces_edges

    def prepare_groups(self, features, mask):
        mask = torch.from_numpy(mask)
        self.groups = self.groups[mask, :].clamp(0, 1).transpose_(1, 0)
        padding = features.shape[1] - self.groups.shape[0]
        if padding > 0:
            padding = torch.nn.ConstantPad2d((0, 0, 0, padding), 0)
            self.groups = padding(self.groups)


class MeshPool(torch.nn.Module):
    r"""Implements the MeshCNN pooling operator.

    Args:
        target (int): number of target edges to pool to. 这个是Pooling需要保留的edge数目
        multi_thread (bool): Optionally run multi-threaded (default: False).

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        r"""Pool edges from the mesh and update features.

        Args:
            fe (torch.Tensor): Face-edge neighbourhood tensor.
            meshes (Iterable[kaolin.rep.TriangleMesh]): List of meshes to pool.

        Returns:
            out_features (torch.Tensor): Updated mesh features.

        """

        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:  # 如果使用多线程
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(
            len(meshes), -1, self.__out_target
        )
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(
            self.__fe[mesh_index, :, : mesh.edges_count], mesh.edges_count
        )
        # recycle = []
        # last_queue_len = len(queue)
        last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)  # heappop优先队列算法
            edge_id = int(edge_id)
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
        self.clean(mesh, mask, edge_groups)
        fe = edge_groups.rebuild_features(
            self.__fe[mesh_index], mask, self.__out_target
        )
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh, edge_id):
            return False
        elif (
                self.__clean_side(mesh, edge_id, mask, edge_groups, 0)
                and self.__clean_side(mesh, edge_id, mask, edge_groups, 2)
                and self.__is_one_ring_valid(mesh, edge_id)
        ):
            self.__merge_edges[0] = self.__pool_side(
                mesh, edge_id, mask, edge_groups, 0
            )
            self.__merge_edges[1] = self.__pool_side(
                mesh, edge_id, mask, edge_groups, 2
            )
            self.merge_vertices(mesh, edge_id)
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False

    def merge_vertices(self, mesh, edgeidx):
        self.remove_edge(mesh, edgeidx)
        edge = mesh.edges[edgeidx]
        v_a = mesh.vertices[edge[0]]
        v_b = mesh.vertices[edge[1]]
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        mesh.vertex_mask[edge[1]] = False
        mask = mesh.edges == edge[1]
        mesh.ve[edge[0]].extend(mesh.ve[edge[1]])
        mesh.edges[mask] = edge[0]

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    def clean(self, mesh, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        mesh.gemm_edges = mesh.gemm_edges[edges_mask]
        mesh.edges = mesh.edges[edges_mask]
        mesh.sides = mesh.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])  # concatenate()数组拼接
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        mesh.gemm_edges[:, :] = (
            torch.from_numpy(new_indices[mesh.gemm_edges[:, :]])
                .to(mesh.vertices.device)
                .long()
        )
        for v_index, ve in enumerate(mesh.ve):
            update_ve = []
            # if mesh.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        mesh.ve = new_ve
        mesh.pool_count += 1

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False

    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))  # set函数创建一个无序不重复元素集，删除重复数据。
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(
            mesh,
            key_a,
            side_a - side_a % 2,
            other_keys_b[0],
            mesh.sides[key_b, other_side_b],
        )
        self.__redirect_edges(
            mesh,
            key_a,
            side_a - side_a % 2 + 1,
            other_keys_b[1],
            mesh.sides[key_b, other_side_b + 1],
        )
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        self.remove_edge(mesh, key_b)
        mesh.edges_count -= 1
        return key_a

    def remove_edge(self, mesh, edgeidx):
        vs = mesh.edges[edgeidx]
        for v in vs:
            mesh.ve[v].remove(edgeidx)  # remove() 方法用于移除集合中的指定元素。

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        (
            key_a,
            key_b,
            side_a,
            side_b,
            other_side_a,
            other_side_b,
            other_keys_a,
            other_keys_b,
        ) = info
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert len(shared_items) == 2
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(
                mesh, edge_id, side + 1, update_key_b, update_side_b
            )
            MeshPool.__redirect_edges(
                mesh,
                update_key_a,
                MeshPool.__get_other_side(update_side_a),
                update_key_b,
                MeshPool.__get_other_side(update_side_b),
            )
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        other_keys_a = [
            mesh.gemm_edges[key_a, other_side_a],
            mesh.gemm_edges[key_a, other_side_a + 1],
        ]
        other_keys_b = [
            mesh.gemm_edges[key_b, other_side_b],
            mesh.gemm_edges[key_b, other_side_b + 1],
        ]
        return (
            key_a,
            key_b,
            side_a,
            side_b,
            other_side_a,
            other_side_b,
            other_keys_a,
            other_keys_b,
        )

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert len(vertex) == 1
        mesh.vertex_mask[vertex[0]] = False
        # mesh.remove_vertex(vertex[0])

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(
            edges_count, device=squared_magnitude.device, dtype=torch.float32
        ).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        # mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        # mesh.remove_group(index)


class MResConv(torch.nn.Module):
    r"""Implements a residual block of MeshCNNConv layers.
    实现残差层
    Args:
        in_channels (int): number of channels (features) in the input.
        out_channels (int): number of channels (features) in the output.
        skip (Optional, int): number of skip connected layers to add (default: 1).
        kernel_size (Optional, int): kernel size of the (2D) conv filter. (default: 5).

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, in_channels, out_channels, skip=1, kernel_size=5):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = 1
        self.conv0 = MeshCNNConv(
            self.in_channels, self.out_channels, kernel_size=kernel_size, bias=False
        )
        for i in range(self.skip):
            setattr(self, f"bn{i + 1}", torch.nn.BatchNorm2d(self.out_channels))  # setattr()用于设置属性值，该属性不一定是存在的。
            setattr(
                self,
                f"conv{i + 1}",  # f"xxx"格式化字符串常量，使格式化字符串的操作更加简便
                MeshCNNConv(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=kernel_size,
                    bias=False,
                ),
            )

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skip):
            x = getattr(self, f"bn{i + 1}")(F.relu(x))
            x = getattr(self, f"conv{i + 1}")(x, mesh)
        x = x + x1
        return F.relu(x)


class MeshCNNClassifier(torch.nn.Module):
    r"""Implements a MeshCNN classifier.

    Args:
        in_channels (int): number of channels (features) in the input.
        out_channels (int): number of channels (features) in the output (usually equal
            to the number of classes).
        conv_layer_sizes (Iterable): List of sizes of residual MeshCNNConv blocks to
            be used.
        pool_sizes (Iterable): Target number of edges in the mesh after each pooling
            step.
        fc_size (int): Number of neurons in the penultimate fully-connected layer.
        num_res_blocks (int): Number of residual blocks to use in the classifier.
        num_edges_in (int): Number of edges in the input mesh.

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            conv_layer_sizes,
            pool_sizes,
            fc_size,
            num_res_blocks,
            num_edges_in,
    ):
        super(MeshCNNClassifier, self).__init__()
        self.layer_sizes = [in_channels] + conv_layer_sizes
        self.edge_sizes = [num_edges_in] + pool_sizes

        for i, size in enumerate(self.layer_sizes[:-1]):
            setattr(
                self,
                f"conv{i}",
                MResConv(size, self.layer_sizes[i + 1], num_res_blocks),
            )
            setattr(self, f"pool{i}", MeshPool(self.edge_sizes[i + 1]))

        self.global_pooling = torch.nn.AvgPool1d(self.edge_sizes[-1])
        self.fc1 = torch.nn.Linear(self.layer_sizes[-1], fc_size)
        self.fc2 = torch.nn.Linear(fc_size, out_channels)

    def forward(self, x, mesh):
        for i in range(len(self.layer_sizes) - 1):
            x = F.relu(getattr(self, f"conv{i}")(x, mesh))
            x = getattr(self, f"pool{i}")(x, mesh)
        x = self.global_pooling(x)
        x = x.view(-1, self.layer_sizes[-1])
        x = F.relu(self.fc1(x))
        return self.fc2(x)
