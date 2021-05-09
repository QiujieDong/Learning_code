import os
import sys
import pytest  # pytest是python的测试工具，可以用于所有类型和级别的软件测试
import torch
from pathlib import Path

from kaolin.rep import TriangleMesh, VoxelGrid, PointCloud
from kaolin.conversions.meshconversions import trianglemesh_to_pointcloud, trianglemesh_to_voxelgrid

# Skip test if import fails unless on CI and not on Windows
if os.environ.get('CI') and not sys.platform == 'win32':  # os.environ.get（）：os模块获取环境变量的一个方法
    from kaolin.visualize.vis_usd import VisUsd
else:
    VisUsd = pytest.importorskip('kaolin.visualize.vis_usd.VisUsd', reason='The pxr library could not be imported')

root = Path('tests/visualize/results')
root.mkdir(exist_ok=True)  # 在目录（这里是root的path）不存在时创建目录，目录已存在时不会抛出异常。
mesh = TriangleMesh.from_obj('tests/model.obj')
voxels = VoxelGrid(trianglemesh_to_voxelgrid(mesh, 32))
pc = PointCloud(trianglemesh_to_pointcloud(mesh, 500)[0])

vis = VisUsd()


@pytest.mark.parametrize('object_3d', [mesh, voxels, pc])  # 测试时的参数设置
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('meet_ground', [True, False])
@pytest.mark.parametrize('center_on_stage', [True, False])
@pytest.mark.parametrize('fit_to_stage', [True, False])
def test_vis(object_3d, device, meet_ground, center_on_stage, fit_to_stage):
    if device == 'cuda':
        if isinstance(object_3d, TriangleMesh):  # isinstance是不是类的一个实例
            object_3d.cuda()
        elif isinstance(object_3d, PointCloud):
            object_3d.points = object_3d.points.to(torch.device(device))
        elif isinstance(object_3d, VoxelGrid):
            object_3d.voxels = object_3d.voxels.to(torch.device(device))

    vis.set_stage(filepath=str(root / f'{type(object_3d).__name__}_{device}.usda'))
    vis.visualize(object_3d, meet_ground=meet_ground, center_on_stage=center_on_stage,
                  fit_to_stage=fit_to_stage)
