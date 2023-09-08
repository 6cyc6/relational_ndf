import glob
import os.path as osp
import os
import pickle
import random

import meshcat
import torch
import trimesh
import numpy as np
from tqdm import tqdm

import polyscope as ps

from rndf_robot.utils import util, path_util, geometry
from rndf_robot.utils.mesh_util import inside_mesh
from rndf_robot.utils.mesh_util.three_util import get_raster_points, get_occ

voxel_res = 128
category = "mug"
index = 0

ndf_root = os.environ['NDF_SOURCE_DIR']
ndf_training_dir = f'{ndf_root}/src/ndf_robot/data/training_data/mug_table_all_pose_4_cam_half_occ_full_rand_scale'
points_dir = f'{ndf_root}/src/ndf_robot/data/training_data'
mesh_dir = f'{ndf_root}/src/ndf_robot/descriptions/objects'
save_dir = f'{ndf_root}/src/ndf_robot/data/training_data/scfs'
pts_dir = f'{ndf_root}/src/ndf_robot/data/training_data'


sample_points = get_raster_points(voxel_res)
# mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')

data_dict = pickle.load(open(f'{points_dir}/occ_shapenet_{category}.p', 'rb'))
# pts_dict = pickle.load(open(osp.join(pts_dir, 'occ_shapenet_mug.p'), 'rb'))
data_ids = list(data_dict.keys())
os.makedirs(f'{save_dir}/{category}', exist_ok=True)
for data_id in tqdm(data_ids):
    category_id, shapenet_id, _, _ = data_id.split('/')
    mesh_path = f'{mesh_dir}/{category}_centered_obj_normalized/{shapenet_id}/models/model_normalized.obj'
    mesh128_path = f'{mesh_dir}/{category}_centered_obj/{shapenet_id}/models/model_128_df.obj'
    # mesh128_path = f'{mesh_dir}/{category}_centered_obj_normalized/{shapenet_id}/models/model_normalized_dec.obj'
    if not os.path.exists(mesh128_path): continue
    break
# create mesh
# obj_fname = mesh_path
obj_fname = mesh128_path
obj_mesh = trimesh.load(obj_fname, process=False)
scene = trimesh.Scene()
scene.add_geometry([obj_mesh])
scene.show()

# coord, voxel_bool, _ = data_dict[data_id]
coord, occ_, _ = data_dict[data_id]

vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
norm_factor = 1 / np.max(obj_mesh.bounding_box.extents)
vertices *= norm_factor
obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)
#
#
occ = inside_mesh.check_mesh_contains(obj_mesh, sample_points)
_, occ, scf = get_occ(obj_mesh, voxel_res, sample_points=sample_points)

occ_pts = sample_points[np.where(occ)[0]]
non_occ_pts = sample_points[np.where(np.logical_not(occ))[0]]
occ_pts_ = coord[np.where(occ_)[0]]
non_occ_pts_ = coord[np.where(np.logical_not(occ_))[0]]
# util.meshcat_pcd_show(mc_vis, occ_pts, color=[255, 0, 0], name='scene/occ_pts')
# util.meshcat_pcd_show(mc_vis, non_occ_pts, color=[255, 0, 255], name='scene/non_occ_pts')

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("obj", occ_pts_, radius=0.005, color=[0, 0, 1], enabled=True)
ps.register_point_cloud("obj_non", non_occ_pts_, radius=0.005, color=[1, 0, 0], enabled=False)
ps.register_point_cloud("obj_new", occ_pts, radius=0.005, color=[0, 1, 0], enabled=True)
ps.register_point_cloud("obj_non_new", non_occ_pts, radius=0.005, color=[1, 0, 0], enabled=False)
ps.show()

# # save
# new_obj_name = obj_name + '_' + obj_number
#
# occ_save_fname = osp.join(occ_save_dir, new_obj_name + '_occupancy.npz')
# normalized_saved_obj_fname = osp.join(mesh_save_dir, new_obj_name + '.obj')
#
# normalized_saved_obj_fname_relative = normalized_saved_obj_fname.split(path_util.get_rndf_obj_descriptions())[1].lstrip(
#     '/')
# obj_fname_relative = obj_fname.split(path_util.get_rndf_obj_descriptions())[1].lstrip('/')
#
# print(f'Saving to... \nnpz file: {occ_save_fname}\nmesh_file: {normalized_saved_obj_fname_relative}')
#
# np.savez(
#     occ_save_fname,
#     mesh_fname=obj_fname_relative,
#     normalized_mesh_fname=normalized_saved_obj_fname_relative,
#     points=sample_points,
#     occupancy=occ.reshape(-1),
#     norm_factor=norm_factor
# )
#
