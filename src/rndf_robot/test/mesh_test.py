# # Run the algorithm
# ./build/manifold --input /home/ikun/obj/hammer/1/model_origin.obj --output /home/ikun/obj/hammer/1/model.obj
# ./build/manifold --input /home/ikun/obj/hammer/2/model_origin.obj --output /home/ikun/obj/hammer/2/model.obj
#
import glob
import os.path as osp
import os
import pickle
import random

import meshcat
import torch
import open3d as o3d
import trimesh
import numpy as np
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import polyscope as ps

from rndf_robot.utils import util, path_util
from rndf_robot.utils.geometry import lift
from rndf_robot.utils.mesh_util import inside_mesh
from rndf_robot.utils.mesh_util.three_util import get_raster_points, get_occ

SCAN_COUNT = 50
SCAN_RESOLUTION = 1024
voxel_res = 128
category = "mug"
side_length = 128
block = 128
bs = 1 / block
hbs = bs * 0.5

pcd_name = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/data/training_data/bottle_table_all_pose_4_cam_half_occ_full_rand_scale/0_4_71.npz"
data = np.load(pcd_name, allow_pickle=True)
shapenet_id = data["shapenet_id"]
print(shapenet_id)
mesh_scale = data['mesh_scale']
posecam = data['object_pose_cam_frame']  # legacy naming, used to use pose expressed in camera frame. global reference frame doesn't matter though

ndf_root = os.environ['NDF_SOURCE_DIR']
pts_dir = f'{ndf_root}/src/ndf_robot/data/training_data'
shapenet_mug_dict = pickle.load(open(osp.join(pts_dir, 'occ_shapenet_bottle.p'), 'rb'))
voxel_path = f'02876657/{shapenet_id}/models/model_normalized_128.mat'
coord, voxel_bool, _ = shapenet_mug_dict[voxel_path]

occ_pts_label = coord[np.where(voxel_bool)[0]]
occ_not_pts = coord[np.where(np.logical_not(voxel_bool))[0]]

sdf_path = f"/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/sdf/bottle_centered_obj_normalized_sdf/cloud/{shapenet_id}.npz"
data = np.load(sdf_path, allow_pickle=True)
coords = data["coords_sdf"]
norm_factor_data = data["norm_factor"]

occ_in = coords[:, :3][np.where(np.abs(coords[:, -1]) < hbs)[0]]

# obj_fname = "/home/ikun/master-thesis/ndf_robot/src/ndf_robot/descriptions/objects/bottle_centered_obj/1b64b36bf7ddae3d7ad11050da24bb12/models/model_128_df.obj"
obj_fname = f"/home/ikun/master-thesis/ndf_robot/src/ndf_robot/descriptions/objects/bottle_centered_obj_normalized/{shapenet_id}/models/model_normalized.obj"
obj_fname_ = f"/home/ikun/master-thesis/ndf_robot/src/ndf_robot/descriptions/objects/bottle_centered_obj/{shapenet_id}/models/model_128_df.obj"

obj_mesh = trimesh.load(obj_fname, process=False)
obj_mesh_ = trimesh.load(obj_fname_, process=False)
# pcd_obj = obj_mesh.sample(5000)
# pcd_mean = np.mean(pcd_obj, axis=0)
# pcd_obj = pcd_obj - pcd_mean
# scene = trimesh.Scene()
# scene.add_geometry([obj_mesh, obj_mesh_])
# scene.show()

vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
distances = np.linalg.norm(vertices, axis=1)
vertices /= np.max(distances)

mesh_unit_sphere = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)
norm_factor = np.max(distances)

scene = trimesh.Scene()
scene.add_geometry([mesh_unit_sphere])
scene.show()

surface_point_cloud = get_surface_point_cloud(mesh_unit_sphere, bounding_radius=1, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)

sdf_points, sdf_values = surface_point_cloud.sample_sdf_near_surface(use_scans=True, sign_method='depth', number_of_points=200000)
# occ_pts_ = sdf_points[np.where(np.abs(sdf_values) < hbs)[0]]
occ_pts_ = sdf_points[np.where(sdf_values < -hbs)[0]]

# new_pts = get_raster_points(128)
# occ = inside_mesh.check_mesh_contains(obj_mesh, new_pts)
# occ_pts_ = new_pts[np.where(occ)[0]]
# occ_not_pts_ = new_pts[np.where(np.logical_not(occ))[0]]
#

data = np.load(pcd_name, allow_pickle=True)
idxs = list(range(posecam.shape[0]))
# random.shuffle(idxs)
# select = random.randint(1, 4)

poses = []
quats = []
for i in idxs:
    pos = posecam[i, :3]
    quat = posecam[i, 3:]

    poses.append(pos)
    quats.append(quat)

shapenet_id = str(data['shapenet_id'].item())
category_id = str(data['shapenet_category_id'].item())

depths = []
segs = []
cam_poses = []

for i in idxs:
    seg = data['object_segmentation'][i, 0]
    depth = data['depth_observation'][i]
    rix = np.random.permutation(depth.shape[0])[:1000]
    seg = seg[rix]
    depth = depth[rix]

    segs.append(seg)
    depths.append(torch.from_numpy(depth))
    cam_poses.append(data['cam_pose_world'][i])

# change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
# y, x = torch.meshgrid(torch.arange(480), torch.arange(640))
x, y = torch.meshgrid(torch.arange(640), torch.arange(480), indexing='xy')

# Compute native intrinsic matrix
sensor_half_width = 320
sensor_half_height = 240

vert_fov = 60 * np.pi / 180

vert_f = sensor_half_height / np.tan(vert_fov / 2)
hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

intrinsics = np.array(
    [[hor_f, 0., sensor_half_width, 0.],
    [0., vert_f, sensor_half_height, 0.],
    [0., 0., 1., 0.]]
)

# Rescale to new sidelength
intrinsics = torch.from_numpy(intrinsics)

# build depth images from data
dp_nps = []
for i in range(len(segs)):
    seg_mask = segs[i]
    dp_np = lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
    dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
    dp_nps.append(dp_np)

# transform everything into the same frame
transforms = []
for quat, pos in zip(quats, poses):
    quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    rotation_matrix = Rotation.from_quat(quat_list)
    rotation_matrix = rotation_matrix.as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, -1] = pos
    transform = torch.from_numpy(transform)
    transforms.append(transform)

transform = transforms[0]
points_world = []

for i, dp_np in enumerate(dp_nps):
    point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
    dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
    points_world.append(dp_np[..., :3])

point_cloud = torch.cat(points_world, dim=0)

# coord = occ_pts_label
coord = occ_pts_
transform = transforms[0]
offset = np.random.uniform(-hbs, hbs, coord.shape)
coord = coord + offset
coord = coord * norm_factor
coord = coord * data['mesh_scale']

coord = torch.from_numpy(coord)

coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
coord = coord[..., :3]
center = point_cloud.mean(dim=0)

coord = coord - center[None, :]
point_cloud = point_cloud - center[None, :]

pcd = point_cloud.cpu().numpy()
coord = coord.cpu().numpy()

ps.init()
ps.set_up_dir("z_up")
# sample_points = ps.register_point_cloud("samples", coords[:, :3] * norm_factor)
# sample_points.add_scalar_quantity("rand vals with range", coords[:, -1], cmap='coolwarm', enabled=True)
# ps.register_point_cloud("occ_pts", occ_pts, radius=0.005, color=[1, 0, 1], enabled=True)
# ps.register_point_cloud("occ_not_pts", occ_not_pts, radius=0.005, color=[1, 0, 1], enabled=True)
# ps.register_point_cloud("pcd", pcd, radius=0.005, color=[1, 0, 1], enabled=True)
# ps.register_point_cloud("occ_in_sdf", occ_in * norm_factor_data, radius=0.005, color=[1, 0, 0], enabled=True)
# ps.register_point_cloud("occ", occ_pts_ * norm_factor, radius=0.005, color=[0, 1, 0], enabled=True)
# ps.register_point_cloud("occ_label", occ_pts_ * norm_factor, radius=0.005, color=[0, 0, 1], enabled=True)
ps.register_point_cloud("coord", coord, radius=0.005, color=[0, 0, 1], enabled=True)
ps.register_point_cloud("pcd", pcd, radius=0.005, color=[1, 1, 1], enabled=True)
# ps.register_point_cloud("occ_not", occ_not_pts_, radius=0.005, color=[1, 1, 1], enabled=True)

ps.show()

# voxel_res = 128
# category = "mug"
# side_length = 128
# block = 128
# bs = 1 / block
# hbs = bs * 0.5
#
# SCAN_COUNT = 50
# SCAN_RESOLUTION = 1024
#
# obj_fname = "/home/ikun/obj/hammer/4/model_origin.obj"
# # obj_fname = "/home/ikun/obj/mug/mug_centered_obj/1a97f3c83016abca21d0de04f408950f/models/model_128_df.obj"
#
# obj_mesh = trimesh.load(obj_fname, process=False)
# obj_pcd = obj_mesh.sample(5000)
#
# # mesh = o3d.io.read_triangle_mesh(obj_fname)
# # # mesh.compute_vertex_normals()
# #
# # # edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
# # # edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
# # # vertex_manifold = mesh.is_vertex_manifold()
# # # self_intersecting = mesh.is_self_intersecting()
# # watertight = mesh.is_watertight()
# # # orientable = mesh.is_orientable()
# #
# # # print(name)
# # # print(f"  edge_manifold:          {edge_manifold}")
# # # print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
# # # print(f"  vertex_manifold:        {vertex_manifold}")
# # # print(f"  self_intersecting:      {self_intersecting}")
# # print(f"  watertight:             {watertight}")
# # # print(f"  orientable:             {orientable}")
#
# surface_point_cloud = get_surface_point_cloud(obj_mesh, bounding_radius=1, scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
#
# sample_points = get_raster_points(voxel_res)
# occ = inside_mesh.check_mesh_contains(obj_mesh, sample_points)
# # check_pts, occ, scf = get_occ(obj_mesh, voxel_res)
# # sdf_values = surface_point_cloud.get_sdf_in_batches(sample_points, use_depth_buffer=True, return_gradients=False)
#
# # sdf_points, sdf_values = surface_point_cloud.sample_sdf_near_surface(use_scans=True, sign_method='depth', number_of_points=200000, min_size=0.015)
# sdf_points, sdf_values = surface_point_cloud.sample_sdf_near_surface(use_scans=True, sign_method='depth', number_of_points=200000)
# occ_in = sdf_points[np.where(np.abs(sdf_values) < 0.002)[0]]
#
# # occ = check_pts[np.where(occ)[0]]
# occ = sample_points[np.where(occ)[0]]
#
# ps.init()
# ps.set_up_dir("z_up")
# # ps.register_point_cloud("obj", occ_pts_, radius=0.005, color=[0, 0, 1], enabled=True)
# # ps.register_point_cloud("obj_non", non_occ_pts_, radius=0.005, color=[1, 0, 0], enabled=True)
# # ps.register_point_cloud("pcd", point_cloud_np, radius=0.005, color=[0, 1, 0], enabled=True)
# # ps.register_point_cloud("obj_", occ_pts, radius=0.005, color=[1, 0, 0], enabled=True)
# ps.register_point_cloud("obj_origin", obj_pcd, radius=0.005, color=[0, 1, 0], enabled=True)
# ps.register_point_cloud("occ", occ, radius=0.001, color=[0, 0, 1], enabled=True)
# ps.register_point_cloud("occ_in", occ_in, radius=0.001, color=[1, 0, 1], enabled=False)
#
# ps.show()
