import glob
import os.path as osp
import os
import pickle
import random

import meshcat
import torch
import trimesh
import numpy as np
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import polyscope as ps

from rndf_robot.utils import util, path_util
from rndf_robot.utils.geometry import lift
from rndf_robot.utils.mesh_util.three_util import get_raster_points, get_occ

index = 0

voxel_res = 128
category = "mug"
side_length = 128
block = 128
bs = 1 / block
hbs = bs * 0.5

SCAN_COUNT = 50
SCAN_RESOLUTION = 1024

ndf_root = os.environ['NDF_SOURCE_DIR']
ndf_training_dir = f'{ndf_root}/src/ndf_robot/data/training_data/mug_table_all_pose_4_cam_half_occ_full_rand_scale'
points_dir = f'{ndf_root}/src/ndf_robot/data/training_data'
mesh_dir = f'{ndf_root}/src/ndf_robot/descriptions/objects'
save_dir = f'{ndf_root}/src/ndf_robot/data/training_data/scfs'
pts_dir = f'{ndf_root}/src/ndf_robot/data/training_data'


# data = np.load(files_total[index], allow_pickle=True)
data = np.load("/home/ikun/master-thesis/relational_ndf/src/rndf_robot/data/training_data/test_hammer/1_3_1.npz", allow_pickle=True)
posecam = data['object_pose_cam_frame']  # legacy naming, used to use pose expressed in camera frame. global reference frame doesn't matter though

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

# shapenet_id = str(data['shapenet_id'].item())
# category_id = str(data['shapenet_category_id'].item())
obj_name = str(data["obj_file"].item())
obj_id = obj_name.split('/')[-1].split('.')[0]
obj_path = osp.join("/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects", obj_name)
label_path = osp.join("/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/sdf/hammer/cloud", obj_id + ".npz")
label = np.load(label_path, allow_pickle=True)
occ_sdf_scf = label["coords_sdf"]
norm_factor = label["norm_factor"]

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

# voxel_path = osp.join(category_id, shapenet_id, 'models', 'model_normalized_128.mat')
# coord, voxel_bool, _ = shapenet_mug_dict[voxel_path]
coord = occ_sdf_scf[:, 0:3]
voxel_bool = occ_sdf_scf[:, 3]

rix = np.random.permutation(coord.shape[0])

# coord = coord[rix[:1500]]
# label_occ = voxel_bool[rix[:1500]]
coord = coord[rix[:]]
label_occ = voxel_bool[rix[:]]
coord_ = coord

offset = np.random.uniform(-hbs, hbs, coord.shape)
coord = coord + offset
coord = coord * norm_factor
coord = coord * data['mesh_scale']

coord = torch.from_numpy(coord)

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
coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
coord = coord[..., :3]

points_world = []

for i, dp_np in enumerate(dp_nps):
    point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
    dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
    points_world.append(dp_np[..., :3])

point_cloud = torch.cat(points_world, dim=0)

rix = torch.randperm(point_cloud.size(0))
point_cloud = point_cloud[rix[:1000]]

point_cloud_np = point_cloud.cpu().numpy()

occ_ = label_occ
coord_np = coord.cpu().numpy()
occ_pts_ = coord_np[np.where(occ_)[0]]
non_occ_pts_ = coord_np[np.where(np.logical_not(occ_))[0]]
occ_pts = coord_np[np.where(occ_)[0]]

ps.init()
ps.set_up_dir("z_up")

ps.register_point_cloud("pcd", point_cloud_np, radius=0.005, color=[1, 0, 0], enabled=True)
ps.register_point_cloud("occ", occ_pts, radius=0.005, color=[0, 0, 1], enabled=True)

ps.show()

print(1)

# check occ label
# sample_points = get_raster_points(voxel_res)
#
# # create mesh
# # obj_fname = mesh_path
# obj_fname = mesh128_path
# obj_mesh = trimesh.load(obj_fname, process=False)
# scene = trimesh.Scene()
# scene.add_geometry([obj_mesh])
# scene.show()
#
# # coord, voxel_bool, _ = data_dict[data_id]
# coord, occ_, _ = data_dict[data_id]
#
# vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
# norm_factor = 1 / np.max(obj_mesh.bounding_box.extents)
# vertices *= norm_factor
# obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)
# #
# #
# occ = inside_mesh.check_mesh_contains(obj_mesh, sample_points)
# # _, occ, scf = get_occ(obj_mesh, voxel_res, sample_points=sample_points)
#
# occ_pts = sample_points[np.where(occ)[0]]
# non_occ_pts = sample_points[np.where(np.logical_not(occ))[0]]
# occ_pts_ = coord[np.where(occ_)[0]]
# non_occ_pts_ = coord[np.where(np.logical_not(occ_))[0]]

# ps.init()
# ps.set_up_dir("z_up")
# ps.register_point_cloud("obj", occ_pts_, radius=0.005, color=[0, 0, 1], enabled=True)
# ps.register_point_cloud("obj_non", non_occ_pts_, radius=0.005, color=[1, 0, 0], enabled=False)
# ps.register_point_cloud("obj_new", occ_pts, radius=0.005, color=[0, 1, 0], enabled=True)
# ps.register_point_cloud("obj_non_new", non_occ_pts, radius=0.005, color=[1, 0, 0], enabled=False)
# ps.show()

