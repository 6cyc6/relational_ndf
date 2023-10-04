from scipy.spatial.transform import Rotation
from rndf_robot.utils.geometry import lift
import trimesh

import numpy as np
import pyvista as pv

import polyscope as ps
import torch
from pyscf import compute_spfs, batch_sph2scf, normalize_sph

label_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/labels/container_occ_sdf_scf.p"
# norm_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/data/training_data_non_shapenet/container_occ.p"
# norm_label = np.load(norm_path, allow_pickle=True)
label = np.load(label_path, allow_pickle=True)

pcd_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/data/training_data_non_shapenet/container/0_14_741.npz"
data = np.load(pcd_path, allow_pickle=True)
obj_file = data['obj_file'].item()

obj_id = obj_file.split('.')[0].split('_')[-1]
print(obj_id)

side_length = 128
block = 256
bs = 1 / block
hbs = bs * 0.5

pts, occ, sdf, scf, norm_factor = label[f"container_{obj_id}"]
# _, _, norm_factor = norm_label[f"container_{obj_id}"]

mesh_scale = data["mesh_scale"]
posecam = data['object_pose_cam_frame']
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

coord = pts[occ == 1]
# coord = pts[occ == 0]
np.random.shuffle(coord)
coord = coord[:2000]

transform = transforms[0]
offset = np.random.uniform(-hbs, hbs, coord.shape)
coord = coord + offset
coord = coord * norm_factor / 0.9
coord = coord * mesh_scale

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

ps.register_point_cloud("occ", coord, radius=0.005, color=[0, 1, 0], enabled=True)
# ps.register_point_cloud("occ_out", occ_out * norm_factor, radius=0.005, color=[1, 0, 0], enabled=True)
ps.register_point_cloud("pcd", pcd, radius=0.005, color=[1, 0, 1], enabled=True)

ps.show()
