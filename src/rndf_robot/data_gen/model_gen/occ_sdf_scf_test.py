from matplotlib import pyplot as plt
from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np
import pyvista as pv

import polyscope as ps
from pyscf import compute_spfs, batch_sph2scf, normalize_sph

mesh = trimesh.load('/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/hammer/hammer_10.obj', process=False)
save_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/sdf/hammer_test/cloud/hammer_10.npz"

data = np.load(save_path, allow_pickle=True)
result = data["coords_sdf"]
norm_factor = data["norm_factor"]

pcd = mesh.sample(5000)

pts = result[:, 0:3]
occ = result[:, 3]
occ_in = pts[occ == 1]
occ_out = pts[occ == 0]

ps.init()
ps.set_up_dir("z_up")

ps.register_point_cloud("occ", occ_in * norm_factor, radius=0.005, color=[0, 1, 0], enabled=True)
ps.register_point_cloud("occ_out", occ_out * norm_factor, radius=0.005, color=[1, 0, 0], enabled=True)
ps.register_point_cloud("pcd", pcd, radius=0.005, color=[1, 0, 1], enabled=True)

ps.show()
