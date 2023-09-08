from matplotlib import pyplot as plt
from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np
import pyvista as pv

import polyscope as ps
from pyscf import compute_spfs, batch_sph2scf, normalize_sph

# mesh = trimesh.load('/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/hammer/model_origin.obj', process=False)
mesh = trimesh.load('/home/ikun/obj/hammer/3/model.obj', process=False)

vertices = mesh.vertices - mesh.bounding_box.centroid
distances = np.linalg.norm(vertices, axis=1)
vertices /= np.max(distances)

mesh_unit_sphere = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

points, sdf = sample_sdf_near_surface(mesh_unit_sphere, number_of_points=200000, sign_method="depth")
# points, sdf = sample_sdf_near_surface(mesh_unit_sphere, number_of_points=200000)

side_length = 128
block = 128
bs = 1 / block
hbs = bs * 0.5

print((sdf < 0).shape)
# colors = np.zeros(points.shape)
# colors[sdf < 0, 2] = 1
# colors[sdf > 0, 0] = 1
# cloud = pyrender.Mesh.from_points(points, colors=colors)
# scene = pyrender.Scene()
# scene.add(cloud)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

occ_pts = points[np.where(sdf < -hbs)[0]]

occ_no = points[np.where(sdf > hbs)[0]]

mesh = mesh.apply_scale(1/mesh.scale)
points = mesh.bounding_sphere.to_mesh().bounding_box.sample_grid(4)
spfs,rays = compute_spfs(mesh, points, return_rays=True)
scfs = batch_sph2scf(spfs)
# plt.imshow(scfs[:, :5].T)
# plt.xlabel('points')
# plt.ylabel('scf')
# # plt.yticks([0,1,2,3,4])
# plt.tight_layout()
# plt.show()

pl = pv.Plotter(window_size=(500, 500))
pl.add_mesh(mesh, opacity=0.5)
for i in range(len(points)):
    sp = rays[i][:, :3]+rays[i][:, 3:]*0.05
    spf = spfs[i]
    spf_norm = normalize_sph(spf)
    pl.add_mesh(sp, scalars=spf_norm, opacity=0.5, show_scalar_bar=False)
pl.show(screenshot='./imgs/spf.png')

ps.init()
ps.set_up_dir("z_up")

ps.register_point_cloud("occ", occ_pts, radius=0.005, color=[0, 1, 0], enabled=True)
ps.register_point_cloud("occ_no", occ_no, radius=0.005, color=[1, 0, 0], enabled=True)

ps.show()
