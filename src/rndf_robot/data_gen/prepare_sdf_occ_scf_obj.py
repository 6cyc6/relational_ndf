import os, os.path as osp

from pyscf import batch_compute_scfs

from rndf_robot.utils.mesh_util import inside_mesh

# Enable this when running on a computer without a screen:
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pickle
import traceback
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import trimesh
from rndf_robot.utils.mesh_util.three_util import get_raster_points
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, BadMeshException

from rndf_robot.utils import util, path_util
import polyscope as ps

ensure_directory = util.safe_makedirs

obj_type = "cup"

n_pts = 200000
voxel_resolution = 128
# Options for virtual scans used to generate SDFs
USE_DEPTH_BUFFER = True
SCAN_COUNT = 50
SCAN_RESOLUTION = 1024

base_dir = path_util.get_rndf_obj_descriptions()

mesh_dir = osp.join(base_dir, obj_type)
save_dir = osp.join(base_dir, "labels", obj_type)
util.safe_makedirs(save_dir)

obj_list = os.listdir(mesh_dir)

n_total = len(obj_list)
bad_list = []
save_dict = {}
for obj in tqdm(obj_list):
    obj_id = obj.split('.')[0]

    if os.path.exists(osp.join(save_dir, f'{obj_id}.npz')):
        print(f"{obj} already exists. ")
        continue
    if obj.split(".")[-1] != "obj":
        print(f"File not in correct format.")
        continue
    try:
        print(f'Processing {obj} ...')
        mesh_path = osp.join(mesh_dir, obj)
        obj_mesh = trimesh.load_mesh(mesh_path, process=False)

        vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
        norm_factor = 0.9 / np.max(obj_mesh.bounding_box.extents)
        vertices *= norm_factor
        mesh_scaled = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)

        sample_points = get_raster_points(voxel_resolution)

        occ = inside_mesh.check_mesh_contains(mesh_scaled, sample_points)

        occ_in_idx = np.where(occ == 1)[0]
        occ_out_idx = np.where(occ == 0)[0]

        assert sample_points.shape[0] == occ_in_idx.shape[0] + occ_out_idx.shape[0]

        n_in = occ_in_idx.shape[0]
        print(n_in)

        if n_in < 20000:
            bad_list.append(obj)
            print(f"bad: {obj}")

        if n_in > 80000:
            slice_in_idx = np.random.choice(occ_in_idx, 80000, replace=False)
            n_in = 80000
        else:
            slice_in_idx = occ_in_idx

        n_out = n_pts - n_in
        slice_out_idx = np.random.choice(occ_out_idx, n_out, replace=False)

        idx = np.concatenate([slice_in_idx, slice_out_idx])
        pts = sample_points[idx]
        occ = occ[idx]

        surface_point_cloud = get_surface_point_cloud(mesh_scaled, bounding_radius=1, scan_count=SCAN_COUNT,
                                                      scan_resolution=SCAN_RESOLUTION)
        print("calculating sdf ...")
        sdf = surface_point_cloud.get_sdf_in_batches(pts, use_depth_buffer=True, return_gradients=False)

        print("calculating scf ...")
        scf = batch_compute_scfs(mesh_scaled, pts)[..., :5]

        combined = np.concatenate((pts, occ[:, np.newaxis], sdf[:, np.newaxis], scf), axis=1)

        # obj_id = obj.split('.')[0]
        print(f'Saving result to obj id: {obj_id}')
        save_path = osp.join(save_dir, obj_id)
        np.savez(save_path, label=combined, norm_factor=norm_factor)

        tup = (pts, occ, sdf, scf, norm_factor)
        save_dict[f"{obj_type}_{obj_id}"] = tup
        # # for visualization
        # pts = mesh_scaled.sample(2000)
        #
        # ps.init()
        # ps.set_up_dir("z_up")
        #
        # ps.register_point_cloud("pcd", pts, radius=0.005, color=[0, 0, 1], enabled=True)
        # ps.register_point_cloud("occ", pts_in, radius=0.005, color=[0, 1, 0], enabled=True)
        # ps.register_point_cloud("out", pts_out[:3000], radius=0.005, color=[1, 0, 0], enabled=True)
        #
        # ps.show()
    except Exception as e:
        print("Could not load file")

with open(osp.join(base_dir, "labels", f'{obj_type}_occ_sdf_scf.p'), 'wb') as f:
    pickle.dump(save_dict, f)
f.close()
print("All saved.")

