import copy
import os, os.path as osp
import time

from pyscf import batch_compute_scfs

from rndf_robot.utils.mesh_util import inside_mesh

# Enable this when running on a computer without a screen:
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pickle
import traceback
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import numpy as np
import trimesh
from rndf_robot.utils.mesh_util.three_util import get_raster_points
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, BadMeshException

from rndf_robot.utils import util, path_util
import polyscope as ps

USE_DEPTH_BUFFER = True
SCAN_COUNT = 50
SCAN_RESOLUTION = 1024

obj_type = "hammer"

base_dir = path_util.get_data_src()
mesh_dir = osp.join(base_dir, "mesh", obj_type)
pcd_dir = osp.join(base_dir, obj_type)
save_dir = osp.join(base_dir, "labels", obj_type)
data_dict = pickle.load(open(f'{base_dir}/{obj_type}_occ_sdf_scf.p', 'rb'))


def process_model_file(filename):
    f_name = filename.split('.')[0]
    save_path = osp.join(save_dir, f_name)
    if os.path.exists(save_path):
        print("File already exists.")
    else:
        try:
            pcd_data = np.load(osp.join(pcd_dir, filename), allow_pickle=True)
            obj = pcd_data['obj_file'].item()
            obj_id = obj.split('/')[-1].split('.')[0]

            mesh_scale = pcd_data["mesh_scale"]
            pts, occ, _, _, _ = data_dict[f"{obj_type}_{obj_id}"]

            # load mesh model and apply scaling
            mesh = trimesh.load_mesh(osp.join(mesh_dir, f"{obj_id}.obj"), process=False)
            trans_mat = np.eye(4)
            trans_mat[0, 0] = mesh_scale[0]
            trans_mat[1, 1] = mesh_scale[1]
            trans_mat[2, 2] = mesh_scale[2]
            mesh_scaled = mesh.apply_transform(trans_mat)

            # choose 100000 pts where half of them are occ and half are not, then apply the same scaling
            occ_in_idx = np.where(occ == 1)[0]
            occ_out_idx = np.where(occ == 0)[0]
            n_in = occ_in_idx.shape[0]

            if n_in > 50000:
                slice_in_idx = np.random.choice(occ_in_idx, 50000, replace=False)
                n_in = 50000
            else:
                slice_in_idx = occ_in_idx

            n_out = 100000 - n_in
            slice_out_idx = np.random.choice(occ_out_idx, n_out, replace=False)

            idx = np.concatenate([slice_in_idx, slice_out_idx])
            pts_label = pts[idx]
            query_pts = pts_label / 0.9 * mesh_scale  # apply scaling to the pts
            occ_pts = occ[idx]  # occ is scaling and rotation invariant

            # # visualization test
            # pcd_obj = mesh_scaled.sample(2000)
            # vis_pts = query_pts[occ_pts == 1]
            # vis_pts_out = query_pts[occ_pts == 0]
            #
            # ps.init()
            # ps.set_up_dir("z_up")
            #
            # ps.register_point_cloud("pcd", pcd_obj, radius=0.005, color=[0, 0, 1], enabled=True)
            # ps.register_point_cloud("occ", vis_pts, radius=0.005, color=[0, 1, 0], enabled=True)
            # ps.register_point_cloud("occ_out", vis_pts_out, radius=0.005, color=[1, 0, 0], enabled=True)
            #
            # ps.show()

            # calculate sdf and scf for the scaled object
            surface_point_cloud = get_surface_point_cloud(mesh_scaled, bounding_radius=1, scan_count=SCAN_COUNT,
                                                          scan_resolution=SCAN_RESOLUTION)
            print("calculating sdf ...")
            sdf_pts = surface_point_cloud.get_sdf_in_batches(query_pts, use_depth_buffer=True, return_gradients=False)

            print("calculating scf ...")
            scf_pts = batch_compute_scfs(mesh_scaled, query_pts)[..., :5]

            # save result
            np.savez(
                save_path,
                obj_file=f"{obj_id}.obj",
                mesh_scale=mesh_scale,
                object_pose_cam_frame=pcd_data["object_pose_cam_frame"],
                depth_observation=pcd_data["depth_observation"],
                object_segmentation=pcd_data["object_segmentation"],
                point_cloud=pcd_data["point_cloud"],
                table_point_cloud=pcd_data["table_point_cloud"],
                obj_pose_world=pcd_data["obj_pose_world"],
                cam_pose_world=pcd_data["cam_pose_world"],
                cam_intrinsics=pcd_data["cam_intrinsics"],
                pts=pts_label,
                occ=occ_pts,
                sdf=sdf_pts,
                scf=scf_pts
            )
            time.sleep(0.5)
            print(f"{filename} finished.")

        except Exception as e:
            print(f"{filename} failed.")


def process_model_files():
    # create save dir if not exists
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    files = os.listdir(pcd_dir)

    print(f"Models directory: {pcd_dir}")
    print(f"Model files (total: {len(files)})")

    worker_count = os.cpu_count() // 2
    # worker_count = 1
    print("Using {:d} processes.".format(worker_count))
    pool = ThreadPool(processes=worker_count)

    progress = tqdm(total=len(files))

    def on_complete(*_):
        progress.update()

    for filename in files:
        pool.apply_async(process_model_file, args=(filename, ), callback=on_complete)

    print("Start ...")
    pool.close()
    pool.join()
    print("All Done.")


if __name__ == '__main__':
    process_model_files()
