import os
import pickle

import numpy as np
import sys
import os.path as osp


# base_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/sdf"
# obj_type = "hammer"
#
# label_path = osp.join(base_path, obj_type, "cloud")
# data_list = os.listdir(label_path)

# save_dict = {}
# for data_id in data_list:
#     data_path = osp.join(label_path, data_id)
#     data = np.load(data_path, allow_pickle=True)
#     label = data["coords_sdf"]
#     norm_factor = data["norm_factor"]
#
#     obj_id = data_id.split('.')[0]
#     tup = (label[:, :3], label[:, 4], label[:, 5:], norm_factor)
#
#     save_dict[obj_id] = tup
#
# with open('hammer_sdf_scf.p', 'wb') as f:
#     pickle.dump(save_dict, f)
# f.close()

# label_dict = pickle.load(open('hammer_sdf_scf.p', 'rb'))

# ---------------------------------------------------------------------------------------------------------- #
label_dir = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/labels/container"
file_list = os.listdir(label_dir)
obj_type = "container"

# norm_path = "/home/ikun/master-thesis/relational_ndf/src/rndf_robot/data/training_data_non_shapenet/container_occ.p"
# norm_label = np.load(norm_path, allow_pickle=True)

save_dict = {}
for file in file_list:
    obj_id = file.split('.')[0]
    label_path = osp.join(label_dir, file)

    # for rndf provided point cloud object
    # _, _, norm_factor = norm_label[f"container_{obj_id}"]
    data = np.load(label_path, allow_pickle=True)
    label = data["label"]
    pts = label[:, :3]
    occ = label[:, 3]
    sdf = label[:, 4]
    scf = label[:, 5:]
    norm_factor = data["norm_factor"]
    tup = (pts, occ, sdf, scf, norm_factor)

    save_dict[f"{obj_type}_{obj_id}"] = tup

with open(osp.join("/home/ikun/master-thesis/relational_ndf/src/rndf_robot/descriptions/objects/labels", f'{obj_type}_occ_sdf_scf.p'), 'wb') as f:
    pickle.dump(save_dict, f)
f.close()
print("All saved.")
