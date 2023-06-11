import copy
import os
import sys

import torch
import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import polyscope as ps
import numpy as np

from rndf_robot.utils import torch_util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))


opt_iterations = 500
n_pts = 2000

# pcd_dir = get_demo_place_dir(obj="y_cups", file_name="0/0.npz")
pcd_dir = os.path.join(BASE_DIR, 'test_data', "y_cups", "t1/0.npz")
data = np.load(pcd_dir, allow_pickle=True)
pcd1 = data["pcd_local"]
mean_1 = np.mean(pcd1, axis=0)
pcd1 = pcd1 - mean_1

pcd_inf = pcd1
pcd_inf_mean = mean_1
np.random.shuffle(pcd_inf)

pcd_dir = os.path.join(BASE_DIR, 'test_data', "t_cups", "t1/0.npz")
data = np.load(pcd_dir, allow_pickle=True)
pcd2 = data["pcd_local"]
obj_pos = data["obj_pos"]
dis_y = obj_pos[1]
dis_z = obj_pos[2] - 0.104
mean_2 = np.mean(pcd2, axis=0)
pcd2 = pcd2 - mean_2

pcd_ref = pcd2
pcd_ref_mean = mean_2
np.random.shuffle(pcd_ref)

# mesh_dir = get_object_mesh_path(obj="white_cups")
# mesh_dir = mesh_dir + '/scaled.obj'
# mesh = trimesh.load(mesh_dir, process=False, force='mesh')
# pcd_sample = mesh.sample(5000)
# mean_sample = np.mean(pcd_sample, axis=0)
# pcd_sample = pcd_sample - mean_sample

# pcd_ref = pcd_sample
# pcd_ref_mean = mean_sample
# np.random.shuffle(pcd_ref)

t_ref = np.array([[1, 0, 0, 0],
                  [0, -1, 0, 1.1],
                  [0, 0, -1, 0.966],
                  [0, 0, 0, 1]])

t_ref_vis = np.array([[1, 0, 0, 0 - pcd_ref_mean[0]],
                      [0, -1, 0, 1.1 - pcd_ref_mean[1]],
                      [0, 0, -1, 0.966 - pcd_ref_mean[2]],
                      [0, 0, 0, 1]])

n_pts_gripper = 500
# radius = 0.08
# height = 0.04
# u_th = np.random.rand(n_pts_gripper, 1)
# u_r = np.random.rand(n_pts_gripper, 1)
# x = radius * np.sqrt(u_r) * np.cos(2 * np.pi * u_th)
# y = radius * np.sqrt(u_r) * np.sin(2 * np.pi * u_th) + dis_y
# z = np.random.rand(n_pts_gripper, 1) * height + dis_z
n = n_pts_gripper
x = np.random.uniform(-0.02, 0.02, n)
y = np.random.uniform(-0.02, 0.02, n)
z = np.random.uniform(-0.06, 0.06, n)
ref_query_pts = np.vstack([x, y, z])
ref_pts_gripper = ref_query_pts.T

ones = np.ones(n)
hom_query_pts = np.vstack([x, y, z, ones])

# transform
ref_query_pts = t_ref @ hom_query_pts
ref_query_pts = ref_query_pts[:3, :] - pcd_ref_mean[:, None]
ref_query_pts = ref_query_pts.T
query_pts_vis = copy.deepcopy(ref_query_pts)

coords = np.array([[0.1, 0, 0, 0],
                   [0., 0.1, 0, 0],
                   [0, 0, 0.1, 0],
                   [1, 1, 1, 1]])
coords = t_ref_vis @ coords
coords = coords[0:3, :]
coords = coords.T
nodes = coords

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("pcd_ref", pcd_ref, radius=0.005, enabled=True)
ps.register_point_cloud("ref_query", ref_query_pts, radius=0.005, enabled=True)
ps.register_point_cloud("pcd_inf", pcd1, radius=0.005, enabled=True)
ps.register_curve_network("edge_x_ref", nodes[[0, 3]], np.array([[0, 1]]), enabled=True, radius=0.002, color=(1, 0, 0))
ps.register_curve_network("edge_y_ref", nodes[[1, 3]], np.array([[0, 1]]), enabled=True, radius=0.002, color=(0, 1, 0))
ps.register_curve_network("edge_z_ref", nodes[[2, 3]], np.array([[0, 1]]), enabled=True, radius=0.002, color=(0, 0, 1))
# ps.register_point_cloud("pcd_sample", pcd_sample, radius=0.005, enabled=True)
ps.show()
# load model and weights
weight_path = '/home/ikun/master-thesis/relational_ndf/src/rndf_robot/model_weights/ndf_vnn/rndf_weights/ndf_mug.pth'
model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True,
                                        sigmoid=True).cuda()
model.load_state_dict(torch.load(weight_path))

# device
dev = torch.device('cuda:0')

# loss
loss_fn = torch.nn.L1Loss()

# start optimization
# get descriptor of the ref grasp of the ref point cloud
reference_model_input = {}
ref_query_pts = torch.from_numpy(ref_query_pts).float().to(dev)
ref_shape_pcd = torch.from_numpy(pcd_ref[:n_pts]).float().to(dev)
reference_model_input['coords'] = ref_query_pts[None, :, :]
reference_model_input['point_cloud'] = ref_shape_pcd[None, :, :]  # get the descriptors for these reference query points
reference_latent = model.extract_latent(reference_model_input).detach()
reference_act_hat = model.forward_latent(reference_latent, reference_model_input['coords']).detach()

best_loss = np.inf
best_tf = np.eye(4)
best_idx = 0
tf_list = []
M = 10

# parameters for optimization
trans = (torch.rand((M, 3)) * 0.1).float().to(dev)
rot = torch.rand(M, 3).float().to(dev)
trans.requires_grad_()
rot.requires_grad_()
opt = torch.optim.Adam([trans, rot], lr=1e-2)

# initialization
rand_rot_init = (torch.rand((M, 3)) * 2 * np.pi).float().to(dev)
rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
rand_mat_init = rand_mat_init.squeeze().float().to(dev)

# now randomly initialize a copy of the query points
opt_query_pts = torch.from_numpy(ref_pts_gripper).float().to(dev)
opt_query_pts = opt_query_pts[None, :, :].repeat((M, 1, 1))
X = torch_util.transform_pcd_torch(opt_query_pts, rand_mat_init)

opt_model_input = {}
opt_model_input['coords'] = X

mi_point_cloud = []
for ii in range(M):
    # mi_point_cloud.append(torch.from_numpy(self.pcd2[:self.n_pts]).float().to(self.dev))
    if ii % 2 == 0:
        np.random.shuffle(pcd_inf)
    mi_point_cloud.append(torch.from_numpy(pcd_inf[:n_pts]).float().to(dev))
mi_point_cloud = torch.stack(mi_point_cloud, 0)
opt_model_input['point_cloud'] = mi_point_cloud
opt_latent = model.extract_latent(opt_model_input).detach()

loss_values = []
vid_plot_idx = None

for i in range(opt_iterations):
    T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()
    # trans_ = trans - (torch.from_numpy(pcd_inf_mean[None, :])).float().to(dev)
    # X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))
    X_new = torch_util.transform_pcd_torch(X, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

    act_hat = model.forward_latent(opt_latent, X_new)
    t_size = reference_act_hat.size()

    losses = [loss_fn(act_hat[ii].view(t_size), reference_act_hat) for ii in range(M)]
    loss = torch.mean(torch.stack(losses))
    if i % 100 == 0:
        losses_str = ['%f' % val.item() for val in losses]
        loss_str = ', '.join(losses_str)
        print(f'i: {i}, losses: {loss_str}')
    loss_values.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()

best_idx = torch.argmin(torch.stack(losses)).item()
best_loss = losses[best_idx]
print('best loss: %f, best_idx: %d' % (best_loss, best_idx))

best_X = X_new[best_idx].detach().cpu().numpy()
grasp_obj = T_mat[best_idx].detach().cpu().numpy()
grasp_trans = trans[best_idx].detach().cpu().numpy()
rand_mat_init_ = rand_mat_init[best_idx].detach().cpu().numpy()

grasp_obj[:3, -1] = grasp_trans
t_opt = grasp_obj @ rand_mat_init_

print(t_opt)
print(t_ref_vis)

coords = np.array([[0.1, 0, 0, 0],
                   [0., 0.1, 0, 0],
                   [0, 0, 0.1, 0],
                   [1, 1, 1, 1]])
coords = t_opt @ coords
coords = coords[0:3, :]
coords = coords.T
nodes = coords
# edges = np.array([[0, 3],
#                   [1, 3],
#                   [2, 3]])

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("pcd_ref", pcd_ref[:2000] - np.array([0.2, 0, 0]), radius=0.005, enabled=True)
ps.register_point_cloud("pcd_inf", pcd_inf[:2000], radius=0.005, enabled=True)
ps.register_point_cloud("ref_query", query_pts_vis - np.array([0.2, 0, 0]), radius=0.005, enabled=True)
ps.register_point_cloud("opt_g", best_X, radius=0.005, enabled=True)
ps_x = ps.register_curve_network("edge_x", nodes[[0, 3]], np.array([[0, 1]]),
                                 enabled=True, radius=0.002, color=(1, 0, 0))
ps_y = ps.register_curve_network("edge_y", nodes[[1, 3]], np.array([[0, 1]]),
                                 enabled=True, radius=0.002, color=(0, 1, 0))
ps_z = ps.register_curve_network("edge_z", nodes[[2, 3]], np.array([[0, 1]]),
                                 enabled=True, radius=0.002, color=(0, 0, 1))
coords = np.array([[0.1, 0, 0, 0],
                   [0., 0.1, 0, 0],
                   [0, 0, 0.1, 0],
                   [1, 1, 1, 1]])
coords = t_ref_vis @ coords
coords = coords[0:3, :]
coords = coords.T
nodes = coords - np.array([0.2, 0, 0])
ps.register_curve_network("edge_x_ref", nodes[[0, 3]], np.array([[0, 1]]), enabled=True, radius=0.002, color=(1, 0, 0))
ps.register_curve_network("edge_y_ref", nodes[[1, 3]], np.array([[0, 1]]), enabled=True, radius=0.002, color=(0, 1, 0))
ps.register_curve_network("edge_z_ref", nodes[[2, 3]], np.array([[0, 1]]), enabled=True, radius=0.002, color=(0, 0, 1))
ps.show()
