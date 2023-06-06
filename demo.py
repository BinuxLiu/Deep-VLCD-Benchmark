import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

import open3d as o3d
import pyscancontext as sc

parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
parser.add_argument('--image_dir', type=str, help='Directory containing images')
parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
parser.add_argument('--image_idx', type=int, help='Index of image to display')

args = parser.parse_args()

model = CameraModel(args.models_dir, args.image_dir)

extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
with open(extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = None

poses_type = re.search('(vo|ins|rtk)\.csv', args.poses_file).group(1)
if poses_type in ['ins', 'rtk']:
    with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
else:
    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle


timestamps_path = os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps')
if not os.path.isfile(timestamps_path):
    timestamps_path = os.path.join(args.image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

timestamp = 0
with open(timestamps_path) as timestamps_file:
    for i, line in enumerate(timestamps_file):
        if i == args.image_idx:
            timestamp = int(line.split(' ')[0])

# with open(self.dataset_path + self.gt_file, 'r') as f:
#             lines = f.readlines()
#         self.lcd_results = [tuple(line.strip().split(', ')) for line in lines]

timestamp_1 = int(1418133739556542)
timestamp_2 = int(1418756773603118)

query = "Autumn_val"
reference = "Night_val"

query_laser_dir = os.path.join("/mnt/sda3/Projects/Deep-VLCD-Benchmark/EE5346_dataset", query, "ldmrs")
reference_laser_dir = os.path.join("/mnt/sda3/Projects/Deep-VLCD-Benchmark/EE5346_dataset", reference, "ldmrs")

query_poses_file = "/mnt/sda3/Projects/Deep-VLCD-Benchmark/EE5346_dataset/Autumn_val/vo/vo.csv"
reference_poses_file = "/mnt/sda3/Projects/Deep-VLCD-Benchmark/EE5346_dataset/Night_val/vo/vo.csv"


# pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file, args.extrinsics_dir,
#                                            timestamp - 1e7, timestamp + 1e7, timestamp)


pointcloud_1, _ = build_pointcloud(query_laser_dir, query_poses_file, args.extrinsics_dir,
                                           timestamp_1 - 1e7, timestamp_1 + 1e7, timestamp_1)

pointcloud_1 = np.dot(G_camera_posesource, pointcloud_1)

image_path_1 = os.path.join("/mnt/sda3/Projects/Deep-VLCD-Benchmark/EE5346_dataset/Autumn_val/stereo/centre", str(timestamp_1) + '.jpg')
image_1 = load_image(image_path_1, model)

uv_1, depth_1 = model.project(pointcloud_1, image_1.shape)


pointcloud_2, _ = build_pointcloud(reference_laser_dir, reference_poses_file, args.extrinsics_dir,
                                           timestamp_2 - 1e7, timestamp_2 + 1e7, timestamp_2)

pointcloud_2 = np.dot(G_camera_posesource, pointcloud_2)

image_path_2 = os.path.join("/mnt/sda3/Projects/Deep-VLCD-Benchmark/EE5346_dataset/Night_val/stereo/centre", str(timestamp_2) + '.jpg')
image_2 = load_image(image_path_2, model)

uv_2, depth_2 = model.project(pointcloud_2, image_2.shape)

scm = sc.SCManager()

def bin2scd(pointcloud, voxel_size=0.75):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz_down = np.asarray(pcd_down.points)
    scd = scm.make_scancontext(xyz_down)
    return scd 

pointcloud_1 = pointcloud_1.T[:, :-1]
pointcloud_2 = pointcloud_2.T[:, :-1]

scd_1 = bin2scd(pointcloud_1)
scd_2 = bin2scd(pointcloud_2)

distance, argmin_rot_idx = scm.scd_distance(scd_1, scd_2)

print(distance)

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'black'

plt.subplot(1, 2, 1)
plt.imshow(image_1)
plt.scatter(np.ravel(uv_1[0, :]), np.ravel(uv_1[1, :]), s=2, c=depth_1, edgecolors='none', cmap='jet')
plt.xlim(0, image_1.shape[1])
plt.ylim(image_1.shape[0], 0)
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(image_2)
plt.scatter(np.ravel(uv_2[0, :]), np.ravel(uv_2[1, :]), s=2, c=depth_2, edgecolors='none', cmap='jet')
plt.xlim(0, image_2.shape[1])
plt.ylim(image_2.shape[0], 0)
plt.xticks([])
plt.yticks([])

plt.show()