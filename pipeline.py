import os
import cv2
import re
import torch
import numpy as np
import kornia as K
import kornia.feature as KF
from SuperGluePretrainedNetwork.models.matching import Matching

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

import open3d as o3d
import pyscancontext as sc


def load_torch_image(img, device=torch.device('cpu')):

    img = K.image_to_tensor(img, False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    
    return img


def load_disk(device=torch.device('cpu')):

    disk = KF.DISK().to(device)
    pretrained_dict = torch.load('./pretrained/disk_outdoor.ckpt', map_location=device)
    disk.load_state_dict(pretrained_dict['extractor'])
    disk.eval()

    return disk


def disk_matching(query_img, ref_img, worker):

    query_tensor = load_torch_image(query_img, device=torch.device('cuda'))
    ref_tensor = load_torch_image(ref_img, device=torch.device('cuda'))

    with torch.inference_mode():

        feature1 = worker(query_tensor, 2048, pad_if_not_divisible=True)[0]
        feature2 = worker(ref_tensor, 2048, pad_if_not_divisible=True)[0]

        desc1 = feature1.descriptors
        desc2 = feature2.descriptors
        desc1 = desc1.reshape(-1, 128).detach().cpu()
        desc2 = desc2.reshape(-1, 128).detach().cpu()

        _, idxs = KF.match_smnn(desc1, desc2, 0.98)
    
    n_matches = len(idxs)

    return n_matches


def load_loftr(device=torch.device('cpu')):

    loftr = KF.LoFTR(pretrained=None)
    loftr.load_state_dict(torch.load('./pretrained/loftr_outdoor.ckpt')['state_dict'])
    loftr = loftr.to(device).eval()

    return loftr


def loftr_matching(query_img, ref_img, worker):
    
    query_tensor = K.color.rgb_to_grayscale(load_torch_image(query_img, device=torch.device('cuda')))
    ref_tensor = K.color.rgb_to_grayscale(load_torch_image(ref_img, device=torch.device('cuda')))
    input_dict = {"image0": query_tensor, "image1": ref_tensor}

    with torch.inference_mode():
        correspondences = worker(input_dict)
        mkpts0 = correspondences['keypoints0'].cpu().numpy()

    n_matches = len(mkpts0)

    return n_matches


def load_superglue(device=torch.device('cpu')):

    config = {
        'superpoint': {
            'nms_radius': 3,
            'keypoint_threshold': 0.005,
            'max_keypoints': 2048
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    s_s = Matching(config).eval().to(device)

    return s_s
    

def superglue_matching(query_img, ref_img, worker):

    query_tensor = K.color.rgb_to_grayscale(load_torch_image(query_img, device=torch.device('cuda')))
    ref_tensor = K.color.rgb_to_grayscale(load_torch_image(ref_img, device=torch.device('cuda')))

    input_dict = {"image0": query_tensor, "image1": ref_tensor}

    pred = worker(input_dict)
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches0, matches1 = pred['matches0'], pred['matches1']

    valid0 = matches0 > -1
    mkpts0 = np.int32(kpts0[valid0])
    n_matches = len(mkpts0)

    return n_matches


def sift_matching(query_img, ref_img, worker):

    kp1, des1 = worker.detectAndCompute(query_img, None)
    kp2, des2 = worker.detectAndCompute(ref_img, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    inlier_matches = [b for a, b in zip(mask, good_matches) if a]

    return len(good_matches)


def gms_matching(query_img, ref_img, worker):
    
    kp1, des1 = worker.detectAndCompute(query_img, None)
    kp2, des2 = worker.detectAndCompute(ref_img, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)

    matches_gms = cv2.xfeatures2d.matchGMS(query_img.shape[:2], ref_img.shape[:2], kp1, kp2, matches_all, withScale=False, withRotation=False, thresholdFactor=6)

    return len(matches_gms)

class LCV():

    def __init__(self, method, scm, thre , time):

        self.dataset_path = None
        self.gt_file = None
        self.method = method
        self.q = None
        self.db = None
        self.model_path = "./EE5346_dataset/robotcar-dataset-sdk-3.0/models"
        self.extrinsics_dir = "./EE5346_2023_project_main/robotcar_dataset_sdk/extrinsics"
        self.scm_state = scm
        self.thre = thre
        self.time = time

        self.scm = sc.SCManager()

        if self.method == "disk":
            self.worker = load_disk(device=torch.device('cuda'))
        elif self.method == "superglue":
            self.worker = load_superglue(device=torch.device('cuda'))
        elif self.method == "loftr":
            self.worker = load_loftr(device=torch.device('cuda'))
        elif self.method == "sift":
            self.worker = cv2.SIFT_create()
        elif self.method == "gms":
            self.worker = cv2.ORB_create(10000)
            self.worker.setFastThreshold(0)


    def re_init(self, dataset_path, gt_file):
        
        self.gt_file = gt_file
        q_match = re.search(r"_q(\w+)_", self.gt_file)
        db_match = re.search(r"_db(\w+)_", self.gt_file)

        if q_match and db_match:
            self.q = q_match.group(1).split("_")[0]
            self.db = db_match.group(1).split("_")[0]

        self.query_path = dataset_path + self.q + "_val/"
        self.ref_path = dataset_path + self.db + "_val/"
        

        self.camera_model = CameraModel(self.model_path, self.query_path + "stereo/centre/")
        self.extrinsics_path = os.path.join(self.extrinsics_dir, self.camera_model.camera + '.txt')
        with open(self.extrinsics_path) as extrinsics_file:
            self.extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
        self.G_camera_posesource = build_se3_transform(self.extrinsics)

        os.makedirs(f"./output/{self.method}", exist_ok=True)
        
        os.makedirs(f"./output/{self.method}_{self.thre}_{self.time}", exist_ok=True)

        if self.scm_state == True:
            os.makedirs(f"./output/scm", exist_ok=True)
        print(f"The target file is {self.gt_file}.")
        with open(dataset_path + self.gt_file, 'r') as f:
            lines = f.readlines()

        self.lcd_results = [tuple(line.strip().split(', ')) for line in lines]

        self.query_laser_dir = os.path.join(self.query_path + "ldmrs")
        self.ref_laser_dir = os.path.join(self.ref_path + "ldmrs")

        self.query_poses_file = os.path.join(self.query_path + "vo/vo.csv")
        self.ref_poses_file = os.path.join(self.ref_path + "vo/vo.csv")
        
        self.query_timestamps = []
        self.ref_timestamps = []
        with open(os.path.join(self.query_path, "stereo.timestamps")) as query_files:
            for line in query_files:
                timestamp = int(line.split(' ')[0])
                self.query_timestamps.append(timestamp)

        with open(os.path.join(self.ref_path, "stereo.timestamps")) as ref_files:
            for line in ref_files:
                timestamp = int(line.split(' ')[0])
                self.ref_timestamps.append(timestamp)


    def bin2scd(self, pointcloud, voxel_size=0.75):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        xyz_down = np.asarray(pcd_down.points)
        scd = self.scm.make_scancontext(xyz_down)

        return scd 

    def scancontext(self, query_time, ref_time):

        pc_1, _ = build_pointcloud(self.query_laser_dir, self.query_poses_file, self.extrinsics_dir,
                                query_time - 1e7, query_time + 1e7, query_time)
        pc_2, _ = build_pointcloud(self.ref_laser_dir, self.ref_poses_file, self.extrinsics_dir,
                                ref_time - 1e7, ref_time + 1e7, ref_time)
        pc_1 = np.dot(self.G_camera_posesource, pc_1)
        pc_2 = np.dot(self.G_camera_posesource, pc_2)
        pc_1 = pc_1.T[:, :-1]
        pc_2 = pc_2.T[:, :-1]
        scd_1 = self.bin2scd(pc_1)
        scd_2 = self.bin2scd(pc_2)

        distance, argmin_rot_idx = self.scm.scd_distance(scd_1, scd_2)

        return distance

    def temporal_consistency(self, query_time, ref_time):
        query_back = None
        ref_back = None

        for i in range(len(self.query_timestamps)):
            if query_time == self.query_timestamps[i]:
                # query_back = self.query_timestamps[i+self.time]
                query_back = self.query_timestamps[i]
                break

        for j in range(len(self.ref_timestamps)):
            if ref_time == self.ref_timestamps[j]:
                ref_back = self.ref_timestamps[j+self.time]
                break

        return query_back, ref_back


    def verification(self):
  
        with open(f"./output/scm/{self.gt_file}", "a") as file:
        # with open(f"./output/{self.method}/{self.gt_file}", "a") as file:
            for query_img_path, ref_img_path, gt in self.lcd_results:
                
                query_img = cv2.imread(self.query_path + "stereo/centre/"+ query_img_path.split("/")[-1])
                ref_img = cv2.imread(self.ref_path + "stereo/centre/"+ ref_img_path.split("/")[-1])

                if self.scm_state == True:
                    match = re.search(r"\d+", query_img_path)
                    query_time = int(match.group(0))
                    match = re.search(r"\d+", ref_img_path)
                    ref_time = int(match.group(0))
                    distance = self.scancontext(query_time, ref_time)
                    confidence = int(1/distance*100)
                    file.write(f"{query_img_path}, {ref_img_path}, {gt}, {confidence}\n")

                # if self.method == "sift":
                #     n_matches = sift_matching(query_img, ref_img, self.worker)
                # elif self.method == "disk":
                #     n_matches = disk_matching(query_img, ref_img, self.worker)
                # elif self.method == "loftr":
                #     n_matches = loftr_matching(query_img, ref_img, self.worker)
                # elif self.method == "superglue":
                #     n_matches = superglue_matching(query_img, ref_img, self.worker)
                # elif self.method == "gms":
                #     n_matches = gms_matching(query_img, ref_img, self.worker)

                # file.write(f"{query_img_path}, {ref_img_path}, {gt}, {n_matches}\n")

    def verification_two_stage(self):

        with open(f"./output/{self.method}_{self.thre}_{self.time}/{self.gt_file}", "a") as file:
            # for query_img_path, ref_img_path, gt in self.lcd_results:
            for query_img_path, ref_img_path in self.lcd_results:    
                query_img = cv2.imread(self.query_path + "stereo/centre/"+ query_img_path.split("/")[-1])
                ref_img = cv2.imread(self.ref_path + "stereo/centre/"+ ref_img_path.split("/")[-1])

                if self.method == "sift":
                    n_matches = sift_matching(query_img, ref_img, self.worker)
                elif self.method == "disk":
                    n_matches = disk_matching(query_img, ref_img, self.worker)
                elif self.method == "loftr":
                    n_matches = loftr_matching(query_img, ref_img, self.worker)
                elif self.method == "superglue":
                    n_matches = superglue_matching(query_img, ref_img, self.worker)

                if n_matches > self.thre:
                    match = re.search(r"\d+", query_img_path)
                    query_time = int(match.group(0))
                    match = re.search(r"\d+", ref_img_path)
                    ref_time = int(match.group(0))
                    if self.scm_state == True:
                        distance = self.scancontext(query_time, ref_time)
                        confidence = int(1/distance*100)
                    else:
                        query_back, ref_back = self.temporal_consistency(query_time, ref_time)
                        query_img = cv2.imread(self.query_path + "stereo/centre/" + str(query_back) + ".jpg")
                        ref_img = cv2.imread(self.ref_path + "stereo/centre/"+ str(ref_back) + ".jpg")
                        if self.method == "sift":
                            confidence = sift_matching(query_img, ref_img, self.worker)
                        elif self.method == "disk":
                            confidence = disk_matching(query_img, ref_img, self.worker)
                        elif self.method == "loftr":
                            confidence = loftr_matching(query_img, ref_img, self.worker)
                        elif self.method == "superglue":
                            confidence = superglue_matching(query_img, ref_img, self.worker) 
                else:
                    confidence = 0

                if confidence > 364:
                    results = 1
                else:
                    results = 0
                file.write(f"{query_img_path}, {ref_img_path}, {results}\n")

def exp_for_LCV(method = "sift"):

    dataset_paths = ['EE5346_dataset/']
    lcv = LCV(method, scm = False, thre = 197, time = 20)
    
    
    for index, dataset_path in enumerate(dataset_paths):
        gt_files = [file for file in os.listdir(dataset_paths[index]) if file.endswith(".txt")]
        for gt_file in gt_files:
            # if gt_file == "robotcar_qAutumn_dbNight_diff_final.txt" or gt_file == "robotcar_qAutumn_dbNight_easy_final.txt":
            # if gt_file != "robotcar_qAutumn_dbNight_diff_final.txt" and gt_file != "robotcar_qAutumn_dbNight_easy_final.txt":
            if gt_file == "robotcar_qAutumn_dbSuncloud_val_final.txt":
                lcv.re_init(dataset_path, gt_file)
                # lcv.verification()
                lcv.verification_two_stage()


if __name__ == '__main__':

    method = "superglue"
    print(method)
    print("----------------------------------------------------------------------------")
    exp_for_LCV(method)


    
