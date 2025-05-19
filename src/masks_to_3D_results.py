import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import copy
import multiprocessing as mp
import random
import argparse
import json
import pickle
import math
import pycocotools.mask
from pyquaternion import Quaternion as Quaternion
from PIL import Image
from os.path import join
import sys
from shapely.geometry import Point, box
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import Quaternion as nQuaternion
from utils.pcd import LidarPointCloud, view_points
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import mini_val, mini_train, train_detect, train, val
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import scipy
import time
import tqdm
import numba
from numba import cuda
import warnings
warnings.filterwarnings("ignore")

VER_NAME = "v1.0-trainval"
INPUT_PATH = "data/nuscenes/"

OUTPUT_DIR = "outputs/"
INPUT_DIR = "outputs/nuscenes/results_mask/nuscenes-gd-sam/"

CAM_LIST = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

step_xi = 0.5
step_yi = 0.5
step_zi = 0.5

custom_vocabulary = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                    'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility',
                    'pushable_pullable', 'debris', 'traffic_cone', 'barrier'] 
GD_PATH = "outputs/nuscenes/results_2D/result_2D_val.json"
with open(GD_PATH, "r") as f:
    results_GD = json.load(f)
results_GD_new = dict()
for result_GD in results_GD:
    file_name = result_GD['file_name']
    x1, y1, w, h = result_GD['bbox']
    x2, y2 = x1 + w, y1 + h    
    if file_name in results_GD_new:
        results_GD_new[file_name]['boxes_filt_list'].append([x1, y1, x2, y2])
        results_GD_new[file_name]['pred_phrases'].append(f"{custom_vocabulary[result_GD['category_id']-1]}({round(result_GD['score'], 2)})")
    else:
        results_GD_new[file_name] = dict()
        results_GD_new[file_name]['boxes_filt_list'] = list()
        results_GD_new[file_name]['pred_phrases'] = list()

@cuda.jit
def center2corner_3d_gpu(xi, yi, zi, xyz, ri, vertices):
    # 对每个顶点进行计算
    for i in range(8):
        x = xyz[0, i]
        y = xyz[1, i]
        z = xyz[2, i]
        
        # 应用旋转矩阵绕z轴旋转
        cos_r = math.cos(ri)
        sin_r = math.sin(ri)
        
        # 旋转后的坐标
        vertices[i, 0] = xi + x * cos_r - y * sin_r
        vertices[i, 1] = yi + x * sin_r + y * cos_r
        vertices[i, 2] = zi + z

@cuda.jit(device=True)
def check_point_num_in_box_fast_v2_gpu(obj_lidar_pts, temp_box):
    total_num = obj_lidar_pts.shape[0]
    if total_num == 0:
        return 0
    in_box_num = 0
    min_x = min(temp_box[:, 0])
    max_x = max(temp_box[:, 0])
    min_y = min(temp_box[:, 1])
    max_y = max(temp_box[:, 1])
    for k, pt in enumerate(obj_lidar_pts):
        if (pt[0] < min_x or pt[0]  > max_x or pt[1] < min_y or pt[1] > max_y):
            continue

        in_flag = False
        j = 3
        for i in range(4):
            if ((temp_box[i, 1] > pt[1]) != (temp_box[j, 1] > pt[1])) and (
                pt[0] < (temp_box[j, 0] - temp_box[i, 0]) * (pt[1] - temp_box[i, 1]) / 
                (temp_box[j, 1] - temp_box[i, 1]) + temp_box[i, 0]
            ):
                in_flag = not in_flag
            j = i
        if in_flag:
            in_box_num = in_box_num + 1 
    
    return in_box_num / total_num

@cuda.jit
def find_best_pts_ratio_box_gpu(obj_lidar_pts, xyz, num_xyz, min_max_values, candidate_boxes, in_box_ratios, yaw_array):
    idx = cuda.grid(1)

    num_xi, num_yi, num_zi, num_ri = num_xyz
    if idx < num_xi * num_yi * num_zi * num_ri:
        min_x, max_x, min_y, max_y, min_z, max_z = min_max_values    
        xi_idx = idx // (num_yi * num_zi * num_ri) % num_xi
        yi_idx = idx // (num_zi * num_ri) % num_yi
        zi_idx = idx // num_ri % num_zi
        ri_idx = int(idx % num_ri)

        xi = min_x + xi_idx * step_xi
        yi = min_y + yi_idx * step_yi
        zi = min_z + zi_idx * step_zi
        ri = yaw_array[ri_idx]
            
        temp_box = cuda.local.array((8, 3), dtype=numba.float32)          
        center2corner_3d_gpu(xi, yi, zi, xyz, ri, temp_box)
        in_box_ratio = check_point_num_in_box_fast_v2_gpu(obj_lidar_pts, temp_box)     
        
        in_box_ratios[idx] = in_box_ratio
        for i in range(8):
            for j in range(3):
                candidate_boxes[idx, i, j] = temp_box[i, j] 

@cuda.jit
def count_max_elements_kernel(d_ratio, d_max_ratio, counter):
    tid = cuda.grid(1)
    if tid < d_ratio.size and d_ratio[tid] == d_max_ratio:
        cuda.atomic.add(counter, 0, 1)

@cuda.jit
def find_all_max_and_indices_kernel(d_ratio, d_max_ratio, indices):
    tid = cuda.grid(1)
    if tid < d_ratio.size and d_ratio[tid] == d_max_ratio:
        idx = cuda.atomic.add(indices, 0, 1) 
        indices[idx + 1] = tid

def find_best_pts_ratio_box_m(obj_lidar_pts, center, box_size, yaw_list):
    # 主机端预计算 min/max 值
    min_x = center[0] - box_size[1]/2
    max_x = center[0] + box_size[1]/2
    min_y = center[1] - box_size[1]/2
    max_y = center[1] + box_size[1]/2
    min_z = center[2] - 1
    max_z = center[2] + 1

    num_xi = int((max_x - min_x) / step_xi) + 1
    num_yi = int((max_y - min_y) / step_yi) + 1
    num_zi = int((max_z - min_z) / step_zi) + 1
    num_ri = 2
    num_xyz = np.array([num_xi, num_yi, num_zi, num_ri], dtype=np.float32)

    w, l, h = box_size[0], box_size[1], box_size[2]
    x_list = [-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2]
    y_list = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_list = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]
    xyz = np.array([x_list, y_list, z_list], dtype=np.float32)

    # 预分配候选框缓冲区
    total_combinations = num_xi * num_yi * num_zi * num_ri
    max_candidate_boxes = total_combinations
    candidate_boxes = np.zeros((max_candidate_boxes, 8, 3), dtype=np.float32)
    # candidate_boxes = np.zeros((max_candidate_boxes, 7), dtype=np.float32)


    # 将数据拷贝到GPU
    d_obj_lidar_pts = cuda.to_device(obj_lidar_pts[:, :3].astype(np.float32))  # 仅需前3列
    d_candidate_boxes = cuda.to_device(candidate_boxes)
    d_ratios = cuda.to_device(np.zeros(total_combinations, dtype=np.float32)) 
    d_yaw_array = cuda.to_device(np.array(yaw_list, dtype=np.float32))
    d_xyz = cuda.to_device(np.array(xyz, dtype=np.float32))


    threads_per_block = 256
    blocks_per_grid = (total_combinations + threads_per_block - 1) // threads_per_block
    find_best_pts_ratio_box_gpu[blocks_per_grid, threads_per_block](
        d_obj_lidar_pts,
        d_xyz,
        num_xyz,
        np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.float32),
        d_candidate_boxes,
        d_ratios,
        d_yaw_array
    )

    counter = np.zeros(1, dtype=np.int32)
    d_max_ratio = max(d_ratios.copy_to_host())
    count_max_elements_kernel[blocks_per_grid, threads_per_block](d_ratios, d_max_ratio, counter)
    num_matches = counter[0]

    indices = np.zeros(num_matches + 1, dtype=np.int32)
    find_all_max_and_indices_kernel[blocks_per_grid, threads_per_block](d_ratios, d_max_ratio, indices)
    candidate_boxes = d_candidate_boxes.copy_to_host()[indices[1:]] 
    return candidate_boxes, d_max_ratio

def box_properties(points):
    center = np.mean(points, axis=0)
    
    length = np.linalg.norm(points[0] - points[2])
    width = np.linalg.norm(points[0] - points[1])
    height = np.linalg.norm(points[0] - points[4])

    v = points[1] - points[0]
    yaw = np.arctan2(v[1], v[0]) + np.pi / 2
    
    return center[0], center[1], center[2], width, length, height, yaw

@numba.jit(nopython=True)
def compute_iou(rec1, rec2):
    rec1 = (rec1[1], rec1[0], rec1[3], rec1[2])
    rec2 = (rec2[1], rec2[0], rec2[3], rec2[2])        
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
    return (intersect / (sum_area - intersect)) * 1.0 

@numba.jit(nopython=True)
def concatenate_columns(a, b):
    # 确保输入数组a和b具有相同的行数
    if a.shape[0] != b.shape[0]:
        raise ValueError("Both arrays must have the same number of rows.")
    
    # 创建一个新的数组用于存放结果，列数是a和b的列数之和
    result = np.empty((a.shape[0], a.shape[1] + b.shape[1]), dtype=a.dtype)
    
    # 将a复制到result的前部
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[i, j] = a[i, j]
            
    # 将b复制到result的后部
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            result[i, a.shape[1] + j] = b[i, j]
    
    return result

@numba.jit(nopython=True)
def world2img(points_world, pose_rec_translation, pose_rec_rotation_matrix, 
              cs_rec_translation, cs_rec_rotation_matrix, cam_intrinsic):
    points_world_homogeneous = concatenate_columns(points_world, np.ones((points_world.shape[0], 1), dtype=points_world.dtype))
    lidar2world = np.eye(4, dtype=points_world.dtype)
    lidar2world[:3, :3] = pose_rec_rotation_matrix
    lidar2world[:3, 3] = pose_rec_translation
    world2lidar = np.linalg.inv(lidar2world)
    points_lidar_homogeneous = points_world_homogeneous @ world2lidar.T
    points_lidar = points_lidar_homogeneous[:, :3] 

    camera2lidar = np.eye(4, dtype=points_lidar.dtype)
    camera2lidar[:3, :3] = cs_rec_rotation_matrix
    camera2lidar[:3, 3] = cs_rec_translation

    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]  
        
    valid = np.ones((points_camera.shape[0]), dtype=np.bool_)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)

    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = cam_intrinsic.astype(np.float32)
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid

@numba.jit(nopython=True)
def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid,
                           np.logical_and(points[:, 0] < width,
                                          points[:, 1] < height))
    return valid

@numba.jit(nopython=True)
def check_2d_iou(box_3d, pose_rec_translation, pose_rec_rotation_matrix, 
                 cs_rec_translation, cs_rec_rotation_matrix, cam_intrinsic, pred_box):
    h, w = 900, 1600
    '''Project 3d prediction Box to image plane getting 2D gt Box'''
    corners_img_temp_box, valid_z_temp_box = world2img(box_3d, pose_rec_translation, pose_rec_rotation_matrix, 
                                                       cs_rec_translation, cs_rec_rotation_matrix, cam_intrinsic)
    valid_shape_temp_box = check_point_in_img(corners_img_temp_box, h, w)
    valid_all_temp_box = np.logical_and(valid_z_temp_box, valid_shape_temp_box)
    valid_shape_temp_box = valid_shape_temp_box.reshape(-1, 8)
    valid_all_temp_box = valid_all_temp_box.reshape(-1, 8)

    if valid_z_temp_box.sum() >= 1:
        min_col = max(min(corners_img_temp_box[valid_z_temp_box, 0].min(), w), 0)
        max_col = max(min(corners_img_temp_box[valid_z_temp_box, 0].max(), w), 0)
        min_row = max(min(corners_img_temp_box[valid_z_temp_box, 1].min(), h), 0)
        max_row = max(min(corners_img_temp_box[valid_z_temp_box, 1].max(), h), 0)        
        if (max_col - min_col) == 0 or (max_row - min_row) == 0:
            return -1            
    
        temp_box_2D = [int(min_col), int(min_row), int(max_col), int(max_row)]
        iou = compute_iou(temp_box_2D, pred_box)
        return iou
    return -1  

@numba.jit(nopython=True)
def find_best_box(candidate_boxes, pose_rec_translation, pose_rec_rotation_matrix, 
                  cs_rec_translation, cs_rec_rotation_matrix, cam_intrinsic, input_boxes):
    iou_thresh = 0
    max_iou = -1
    for box_2d in input_boxes:
        for box_3d in candidate_boxes:
            iou = check_2d_iou(box_3d, pose_rec_translation, pose_rec_rotation_matrix, 
                               cs_rec_translation, cs_rec_rotation_matrix, cam_intrinsic, box_2d)
            if iou > iou_thresh:
                if iou > max_iou:
                    max_iou = iou
                    best_box = box_3d
    return best_box

def hierarchical_occupancy_score(points, box, parts=[7,5,3]):
    all_confi = 0
    for part in parts:
        all_confi += compute_confidence(points, box, part)
    return all_confi/len(parts)

def compute_confidence(points, box, parts=6):
    x, y, z, w, l, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    cloud = np.zeros(shape=(points.shape[0], 4))
    cloud[:, 0:3] = points[:, 0:3]
    cloud[:, 3] = 1

    trans_mat = np.eye(4, dtype=np.float32)
    trans_mat[0, 0] = np.cos(yaw)
    trans_mat[0, 1] = -np.sin(yaw)
    trans_mat[0, 3] = x
    trans_mat[1, 0] = np.sin(yaw)
    trans_mat[1, 1] = np.cos(yaw)
    trans_mat[1, 3] = y
    trans_mat[2, 3] = z

    trans_mat_i = np.linalg.inv(trans_mat)
    cloud = np.matmul(cloud, trans_mat_i.T)

    delta_l = l/parts
    delta_w = w/parts

    valid_vol = 0

    for i in range(parts):
        for j in range(parts):
            mask_x_l = -l/2+i*delta_l<=cloud[:, 0]
            mask_x_r = cloud[:,0]<-l/2+(i+1)*delta_l
            mask_y_l = -w/2+j*delta_w<=cloud[:, 1]
            mask_y_r = cloud[:, 1]<-w/2+(j+1)*delta_w

            mask = mask_x_l*mask_x_r*mask_y_l*mask_y_r

            this_pts = cloud[mask]

            if len(this_pts)>1:
                valid_vol+=1

    return valid_vol/(parts**2)

# utils
def count_frames(nusc, sample):
    frame_count = 1

    if sample["next"] != "":
        frame_count += 1

        # Don't want to change where sample['next'] points to since it's used later, so we'll create our own pointer
        sample_counter = nusc.get('sample', sample['next'])

        while sample_counter['next'] != '':
            frame_count += 1
            sample_counter = nusc.get('sample', sample_counter['next'])
    
    return frame_count

def get_medoid(points):
    dist_matrix = torch.cdist(points.T, points.T, p=2)

    return torch.argmin(dist_matrix.sum(axis=0))


def get_detection_name(name):
    if name not in ["trafficcone", "constructionvehicle", "human"]:
        detection_name = name
    elif name == "trafficcone":
        detection_name = "traffic_cone"
    elif name == "constructionvehicle":
        detection_name = "construction_vehicle"
    elif name == "human":
        detection_name = "pedestrian"
    
    return detection_name

def push_centroid(centroid, extents, rot_quaternion, poserecord):
    centroid = np.squeeze(centroid)
    av_centroid = poserecord["translation"]
    ego_centroid = centroid - av_centroid

    l = extents[0]
    w = extents[1]

    
    angle = R.from_quat(list(rot_quaternion)).as_euler('xyz', degrees=False)

    theta = -angle[0]

    if np.isnan(theta):
        theta = 0.5 * np.pi
    
    alpha = np.arctan(np.abs(ego_centroid[1]) / np.abs(ego_centroid[0]))

    if ego_centroid[0] < 0:
        if ego_centroid[1] < 0:
            alpha = -np.pi + alpha
        else:
            alpha = np.pi - alpha
    else:
        if ego_centroid[1] < 0:
            alpha = -alpha

    offset = np.min( [np.abs(w / (2*np.sin(theta - alpha))), np.abs(l / (2*np.cos(theta - alpha)))] )

    x_dash = centroid[0] + offset * np.cos(alpha)
    y_dash = centroid[1] + offset * np.sin(alpha)

    pushed_centroid = np.array([x_dash, y_dash, centroid[2]])

    return pushed_centroid

def get_nusc_map(nusc, scene):
    # Get scene location
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    # Get nusc map
    nusc_map = NuScenesMap(dataroot=INPUT_PATH, map_name=location)

    return nusc_map

def get_all_lane_points_in_scene(nusc_map):
    lane_records = nusc_map.lane + nusc_map.lane_connector

    lane_tokens = [lane["token"] for lane in lane_records]

    lane_pt_dict = nusc_map.discretize_lanes(lane_tokens, 0.5)

    all_lane_pts = []
    for lane_pts in lane_pt_dict.values():
        for lane_pt in lane_pts:
            all_lane_pts.append(lane_pt)
    
    return lane_pt_dict, all_lane_pts


def lane_yaws_distances_and_coords(all_centroids, all_lane_pts):
    all_lane_pts = torch.Tensor(all_lane_pts).to(device='cpu')
    all_centroids = torch.Tensor(all_centroids).to(device='cpu')

    start = time.time()

    DistMat = scipy.spatial.distance.cdist(all_centroids[:, :2], all_lane_pts[:, :2])
    
    min_lane_indices = np.argmin(DistMat, axis=1)

    distances = np.min(DistMat, axis=1)

    all_lane_pts = np.array(all_lane_pts)
    min_lanes = np.array([all_lane_pts[min_lane_indices[0]]])
    for idx in min_lane_indices:
        min_lanes = np.vstack([min_lanes, all_lane_pts[idx, :]])
    

    yaws = min_lanes[1:, 2]
    coords = min_lanes[1:, :2]

    end = time.time()

    timer['closest lane'] += end - start

    return yaws, distances, coords


# @numba.jit(nopython=True)
def circle_nms(dets, det_labels, threshs_by_label):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i]-x1[j])**2 + (y1[i]-y1[j])**2

            # ovr = inter / areas[j]
            if dist <= threshs_by_label[det_labels[j]] and det_labels[j] == det_labels[i]:
                suppressed[j] = 1
    return keep




if __name__ == "__main__":
    total_start = time.time()
    pointsensor_channel="LIDAR_TOP"
    min_dist= 2.3
    aggregation_pre_num = 1
    aggregation_next_num = 1
    use_mlo_score = True

    predictions = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": True,
            "use_external": False,
        },
        "results": {}
    }

    timer = {
        "io": 0,
        "points in mask": 0,
        "mvp": 0,
        "medoid": 0,
        "drivable": 0,
        "closest lane": 0,
        "lane pose": 0,
        "nms": 0,
        "total": 0,
    }

    nusc = NuScenes(VER_NAME, INPUT_PATH, True)

    shape_priors = json.load(open("src/cfg/shape_priors_gpt4.json"))

    progress_bar_main = tqdm.tqdm(enumerate(val))
    for scene_num, scene_name in progress_bar_main:
        progress_bar_main.set_description(f"Scene Progress:")

        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
        scene = nusc.get('scene', scene_token)
        sample = nusc.get('sample', scene['first_sample_token'])

        # Get map
        nusc_map = get_nusc_map(nusc, scene)

        drivable_records = nusc_map.drivable_area

        drivable_polygons = []
        for record in drivable_records:
            polygons = [nusc_map.extract_polygon(token) for token in record["polygon_tokens"]]
            drivable_polygons.extend(polygons)

        lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)

        all_centroids_list = []
        centroid_ids = []
        global_masked_points_list = []
        id_offset = -1
        id_offset_list1 = []

        if os.path.exists(os.path.join(INPUT_DIR, scene_name)) == False:
            continue

        num_frames = int(len(os.listdir(os.path.join(INPUT_DIR, scene_name)))/2)
        
        progress_bar = tqdm.tqdm(range(num_frames))
        for frame_num in progress_bar:
            progress_bar.set_description(f"Processing {scene_name} ({scene_num})")

            image_size = [900, 1600]
            ratio = 1.0
            image_size = [int(i * ratio) for i in image_size]
            io_start = time.time()
            try:
                masks_compressed = pickle.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_masks.pkl"), 'rb'))
                data = json.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_data.json")))
            except KeyError:
                continue
            
            depth_images = np.array(pycocotools.mask.decode(masks_compressed))
        
            depth_images = depth_images.transpose([2, 1, 0]) # num_masks_for_frames x h x w

            pointsensor_token = sample['data'][pointsensor_channel]
            pointsensor = nusc.get('sample_data', pointsensor_token)
            
            aggr_set = []
            
            # Loop for LiDAR pcd aggregation
            pointsensor_pre = nusc.get('sample_data', pointsensor_token)            
            for i in range(aggregation_pre_num):
                pcl_path = os.path.join(nusc.dataroot, pointsensor_pre['filename'])
                pc = LidarPointCloud.from_file(pcl_path, DEVICE)

                lidar_points = pc.points
                mask = torch.ones(lidar_points.shape[1]).to(device=DEVICE)
                mask = torch.logical_and(mask, torch.abs(lidar_points[0, :]) < np.sqrt(min_dist))
                mask = torch.logical_and(mask, torch.abs(lidar_points[1, :]) < np.sqrt(min_dist))
                lidar_points = lidar_points[:, ~mask]
                pc = LidarPointCloud(lidar_points)

                # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
                # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                cs_record = nusc.get('calibrated_sensor', pointsensor_pre['calibrated_sensor_token'])
                pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))

                # Second step: transform from ego to the global frame.
                poserecord = nusc.get('ego_pose', pointsensor_pre['ego_pose_token'])
                pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))

                aggr_set.append(pc.points)
                try:
                    pointsensor_pre = nusc.get('sample_data', pointsensor_pre['prev'])
                except KeyError:
                    break
            
            pointsensor_next = nusc.get('sample_data', pointsensor_token)
            for i in range(aggregation_next_num):
                if i == 0:
                    try:
                        pointsensor_next = nusc.get('sample_data', pointsensor_next['next'])
                        continue
                    except KeyError:
                        break                    
                pcl_path = os.path.join(nusc.dataroot, pointsensor_next['filename'])
                pc = LidarPointCloud.from_file(pcl_path, DEVICE)

                lidar_points = pc.points
                mask = torch.ones(lidar_points.shape[1]).to(device=DEVICE)
                mask = torch.logical_and(mask, torch.abs(lidar_points[0, :]) < np.sqrt(min_dist))
                mask = torch.logical_and(mask, torch.abs(lidar_points[1, :]) < np.sqrt(min_dist))
                lidar_points = lidar_points[:, ~mask]
                pc = LidarPointCloud(lidar_points)

                # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
                # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                cs_record = nusc.get('calibrated_sensor', pointsensor_next['calibrated_sensor_token'])
                pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))

                # Second step: transform from ego to the global frame.
                poserecord = nusc.get('ego_pose', pointsensor_next['ego_pose_token'])
                pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))

                aggr_set.append(pc.points)
                try:
                    pointsensor_next = nusc.get('sample_data', pointsensor_next['next'])
                except KeyError:
                    break
            
            aggr_pc_points = torch.hstack(tuple([pcd for pcd in aggr_set]))

            ratio = 1.0

            # Storing the camera intrinsic and extrinsic matrices for all cameras.
            # To store: camera_intrinsic matrix, 
            cam_data_list = []
            for camera in CAM_LIST:
                camera_token = sample['data'][camera]

                # Here we just grab the front camera
                cam_data = nusc.get('sample_data', camera_token)
                poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
                cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

                cam_data_dict = {
                    "ego_pose": poserecord,
                    "calibrated_sensor": cs_record,
                }

                cam_data_list.append(cam_data_dict)

            io_end = time.time()
            timer["io"] += io_end - io_start
            
            
            # Loop on each depth mask
            for i, (label, score, c) in enumerate(zip(data["labels"], data["detection_scores"], data["cam_nums"])):
                id_offset += 1

                pim_start = time.time()
                cam_data = cam_data_list[c]

                maskarr_1 = depth_images[i]
                mask_px_count = np.count_nonzero(maskarr_1)

                """ Commentable code for erosion """
                kernel = np.ones((3, 3), np.uint8)
                maskarr_1 = cv2.erode(maskarr_1, kernel)
                """ """

                mask_1 = Image.fromarray(maskarr_1)
                maskarr_1 = maskarr_1[:, :].astype(bool)
                maskarr_1 = torch.transpose(torch.from_numpy(maskarr_1).to(device=DEVICE, dtype=bool), 1, 0)

                
                # array to track id of masked points
                track_points = np.array(range(aggr_pc_points.shape[1]))
                

                # pass in a copy of the aggregate pointcloud array
                # reset the lidar pointcloud
                cam_pc = LidarPointCloud(torch.clone(aggr_pc_points))

                poserecord = cam_data['ego_pose']
                cam_pc.translate(torch.from_numpy(-np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))
                cam_pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

                # transform from ego into the camera.
                # cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                cs_record = cam_data['calibrated_sensor']
                cam_pc.translate(torch.from_numpy(-np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))
                cam_pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

                # actually take a "picture" of the point cloud.
                # Grab the depths (camera frame z axis points away from the camera).
                depths = cam_pc.points[2, :]

                coloring = depths

                camera_intrinsic = torch.from_numpy(np.array(cs_record["camera_intrinsic"])).to(device=DEVICE, dtype=torch.float32)
                camera_intrinsic = camera_intrinsic*ratio
                camera_intrinsic[2, 2] = 1

                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points, point_depths = view_points(cam_pc.points[:3, :], camera_intrinsic, normalize=True, device=DEVICE)

                image_mask = maskarr_1 # (W, H)
                # Create a boolean mask where True corresponds to masked pixels
                masked_pixels = (image_mask == 1) # (W, H)

                # Use np.logical_and to find points within masked pixels
                points_within_image = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(
                    depths > min_dist,                      # depths (N)
                    points[0] > 0),                          # points (3, N) -> points[0, :] (1, N)
                    points[0] < image_mask.shape[0] - 1),    # ^
                    points[1] > 0),                          # ^
                    points[1] < image_mask.shape[1] - 1     # ^
                )

                floored_points = torch.floor(points[:, points_within_image]).to(dtype=int) # (N_masked,)
                track_points = track_points[points_within_image.cpu()]

                points_within_mask = torch.logical_and(
                    floored_points,
                    masked_pixels[floored_points[0], floored_points[1]]
                )

                indices_within_mask = torch.where(torch.logical_and(torch.logical_and(points_within_mask[0, :], points_within_mask[1, :]), points_within_mask[2, :]))[0]
                masked_points_pixels = torch.where(points_within_mask)

                # Now, indices_within_mask contains the indices of points within the masked pixels
                track_points = track_points[indices_within_mask.cpu()]

                
                global_masked_points = aggr_pc_points[:, track_points]

                pim_end = time.time()
                timer["points in mask"] += pim_end - pim_start

                
                if global_masked_points.numel() == 0:
                    # No lidar points in the mask
                    continue

                id_offset_list1.append(id_offset)


                """ Centroid using medoid """
                medoid_start = time.time()
                
                if len(global_masked_points.shape) == 1:
                    global_masked_points = torch.unsqueeze(global_masked_points, 1)
                global_centroid = get_medoid(global_masked_points[:3, :].to(dtype=torch.float32, device=DEVICE))
                
                mask_pc = LidarPointCloud(global_masked_points[:, global_centroid][None].T)

                centroid = mask_pc.points[:3]

                all_centroids_list.append(torch.Tensor(centroid).to(DEVICE, dtype=torch.float32))
                global_masked_points_list.append(torch.Tensor(global_masked_points).to(DEVICE, dtype=torch.float32))
                centroid_ids.append(id_offset)
                medoid_end = time.time()
                timer["medoid"] += medoid_end - medoid_start
                final_id_offset = id_offset

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])
        """ End of object centroids loop """

        all_centroids_list = torch.stack(all_centroids_list)
        all_centroids_list = torch.squeeze(all_centroids_list)
        

        yaw_list, min_distance_list, lane_pt_coords_list = lane_yaws_distances_and_coords(
            all_centroids_list, lane_pt_list
        )

        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
        scene = nusc.get('scene', scene_token)
        sample = nusc.get('sample', scene['first_sample_token'])

        # Get map
        nusc_map = get_nusc_map(nusc, scene)

        drivable_records = nusc_map.drivable_area

        drivable_polygons = []
        for record in drivable_records:
            polygons = [nusc_map.extract_polygon(token) for token in record["polygon_tokens"]]
            drivable_polygons.extend(polygons)

        lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)

        id_offset = -1
        id_offset_list2 = []

        for frame_num in range(num_frames):

            cam_data_list = []
            for camera_name in CAM_LIST:
                camera_token = sample['data'][camera_name]
                sd_rec = nusc.get('sample_data', camera_token)
                file_name = sd_rec['filename'].split('/')[-1]
                if file_name not in results_GD_new:
                    boxes_filt_list = []
                    pred_phrases = ''
                else:
                    boxes_filt_list = results_GD_new[file_name]['boxes_filt_list']
                    pred_phrases = results_GD_new[file_name]['pred_phrases'] 

                cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                cs_rec_translation = np.array(cs_rec['translation'], dtype=np.float32)
                pose_rec_translation = np.array(pose_rec['translation'], dtype=np.float32)
                cs_rec_rotation_matrix = Quaternion(cs_rec['rotation']).rotation_matrix.astype(np.float32)
                pose_rec_rotation_matrix = Quaternion(pose_rec['rotation']).rotation_matrix.astype(np.float32)
                cam_intrinsic = np.array(cs_rec['camera_intrinsic'], dtype=np.float32)

                cam_data_dict = {
                    "boxes_filt_list": boxes_filt_list,
                    "pred_phrases": pred_phrases,
                    "cs_rec_translation": cs_rec_translation,
                    "pose_rec_translation": pose_rec_translation,
                    "cs_rec_rotation_matrix": cs_rec_rotation_matrix,
                    "pose_rec_rotation_matrix": pose_rec_rotation_matrix,
                    "cam_intrinsic": cam_intrinsic,
                }
                cam_data_list.append(cam_data_dict)


            data = json.load(open(os.path.join(INPUT_DIR, scene_name, f"{frame_num}_data.json")))
            predictions["results"][sample["token"]] = []
            for i, (label, score, c) in enumerate(zip(data["labels"], data["detection_scores"], data["cam_nums"])):

                id_offset += 1
                if id_offset not in centroid_ids:
                    continue
                else:
                    id = centroid_ids.index(id_offset)
                    final_id_offset2 = id_offset
                
                id_offset_list2.append(id_offset)
                detection_name = get_detection_name(label)
                centroid = np.squeeze(np.array(all_centroids_list[id, :].to(device='cpu')))

                global_masked_points = np.array(global_masked_points_list[id].to(device='cpu'))
                global_masked_points = np.swapaxes(global_masked_points, 0, 1)

                cl_start = time.time()
                m_x, m_y, m_z = [float(i) for i in centroid]

                dist_from_lane = min_distance_list[id]
                lane_yaw = yaw_list[id]

                extents = shape_priors[detection_name]


                if detection_name in ["car", "truck", "bus", "bicycle", "motorcycle", "emergency_vehicle",
                                      "stroller", "personal_mobility", "barrier"]:
                    point = Point(m_x, m_y)
                    drivable_start = time.time()
                    is_drivable = False
                    for polygon in drivable_polygons:
                        if point.within(polygon):
                            is_drivable = True
                            break
                    drivable_end = time.time()
                    timer["drivable"] += drivable_end - drivable_start
                    
                    lane_yaw_list = [lane_yaw, lane_yaw+np.pi/2]
                    candidate_boxes, max_in_box_ratio = find_best_pts_ratio_box_m(global_masked_points, centroid, list(extents), lane_yaw_list) 

                    input_boxes = []
                    for i in range(len(cam_data_list[c]['boxes_filt_list'])):
                        if cam_data_list[c]['pred_phrases'][i] == f"{detection_name}({score})":
                            input_boxes.append(cam_data_list[c]['boxes_filt_list'][i])
                    if len(input_boxes) == 0:
                        continue
                    input_boxes = np.array(input_boxes)
                    best_box = np.zeros((8, 3), dtype=np.float32)
                    best_box = find_best_box(candidate_boxes, cam_data_list[c]['pose_rec_translation'], cam_data_list[c]['pose_rec_rotation_matrix'], 
                                             cam_data_list[c]['cs_rec_translation'], cam_data_list[c]['cs_rec_rotation_matrix'], 
                                             cam_data_list[c]['cam_intrinsic'], input_boxes)

                    if best_box.sum() == 0:
                        continue 
                    x_pro, y_pro, z_pro, w_pro, l_pro, h_pro, yaw_pro = box_properties(best_box)
                    pushed_centroid = [x_pro, y_pro, z_pro]
                    align_mat = np.eye(3)
                    align_mat[0:2, 0:2] = [[np.cos(yaw_pro), -np.sin(yaw_pro)], [np.sin(yaw_pro), np.cos(yaw_pro)]]
                elif detection_name in ["trailer", "construction_vehicle"]:
                    point = Point(m_x, m_y)
                    drivable_start = time.time()
                    is_drivable = False
                    for polygon in drivable_polygons:
                        if point.within(polygon):
                            is_drivable = True
                            break
                    drivable_end = time.time()
                    timer["drivable"] += drivable_end - drivable_start

                    align_mat = np.eye(3)
                    align_mat[0:2, 0:2] = [[np.cos(lane_yaw), -np.sin(lane_yaw)], [np.sin(lane_yaw), np.cos(lane_yaw)]]

                    """ Pushback code """
                    # Push centroid back for large objects
                    pointsensor_token = sample['data'][pointsensor_channel]
                    pointsensor = nusc.get('sample_data', pointsensor_token)
                    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
                    pushed_centroid = push_centroid(centroid, extents, Quaternion(matrix=align_mat), poserecord)
                    # pushed_centroid = centroid
                    """ """                    
                else:
                    """ Commented out code ends """
                    align_mat = np.eye(3)
                    pushed_centroid = centroid

                #  mlo score to define the detection score again
                if use_mlo_score:
                    box_pro = [float(i) for i in pushed_centroid] + list(extents) + [float(lane_yaw)]
                    mlo_parts = [9, 7, 5, 3]
                    mlo_score = hierarchical_occupancy_score(global_masked_points, box_pro, mlo_parts)


                rot_quaternion = Quaternion(matrix=align_mat)
                box_dict = {
                        "sample_token": sample["token"],
                        "translation": [float(i) for i in pushed_centroid],
                        "size": list(extents),
                        "rotation": list(rot_quaternion),
                        "velocity": [0, 0],
                        "detection_name": detection_name,
                        "detection_score": score,
                        "mlo_score": mlo_score,
                        "attribute_name": '',
                        "points_num": global_masked_points.shape[0]
                    }

                
                assert sample["token"] in predictions["results"]
                    
                predictions["results"][sample["token"]].append(box_dict)

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])


    print("\nRunning NMS on the predictions.\n")
    nms_start = time.time()
    
    final_predictions = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": True,
            "use_external": False,
        },
        "results": {}
    }

    for sample in predictions["results"]:
        final_predictions["results"][sample] = []

        dets = []
        det_labels = []
        # threshs borrowed from centerpoint
        threshs_by_label = {
            "barrier": 1,
            "traffic_cone": 0.175,
            "bicycle": 0.85,
            "motorcycle": 0.85,
            "pedestrian": 0.175,
            "car": 4,
            "bus": 10,
            "construction_vehicle": 12,
            "trailer": 10,
            "truck": 12,

            "emergency_vehicle": 4,
            "adult": 0.175,
            "child": 0.175,
            "police_officer": 0.175,
            "construction_worker": 0.175,
            "stroller": 1,
            "personal_mobility": 0.175,
            "pushable_pullable": 1,
            "debris": 1,
        }     

        centroids_list = []
        extents_list = []
        rot_list = []
        det_names_list = []
        attr_names_list = []
        vertices_list = []
        scores = []
        mlo_scores = []
        points_num_list = []

        for box_dict in predictions["results"][sample]:
            centroid = box_dict["translation"]
            extents = box_dict["size"]
            score = box_dict["detection_score"]
            mlo_score = box_dict["mlo_score"]
            rot = box_dict["rotation"]
            detection_name = box_dict["detection_name"]
            attr_name = box_dict["attribute_name"]
            points_num = box_dict["points_num"]

            rot_quaternion = Quaternion(rot)

            dets.append(np.array([centroid[0], centroid[1], score]))
            det_labels.append(detection_name)

            centroids_list.append(centroid)
            extents_list.append(extents)
            rot_list.append(rot)
            det_names_list.append(detection_name)
            attr_names_list.append(attr_name)
            scores.append(score)
            mlo_scores.append(mlo_score)
            points_num_list.append(points_num)

        dets = np.array(dets)
        print(len(det_labels), end=" ")
        if len(det_labels) > 0:
            keep_indices = list(circle_nms(dets, det_labels, threshs_by_label))
        else:
            # Skip this sample if we dont have any predictions in it
            continue

        print(len(keep_indices))

        """ Commentable code for NMS """
        centroids_list = [centroids_list[c] for c in range(len(centroids_list)) if c in keep_indices]
        extents_list = [extents_list[c] for c in range(len(extents_list)) if c in keep_indices]
        rot_list = [rot_list[c] for c in range(len(rot_list)) if c in keep_indices]
        det_names_list = [det_names_list[c] for c in range(len(det_names_list)) if c in keep_indices]
        attr_names_list = [attr_names_list[c] for c in range(len(attr_names_list)) if c in keep_indices]
        scores = [scores[c] for c in range(len(scores)) if c in keep_indices]
        mlo_scores = [mlo_scores[c] for c in range(len(mlo_scores)) if c in keep_indices]
        points_num_list = [points_num_list[c] for c in range(len(points_num_list)) if c in keep_indices]

        # """ Commentable code """

        for i, (centroid, extents, rot, det_name, attr_name, score, mlo_score, points_num) in enumerate(zip(
            centroids_list, extents_list, rot_list, det_names_list, attr_names_list, scores, mlo_scores, points_num_list
        )):
            # print(i, end='')
            box_dict = {
                "sample_token": sample,
                "translation": centroid,
                "size": extents,
                "rotation": rot,
                "velocity": [0, 0],
                "detection_name": det_name,
                "detection_score": score,
                "mlo_score": mlo_score,
                "attribute_name": attr_name,
                "points_num": points_num,
            }

            final_predictions["results"][sample].append(box_dict)

    nms_end = time.time()
    timer["nms"] += nms_end - nms_start

    with open(os.path.join(OUTPUT_DIR, "GD_OriTrained_AggCurr_CM3D_PointsNum&Mlo.json"), "w") as f:
        json.dump(final_predictions, f)

    print(f"wrote {len(final_predictions['results'])} samples.")

    total_end = time.time()
    timer["total"] += total_end - total_start

    for operation in timer:
        print(operation, ":\t\t", timer[operation])