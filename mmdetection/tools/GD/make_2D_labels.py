import os
import numpy as np
import cv2
import json
import pickle
from argparse import ArgumentParser
from typing import List
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes

from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords, generate_record

CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
           'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility',
           'pushable_pullable', 'debris', 'traffic_cone', 'barrier']
class_names_mapping = {
        "vehicle.car": "car", 
        "vehicle.bus.bendy": "bus", 
        "vehicle.bus.rigid": "bus", 
        "vehicle.truck": "truck", 
        "vehicle.construction": "construction_vehicle",
        "vehicle.emergency.ambulance": "emergency_vehicle", 
        "vehicle.emergency.police": "emergency_vehicle", 
        "vehicle.trailer": "trailer", 
        "vehicle.motorcycle": "motorcycle", 
        "vehicle.bicycle": "bicycle", 

        "human.pedestrian.adult": "adult", 
        "human.pedestrian.child": "child", 
        "human.pedestrian.police_officer": "police_officer", 
        "human.pedestrian.construction_worker": "construction_worker", 
        "human.pedestrian.personal_mobility": "personal_mobility", 
        "human.pedestrian.stroller": "stroller",

        "movable_object.pushable_pullable": "pushable_pullable",
        "movable_object.debris": "debris", 
        "movable_object.barrier": "barrier", 
        "movable_object.trafficcone": "traffic_cone", 
    }
views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

def get_2d_boxes(sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


if __name__ == "__main__": 
    argparser = ArgumentParser()
    argparser.add_argument(
        "--info_path", default="data/nuscenes/nuscenes_infos_val.pkl")   
    argparser.add_argument("--output_dir_path", default="data/nuscenes/samples/labels_2D_COCO/CAM_ALL_val/") 
    argparser.add_argument("--show", default=False)    
    config = argparser.parse_args()
 
    '''Load nuscenes data'''
    dataroot = 'data/nuscenes'
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    info_data = pickle.load(open(config.info_path, 'rb'))
    info_data = list(info_data['infos'])

    for frame_num, infos in enumerate(info_data): 
        token = infos['token']
        s_rec = nusc.get('sample', token)        
        for view in views:
            sample_data_token = s_rec['data'][view] 
            
            visibilities = ['', '1', '2', '3', '4']
            label_boxes = get_2d_boxes(sample_data_token, visibilities)

            label_path = config.output_dir_path + infos['cams'][view]['data_path'].split('/')[-1].split('.jpg')[0] + '.txt'
            label_file = open(label_path, 'w')
            label_num = len(label_boxes)
            for i in range(label_num):
                box_name = label_boxes[i]['category_name']
                if box_name not in class_names_mapping:
                    continue
                gt_cls = class_names_mapping[box_name]

                cls_id = CLASSES.index(gt_cls)
                bbox = label_boxes[i]['bbox_corners']
                if config.show:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                label_file.write(str(cls_id)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2]-bbox[0])+' '+str(bbox[3]-bbox[1])+'\n')
            label_file.close()


 