import json
import pickle
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R

classes = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                 'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                 'pushable_pullable', 'debris', 'traffic_cone', 'barrier')

DefaultAttribute = {
    'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.stopped',
    'construction_vehicle': 'vehicle.parked',
    'barrier': '',
    'traffic_cone': '',
    "adult" : 'pedestrian.moving',
    "child" : 'pedestrian.moving',
    "stroller" : '',
    "personal_mobility" : '',
    "police_officer" : 'pedestrian.moving',
    "construction_worker" : 'pedestrian.moving',
    "car" : 'vehicle.parked',
    "motorcycle" : 'cycle.without_rider',
    "bicycle" : 'cycle.without_rider',
    "emergency_vehicle" : 'vehicle.parked',
    "pushable_pullable" : '',
    "debris" : '',
    
    'vehicle': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'movable': '',        
}

if __name__ == "__main__": 
    argparser = ArgumentParser()
    argparser.add_argument(
        "--info_data_path", default="./data/nuscenes/nuscenes_infos_val.pkl")
    argparser.add_argument("--prediction_path", default="outputs/nuscenes/results_3D/test.json")
    argparser.add_argument(
        "--prediction_sorted_pkl_path", default="outputs/nuscenes/results_3D/test.pkl")
    
    config = argparser.parse_args()

    with open(config.info_data_path, "rb") as f:
        raw_labels = pickle.load(f)
    raw_labels = list(sorted(raw_labels["infos"], key=lambda e: e["timestamp"])) 
    with open(config.prediction_path) as f:
        predictions = json.load(f)

    # Sort prediction and add attribute_name
    predictions_sorted = {
        'meta': predictions['meta'],
        'results': {}
    }
    for raw_label in raw_labels:
        token_name = raw_label['token']
        if token_name in predictions['results']:
            for pre in predictions['results'][token_name]:
                pre['attribute_name'] = DefaultAttribute[pre['detection_name']]
            predictions_sorted['results'][token_name] = predictions['results'][token_name]
    
    # json to pkl
    prediction_sorted_pkl = []
    for raw_label in raw_labels:
        token_name = raw_label['token']    
        translation_list, size_list, yaw_list, = [], [], []
        velocity_list, detection_name_list, detection_label_list, detection_score_list, = [], [], [], []
        points_path_list = []
        for prediction in predictions_sorted['results'][token_name]:
            translation_list.append(prediction['translation'])
            size_list.append(prediction['size'])
            q = [prediction['rotation'][-1], prediction['rotation'][1], prediction['rotation'][2], prediction['rotation'][0]] 
            r = R.from_quat(q)
            yaw = r.as_euler('xyz')[0]
            yaw_list.append(yaw)
            velocity_list.append(prediction['velocity'])
            detection_name_list.append(prediction['detection_name'])
            detection_label_list.append(classes.index(prediction['detection_name']))
            detection_score_list.append(prediction['detection_score'])
            if 'points_path' in prediction:
                points_path_list.append(prediction['points_path'])

        translations = np.array(translation_list)
        sizes = np.array(size_list)
        yaws = np.array(yaw_list).reshape(-1, 1)
        velocitys = np.array(velocity_list)    
        boxes_3d = np.concatenate((translations, sizes, yaws, velocitys), axis=1)
        labels_3d = np.array(detection_label_list)
        scores_3d = np.array(detection_score_list)
        if len(points_path_list) > 0:
            prediction_sorted_pkl.append(
                {
                    'boxes_3d': boxes_3d, 
                    'labels_3d': labels_3d,
                    'scores_3d': scores_3d,
                    'points_path_list': points_path_list
                }
            )       
        else:
            prediction_sorted_pkl.append(
                {
                    'boxes_3d': boxes_3d, 
                    'labels_3d': labels_3d,
                    'scores_3d': scores_3d
                }
            )
    with open(config.prediction_sorted_pkl_path, "wb") as f:
        pickle.dump(prediction_sorted_pkl, f)

