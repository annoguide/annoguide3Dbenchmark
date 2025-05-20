import os
import torch
import mmcv
from argparse import ArgumentParser
import pyquaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox

from tracker.data.nuscenes_adapter import (
    transform_to_reference_pose,
    unpack_and_annotate_labels,
)

from tracker.data.utils import *
from tracker.tracking.greedy_tracker import CLS_VELOCITY_ERROR_BY_DATASET, GreedyTracker
from tracker.tracking.utils import average_scores_across_track

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
    'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
    'pushable_pullable', 'debris', 'traffic_cone', 'barrier'
]

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

def load_annotations(ann_file):
    """Load annotations from ann_file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: List of annotations sorted by timestamps.
    """
    data = mmcv.load(ann_file, file_format='pkl')
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    data_infos = data_infos[::1]
    metadata = data['metadata']
    version = metadata['version']
    return data_infos

def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d']
    labels = detection['labels_3d']

    box_gravity_center = box3d[:, 0:3]
    box_dims = box3d[:, 3:6]
    box_yaw = box3d[:, 6]

    # our LiDAR coordinate system -> nuScenes box coordinate system
    # nus_box_dims = box_dims[:, [1, 0, 2]]
    nus_box_dims = box_dims[:, [0, 1, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (0, 0, 0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity
            )
        box_list.append(box)
    return box_list

def format_bbox(results, data_infos, output_path):
    """Convert the results to the standard format.

    Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str): The prefix of the output jsonfile.
            You can specify the output directory/filename by
            modifying the jsonfile_prefix. Default: None.

    Returns:
        str: Path of the output json file.
    """
    nusc_annos = {}
    mapped_class_names = class_names
    modality = dict(
        use_lidar=True,
        use_camera=False,
        use_radar=False,
        use_map=False,
        use_external=False)
    print('Start to convert detection format...')
    for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
        annos = []
        boxes = output_to_nusc_box(det, with_velocity=True)
        sample_token = data_infos[sample_id]['token']

        for i, box in enumerate(boxes):
            name = mapped_class_names[box.label]
            if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                ]:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = DefaultAttribute[name]
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = DefaultAttribute[name]
            nusc_anno = dict(
                sample_token=sample_token,
                translation=box.center.tolist(),
                size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(),
                detection_name=name,
                detection_score=box.score,
                attribute_name=attr)
            annos.append(nusc_anno)
        nusc_annos[sample_token] = annos
    nusc_submissions = {
        'meta': modality,
        'results': nusc_annos,
    }

    print('Results writes to', output_path)
    mmcv.dump(nusc_submissions, output_path)

if __name__ == "__main__": 
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="nuscenes", choices=["nuscenes"])
    argparser.add_argument("--dataset_dir_path", default="./data/nuscenes")
    argparser.add_argument(
        "--tracker", default="greedy_tracker", choices=["greedy_tracker"]
    )
    argparser.add_argument(
        "--info_data_path", default="./data/nuscenes/nuscenes_infos_val.pkl")   
    argparser.add_argument("--prediction_path", default="outputs/nuscenes/results_3D/test.pkl")   
    argparser.add_argument("--objective_metric", default="HOTA", choices=["HOTA", "MOTA"])   
    argparser.add_argument("--output_path", default="outputs/nuscenes/results_3D/test_average_score.json")
    config = argparser.parse_args()

    # constants
    config.max_track_age = 0
    config.score_threshold = 0.0
    config.num_score_thresholds = 10

    raw_predictions = load(config.prediction_path)
    raw_labels_dict = load(config.info_data_path)
    raw_labels, nusc_version = (
        raw_labels_dict["infos"],
        raw_labels_dict["metadata"]["version"],
    )
    raw_labels = list(sorted(raw_labels, key=lambda e: e["timestamp"]))
    nusc = NuScenes(nusc_version, dataroot=config.dataset_dir_path)

    prediction_list = unpack_predictions_cm3d(raw_predictions, class_names)
    label_list = unpack_and_annotate_labels(raw_labels, nusc, class_names)
    annotate_frame_metadata(
        prediction_list,
        label_list,
        [
            "seq_id",
            "timestamp_ns",
            "frame_id",
        ],
    )
    predictions = group_frames(prediction_list)
    labels = group_frames(label_list)
    labels = transform_to_reference_pose(labels)

    # generate tracks
    tracker = GreedyTracker(
        CLS_VELOCITY_ERROR_BY_DATASET[config.dataset], max_age=config.max_track_age
    )
    
    track_predictions = {}
    for seq_id, frames in progressbar(predictions.items(), "running tracker on logs"):
        tracker.reset()
        last_t = frames[0]["timestamp_ns"]
        track_predictions[seq_id] = []
        for frame in frames:
            time_delta_seconds = (frame["timestamp_ns"] - last_t) * 1e-9
            last_t = frame["timestamp_ns"]
            # filter predictions with confidence below threshold
            frame_predictions = index_array_values(
                frame, frame["score"] > config.score_threshold
            )
            track_frame = tracker.step(frame_predictions, time_delta_seconds)
            # only store active tracks
            track_frame = index_array_values(track_frame, track_frame["active"] > 0)
            for k, v in frame.items():
                if not isinstance(v, np.ndarray):
                    track_frame[k] = v
            track_predictions[seq_id].append(track_frame)

    # average scores across a single track
    track_predictions = average_scores_across_track(track_predictions)

    # Trans format to results
    result = list()
    for seq_id in track_predictions:
        for frame in track_predictions[seq_id]:
            result.append(
                {
                    'boxes_3d': frame['boxes_3d'], 
                    'labels_3d': frame['label'],
                    'scores_3d': frame['score'],
                }
            )   
    format_bbox(result, raw_labels, config.output_path)
