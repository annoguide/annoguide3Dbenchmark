from collections import defaultdict
from copy import deepcopy

import numpy as np

classes_not_average = ['car', 'bicycle', 'motorcycle', 'emergency_vehicle',
                       'child', 'personal_mobility', 
                       'traffic_cone']
def average_scores_across_track(tracks):
    score_avg_tracks = deepcopy(tracks)
    for seq_id in tracks:
        scores_by_id = defaultdict(list)
        for frame in tracks[seq_id]:
            for id, score in zip(frame["track_id"], frame["score"]):
                scores_by_id[id].append(score)
        score_by_id = {id: np.mean(scores) for id, scores in scores_by_id.items()}
        # score_by_id = {id: np.max(scores) for id, scores in scores_by_id.items()}
        for frame in score_avg_tracks[seq_id]:
            frame["detection_score"] = frame["score"]
            for index, id in enumerate(frame["track_id"]):
                if frame['name'][index] not in classes_not_average:
                    frame["score"][index] = score_by_id[id]
            # frame["score"] = np.array([score_by_id[id] for id in frame["track_id"]])
    return score_avg_tracks
