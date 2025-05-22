import os
import shutil
from mmengine.utils import mkdir_or_exist

images_few_shot_dir_path = 'data/nuscenes/images_few_shot_cls_dir/'
output_dir_path = 'data/nuscenes/images_few_shot/'
class_name_list = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
           'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility',
           'pushable_pullable', 'debris', 'traffic_cone', 'barrier'] 
class_prompt_list = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
           'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility',
           'pushable_pullable', 'debris', 'traffic_cone', 'barrier'] 

for index, class_name in enumerate(class_name_list):
    images_name_list = os.listdir(images_few_shot_dir_path+class_name)
    for image_name in images_name_list:
        class_prompt = class_prompt_list[index]
        image_name_new = class_prompt+'&'+image_name
        shutil.copy(images_few_shot_dir_path+class_name+'/'+image_name, output_dir_path+image_name_new)

