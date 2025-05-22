import os
import cv2
import json
from argparse import ArgumentParser


class_prompt_list = ['car', 'truck', 'trailer', 'bus', 
                    'construction_vehicle', 'bicycle bike', 'narrow motorcycle', 
                    'emergency vehicle', 'adult', 
                    'single little short youth children', 'law enforcement officer', 
                    'construction worker', 'stroller', 
                    'small kick scooter', 
                    'pushable pullable garbage container',
                    'full trash bags', 'traffic_cone', 'barrier']

if __name__ == "__main__": 
    argparser = ArgumentParser()
    argparser.add_argument(
        "-images_few_shot_dir_path", default="data/nuscenes/images_few_shot/")   
    argparser.add_argument("--labels_dir_path", default="data/nuscenes/samples/labels_2D_COCO/CAM_ALL_train/") 
    argparser.add_argument("--show", default=False)    
    argparser.add_argument("--show_path", default="outputs/annotation_test/") 
    argparser.add_argument("--output_path", default="data/nuscenes/train_2D_few_shot.json") 
    config = argparser.parse_args()


    coco_output = {
        "images": [], 
        "annotations": [], 
        "categories": []
    }
    categories = []
    for i, cls in enumerate(class_prompt_list):
        category = {"id": class_prompt_list.index(cls)+1, "name": cls}
        categories.append(category)
    coco_output["categories"] = categories
    annotation_id = 1
    for i, image_name in enumerate(os.listdir(config.images_few_shot_dir_path)):
        image_path =  os.path.join(config.images_few_shot_dir_path, image_name)
        img = cv2.imread(os.path.join(config.images_few_shot_dir_path, image_name))
        HEIGHT, WIDTH, _ = img.shape
        image_info = {
            "id": i+1,
            "file_name": image_name,
            "width": WIDTH,
            "height": HEIGHT,
        }
        coco_output["images"].append(image_info)
        label_path = config.labels_dir_path + image_name.split('.jpg')[0].split('&')[-1] + '.txt'
        labels_list = open(label_path, 'rb').readlines()      
        for label in labels_list:
            c, x, y, w, h = [float(_) for _ in label.strip().split()] 
            cls_name = class_prompt_list[int(c)]
            if cls_name not in class_prompt_list:
                continue   
            annotation_info = {
                "id": annotation_id,
                "image_id": i+1,
                "category_id": int(c)+1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [[x,y,x,y+h,x+w,y+h,x+w,y]]
            }
            coco_output["annotations"].append(annotation_info)
            annotation_id += 1
            if config.show:
                cv2.rectangle(img, (int(float(x)), int(float(y))), (int(float(x+w)), int(float(y+h))), (0, 255, 0), 2)
        if config.show:
            cv2.imwrite(config.show_path+image_name+".jpg", img)
    with open(config.output_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)



 