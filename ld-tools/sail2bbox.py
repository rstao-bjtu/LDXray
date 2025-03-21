
import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def get_bounding_boxes(salmap, threshold=50):
    _, binary_map = cv2.threshold(salmap, threshold, 255, cv2.THRESH_BINARY)
    binary_map = binary_map.astype(np.uint8)

    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = [cv2.boundingRect(contour) for contour in contours]

    return bboxes

def get_non_white_bbox(img, x_min, x_max):
    white_threshold = 240
    non_white_mask = np.all(img[:, x_min:x_max] < white_threshold, axis=-1)

    coords = np.column_stack(np.where(non_white_mask))
    
    if coords.size == 0:
        height = img.shape[0]
        y_min = height // 2
        y_max = height - 1
        return x_min, y_min,x_max - x_min, y_max - y_min
    

    y_min = coords[:, 0].min()
    y_max = coords[:, 0].max()
    

    return x_min, y_min, x_max - x_min, y_max - y_min


def convert_to_standard_types(data):
    if isinstance(data, dict):
        return {k: convert_to_standard_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_standard_types(v) for v in data]
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    else:
        return int(data)  

def create_new_coco_dataset(input_json_path, image_folder, output_json_path, target_category_ids):
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    new_coco_data = {
        "images": [],
        "annotations": [],
        "categories": [        {
            "id": 1,
            "name": "Mobile_Phone",
            "supercategory": "Mobile_Phone"
        },
        {
            "id": 2,
            "name": "Orange_Liquid",
            "supercategory": "Orange_Liquid"
        },
        {
            "id": 3,
            "name": "Charger_Without_Cell",
            "supercategory": "Charger_Without_Cell"
        },
        {
            "id": 4,
            "name": "Laptop",
            "supercategory": "Laptop"
        },
        {
            "id": 5,
            "name": "Green_Liquid",
            "supercategory": "Green_Liquid"
        },
        {
            "id": 6,
            "name": "Charger_With_Cell",
            "supercategory": "Charger_With_Cell"
        },
        {
            "id": 7,
            "name": "Tablet",
            "supercategory": "Tablet"
        },
        {
            "id": 8,
            "name": "Blue_Liquid",
            "supercategory": "Blue_Liquid"
        },
        {
            "id": 9,
            "name": "Cylindrical_Orange_Liquid",
            "supercategory": "Cylindrical_Orange_Liquid"
        },
        {
            "id": 10,
            "name": "Nonmetallic_Lighter",
            "supercategory": "Nonmetallic_Lighter"
        },
        {
            "id": 11,
            "name": "Umbrella",
            "supercategory": "Umbrella"
        },
        {
            "id": 12,
            "name": "Cylindrical_Green_Liquid",
            "supercategory": "Cylindrical_Green_Liquid"
        }]
    }

    image_id_map = {}
    annotation_id = 1

    for image_info in tqdm(coco_data['images']):
        image_id = image_info['id']
        image_path = os.path.join(image_folder, image_info['file_name'])
        img = Image.open(image_path).convert("RGB")
        img = np.asarray(img)

        relevant_annos = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id and anno['category_id'] in target_category_ids]

        if relevant_annos:
            new_image_info = image_info.copy()
            new_image_info['id'] = len(new_coco_data['images']) + 1
            image_id_map[image_info['id']] = new_image_info['id']
            new_coco_data['images'].append(new_image_info)

            for anno in relevant_annos:
                x, y, w, h = anno['bbox']

                x_min, x_max = int(x-0.15*w), int(x + w + 0.15*w)



                m_bbox = get_non_white_bbox(img, x_min, x_max)

                x, y, w, h = m_bbox
                new_anno = {
                    "id": annotation_id,
                    "image_id": new_image_info['id'],
                    "category_id": anno["category_id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                }
                new_coco_data['annotations'].append(new_anno)
                annotation_id += 1


    new_coco_data = convert_to_standard_types(new_coco_data)

    with open(output_json_path, 'w') as f:
        print(new_coco_data, output_json_path)
        json.dump(new_coco_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new COCO dataset with salient object bounding boxes.")
    parser.add_argument('input_json_path', type=str, help="Path to the input COCO JSON file.")
    parser.add_argument('image_folder', type=str, help="Path to the folder containing images.")
    parser.add_argument('output_json_path', type=str, help="Path to the output COCO JSON file.")
    parser.add_argument('target_category_ids', type=int, nargs='+', help="Target category IDs to filter annotations.")

    args = parser.parse_args()

    create_new_coco_dataset(
        input_json_path=args.input_json_path,
        image_folder=args.image_folder,
        output_json_path=args.output_json_path,
        target_category_ids=args.target_category_ids
    ) 