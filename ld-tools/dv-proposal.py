import os
import json
import sys
from tqdm import tqdm
from PIL import Image
from collections import Counter

def is_overlapping(bbox1, bbox2):

    x1_min, y1_min, w1, h1 = bbox1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_min, y2_min, w2, h2 = bbox2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def main(json_file_path, ori_img_path, new_image_dir, new_json_dir, score_threshold, non_overlap_threshold):

    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
        print(f"create: {new_image_dir}")


    with open(json_file_path, 'r') as f:
        predictions = json.load(f)

    images = []
    annotations = []
    annotation_id = 1


    predictions_by_image_id = {}
    for prediction in predictions:
        if prediction['score'] < score_threshold:
            continue
        image_id = prediction['image_id']
        if image_id not in predictions_by_image_id:
            predictions_by_image_id[image_id] = []
        predictions_by_image_id[image_id].append(prediction)


    for image_id, preds in tqdm(predictions_by_image_id.items(), desc="Processing images"):
        image_path = preds[0]['image_path']
        image_name = os.path.basename(image_path)
        ori_img = os.path.join(ori_img_path, image_name)
        image = Image.open(ori_img)
        image_width, image_height = image.size


        merged_boxes = []
        for i, pred in enumerate(preds):
            bbox = pred['bbox']
            category_id = pred['category_id']
            score = pred['score']

            overlapping = False
            for j, other_pred in enumerate(preds):
                if i != j and is_overlapping(bbox, other_pred['bbox']):
                    overlapping = True
                    break

            if not overlapping and score < non_overlap_threshold:
                continue

            merged = False
            for mbox in merged_boxes:
                if (bbox[0] < mbox['bbox'][0] + mbox['bbox'][2] and bbox[0] + bbox[2] > mbox['bbox'][0] and
                    bbox[1] < mbox['bbox'][1] + mbox['bbox'][3] and bbox[1] + bbox[3] > mbox['bbox'][1]):
                    mbox['bbox'][0] = min(mbox['bbox'][0], bbox[0])
                    mbox['bbox'][1] = min(mbox['bbox'][1], bbox[1])
                    mbox['bbox'][2] = max(mbox['bbox'][0] + mbox['bbox'][2], bbox[0] + bbox[2]) - mbox['bbox'][0]
                    mbox['bbox'][3] = max(mbox['bbox'][1] + mbox['bbox'][3], bbox[1] + bbox[3]) - mbox['bbox'][1]
                    mbox['category_ids'].append(category_id)
                    merged = True
                    break
            if not merged:
                merged_boxes.append({'bbox': bbox, 'category_ids': [category_id]})


        for mbox in merged_boxes:
            min_x, min_y, width, height = mbox['bbox']
            max_x = min_x + width
            max_y = min_y + height

            cropped_image = image.crop((min_x, 0, max_x, image_height))
            new_image_name = f"{os.path.splitext(image_name)[0]}_{int(min_x)}_{int(min_y)}_{int(width)}_{int(height)}.jpg"
            new_image_path = os.path.join(new_image_dir, new_image_name)
            cropped_image.save(new_image_path)

            most_common_category_id = Counter(mbox['category_ids']).most_common(1)[0][0]

            images.append({
                "id": image_id,
                "file_name": new_image_name,
                "width": int(width),
                "height": int(height)
            })


            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": most_common_category_id,
                "bbox": [0, 0, int(width), int(height)],
                "area": int(width * height),
                "segmentation": [],
                "iscrowd": 0
            })

            annotation_id += 1


    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [
        {
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
        }
        ]
    }

    with open(new_json_dir, 'w') as f:
        json.dump(coco_format, f, indent=4)



if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(" python script.py <json_file_path> <ori_img_path> <new_image_dir> <new_json_dir> <score_threshold> <non_overlap_threshold>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    ori_img_path = sys.argv[2]
    new_image_dir = sys.argv[3]
    new_json_dir = sys.argv[4]
    score_threshold = float(sys.argv[5])
    non_overlap_threshold = float(sys.argv[6])
    main(json_file_path, ori_img_path, new_image_dir, new_json_dir, score_threshold, non_overlap_threshold)