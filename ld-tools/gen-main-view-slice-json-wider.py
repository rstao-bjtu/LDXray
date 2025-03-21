import json
import os
import argparse
from PIL import Image
from tqdm import tqdm

def is_overlapping(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1)

def crop_images_by_non_overlapping_annotations(coco_json_path, images_dir, output_images_dir, output_json_path):
    os.makedirs(output_images_dir, exist_ok=True)
    print('loading json......')
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    print('finished loading')
    new_images = []
    new_annotations = []
    annotation_id = 1

    print('start processing')
    for image_info in tqdm(coco_data['images'], desc="Processing images"):
        image_id = image_info['id']
        image_path = os.path.join(images_dir, image_info['file_name'])
        image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

        with Image.open(image_path) as img:
            for annotation in image_annotations:
                if all(not is_overlapping(annotation['bbox'], other['bbox']) for other in image_annotations if other != annotation):
                    x, y, width, height = annotation['bbox']
                    img_width, img_height = img.size  


                    delta = 0.3 * width

                    new_x = max(x - delta, 0)
                    new_width = width + 2 * delta

                    if new_x + new_width > img_width:
                        new_width = img_width - new_x


                    cropped_img = img.crop((new_x, 0, new_x + new_width, img_height))

          
                    new_bbox_x = x - new_x 
                    new_bbox = [new_bbox_x, y, width, height]

                    new_image_name = f"{image_info['file_name'][:-4]}_{annotation_id}.jpg"
                    new_image_path = os.path.join(output_images_dir, new_image_name)
                    cropped_img.save(new_image_path)

                    new_image_info = {
                        'id': image_id,
                        'file_name': new_image_name,
                        'width': new_width,
                        'height': img_height
                    }
                    new_images.append(new_image_info)

                    new_annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': annotation['category_id'],
                        'bbox': new_bbox,
                        'area': width * height,
                        'iscrowd': annotation['iscrowd']
                    }
                    new_annotations.append(new_annotation)

                    annotation_id += 1

    new_coco_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_data['categories']
    }

    with open(output_json_path, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images by non-overlapping annotations from COCO dataset with extended width.")
    parser.add_argument('coco_json_path', type=str, help="Path to the COCO JSON file.")
    parser.add_argument('images_dir', type=str, help="Directory containing the images.")
    parser.add_argument('output_images_dir', type=str, help="Directory to save the cropped images.")
    parser.add_argument('output_json_path', type=str, help="Path to save the new COCO JSON file.")

    args = parser.parse_args()

    crop_images_by_non_overlapping_annotations(
        coco_json_path=args.coco_json_path,
        images_dir=args.images_dir,
        output_images_dir=args.output_images_dir,
        output_json_path=args.output_json_path
    )