import os
import os.path as osp
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import argparse

def pickle2json(preds_pickle, result_path):
    # Load pickle
    with open(preds_pickle, 'rb') as file:
        data = pickle.load(file)

    result = list()
    for img_id, d in tqdm(enumerate(data), total=len(data), desc="Processing"):
        img_path = d['img_path']
        bboxes = d['pred_instances']['bboxes'].detach().numpy().tolist()
        labels = d['pred_instances']['labels'].detach().numpy().tolist()
        scores = d['pred_instances']['scores'].detach().numpy().tolist()
        size = len(bboxes)
        for i in range(size):
            x,y,x2,y2 = bboxes[i]
            # w = x2-x
            # h = 
            result.append({
                'image_id': img_id+1,
                'category_id': labels[i]+1,
                'image_path': img_path,
                'bbox': [x,y,x2-x,y2-y],
                'score': scores[i]
            })
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

def main(preds_pickle, result_path):
    pickle2json(preds_pickle, result_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert pickle to JSON")
    parser.add_argument("preds_pickle", type=str, help="Path to the pickle file")
    parser.add_argument("result_path", type=str, help="Path to save the JSON result")
    args = parser.parse_args()
    
    main(args.preds_pickle, args.result_path)
    print('Done')