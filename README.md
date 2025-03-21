# LDXray: A Dataset for X-ray Prohibited Items Detection

## Introduction
LDXray is a large-scale dual-view dataset collected from real security scenes in transportation hubs. It contains a wide range of prohibited items categorized into 12 different classes.

![Sample image from LDXray dataset](dataset.png)

<!-- ## Experimental Results
We have extensively evaluated existing representative detection models and established baselines. Additionally, we have implemented several different strategies for utilizing side views, including:

### Baselines
![Baseline results](baseline.png)

### Data Augmentation
![Data augmentation methods](sideview.png)

### Contrastive Learning
![Contrastive learning techniques](contrastive.png)

### Feature Fusion
![Feature fusion methods](fusion.png) -->

## Dataset Download
The dataset and code can be accessed from the following links:

- **LDXray Dataset**: [Kaggle](https://www.kaggle.com/datasets/yuzheguocs/LDXray) | [BaiduNetDisk](https://pan.baidu.com/s/1YyUcBe7usxMUb1UyzTorgQ?pwd=m5pl)

## Code
- **Installation**
Please follow the official environment setup guide of [MMdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html).

- **Side-View pseudo-label**
Use ld-tools/sail2bbox.py to get pseudo-label for side-view data , you can define the target_category_ids to any classes you need.The input_json_path should be the main view annotation file.

- **Main-View Slicing**
Use ld-tools/gen-main-view-slice-json-wider.py to get sliced main-view data for training sliced images detector.You can extract a subset of spesific classes first to get spesific classes dataset.


- **Config**
Update the config file of dataset in ld-config/_base_/datasets. \
Update the num_classes in main config when taining specific classes of side-view model and sliced image model.

- **Train**
```shell
python tools/train.py ld-config/ld-retinanet/main-view-retinanet.py
python tools/train.py ld-config/ld-retinanet/sliced-retinanet.py
python tools/train.py ld-config/ld-retinanet/side-view-retinanet.py
```

- **Side-view inference**
```shell
python tools/test.py ld-config/ld-retinanet/side-view-retinanet.py work_dirs/side-view-retinanet/epoch_12.pth --out side-view-retinanet.pkl
```

- **pkl2json**
Use ld-tools/pkl2jon.py to convert the side-view-retinanet.pkl to side-view-retinanet.pkl. The categoriy_id will be start from 1 ,you should update the categoriy_id by yourself.

- **Side-view to main-view**
Use ld-tools/dv-proposal.py to get main-view region proposals from side-view prediction.

- **Auxiliary Prediction**
Update the test config in sliced-retinanet.py(Using the new json of Step 'side-view to main-view' ) to inference main-view region proposals predicted by side-view model.
```shell
python tools/test.py ld-config/ld-retinanet/sliced-retinanet.py work_dirs/side-view-retinanet/epoch_12.pth --out sliced-retinanet.pkl
```


## Citation
If you use LDXray in your research, please consider citing:

```bibtex
@inproceedings{tao2024dual,
  title={Dual-view X-ray Detection: Can AI Detect Prohibited Items from Dual-view X-ray Images like Humans?},
  author={Tao, Renshuai and Wang, Haoyu and Guo, Yuzhe and Chen, Hairong and Zhang, Li and Liu, Xianglong and Wei, Yunchao and Zhao, Yao},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  organization={IEEE}
}
```

## Contact
For any questions, please contact [Renshuai Tao](https://rstao-bjtu.github.io/).

---

