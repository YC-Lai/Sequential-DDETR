# ------------------------------------------------------------------------
# Sequential DDETR
# Copyright (c) 2022 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torch import multiprocessing
import torch.utils.data
import torch.nn.functional as F
from .torchvision_datasets import CocoDetection
from torchvision.datasets.vision import VisionDataset

from .coco import build as build_coco
from .scannet import build as build_scan_net
from util.box_ops import box_cxcywh_to_xyxy
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from joblib import Parallel, delayed


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def process_file(ds, img_idx):
    dataset = {"images": [], "annotations": []}
    targets = ds.get_item_for_cocoApi(img_idx)
    image_id = targets["image_id"].item()
    img_dict = {}
    img_dict["id"] = image_id
    img_h, img_w = targets['orig_size'][0].item(), targets['orig_size'][1].item()
    img_dict["height"] = img_h
    img_dict["width"] = img_w
    dataset["images"].append(img_dict)
    bboxes = box_cxcywh_to_xyxy(targets["boxes"])
    bboxes = convert_to_xywh(bboxes)
    bboxes *= torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
    bboxes = bboxes.tolist()
    labels = targets["labels"].tolist()
    areas = targets["area"].tolist()
    iscrowd = targets["iscrowd"].tolist()
    if "masks" in targets:
        masks = targets["masks"]
        masks = F.interpolate(masks.unsqueeze(1).float(), size=tuple([img_h, img_w]), mode="nearest").byte()
        # make masks Fortran contiguous for coco_mask
        masks = masks.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    if "keypoints" in targets:
        keypoints = targets["keypoints"]
        keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
    num_objs = len(bboxes)
    for i in range(num_objs):
        ann = {}
        ann["image_id"] = image_id
        ann["bbox"] = bboxes[i]
        ann["category_id"] = labels[i]
        ann["area"] = areas[i]
        ann["iscrowd"] = iscrowd[i]
        if "masks" in targets:
            rle = coco_mask.encode(masks[i].permute(1, 2, 0).numpy().astype("uint8", order="F"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            ann["segmentation"] = rle
        if "keypoints" in targets:
            ann["keypoints"] = keypoints[i]
            ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
        dataset["annotations"].append(ann)
    return dataset


def convert_to_coco_api_parallel(ds):    
    coco_ds = COCO()
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    
    num_cpu = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cpu)(delayed(process_file)(ds, i) for i in range(ds.get_len_for_cocoApi()))

    for result in results:
        # images
        dataset["images"].extend(result["images"])
        # annotations
        for ann in result["annotations"]:
            ann['id'] = ann_id
            categories.add(ann['category_id'])
            ann_id += 1
        dataset["annotations"].extend(result["annotations"])
        
    # categories
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    return dataset


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'ScanNet':
        return build_scan_net(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
