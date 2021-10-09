from genericpath import isfile
from math import isnan
from operator import le, pos
from numpy.lib.type_check import imag
import torch
from torch import tensor
from torch.utils import data
from torchvision.datasets.vision import VisionDataset
from numpy.lib.polynomial import roots
from pathlib import Path
from util.misc import get_local_rank, get_local_size
import os
import datasets.transforms as T
from PIL import Image
import cv2
import numpy as np
from numpy.linalg import inv
from skimage.measure import regionprops
import json
import pandas as pd


class ScanNetDetection(VisionDataset):
    def __init__(self, data_root_path, data_list, tsv_map, transforms, target_transform=None, transform=None,
                 cache_mode=False, local_rank=0, local_size=1, seq=3):
        super(ScanNetDetection, self).__init__(
            data_root_path, transform, target_transform)

        self.data_root = data_root_path
        self.seq_size = seq

        text_file = open(data_list, "r")
        self.scene_list = text_file.read().splitlines()
        text_file.close()

        # get the number of data in each scene
        self.scene_data_num = []
        for scene in self.scene_list:
            scene_img_path = os.path.join(data_root_path, scene, "color")
            self.scene_data_num.append(len(os.listdir(scene_img_path)))
        self._transforms = transforms
        self.test_target = {'labels': torch.tensor([1]), 'image_id': torch.tensor([0]), 'area': torch.tensor(
            [0]), 'iscrowd': torch.tensor([0]), 'orig_size': torch.tensor([1296, 968]), 'size': torch.tensor([1296, 968])}

        self.label_nyu_map = pd.read_csv(tsv_map, sep='\t', header=0,
                                         usecols=["raw_category", "nyu40id"])

    def convert_label_map(self, label_json):
        with open(label_json) as f:
            anno = json.load(f)
        return anno["segGroups"]

    def get_image(self, path):
        return Image.open(path).convert('RGB')

    def get_target(self, path, id, scene_aggre_label):
        masks = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        regions = regionprops(masks)
        bboxes = torch.zeros((len(regions), 4))
        labels = torch.zeros((len(regions)), dtype=torch.long)
        masks_tensor = torch.zeros(
            (len(regions), masks.shape[0], masks.shape[1]))
        area = torch.zeros((len(regions)))
        iscrowd = torch.zeros((len(regions)))
        orig_size = torch.zeros((len(regions)))
        size = torch.zeros((len(regions)))
        for i, pros in enumerate(regions):
            bboxes[i] = torch.tensor(pros.bbox)

            labels[i] = torch.tensor(
                self.label_nyu_map.loc[self.label_nyu_map['raw_category'] == scene_aggre_label[pros.label-1]["label"], 'nyu40id'].iloc[0])
            masks_tensor[i] = torch.tensor((masks == pros.label).astype(int))
            area[i] = torch.tensor(pros.area)
            orig_size = torch.tensor((masks.shape[0], masks.shape[1]))
            size = orig_size

        target = {
            'boxes': bboxes,
            'labels': labels,
            'masks': masks_tensor,
            'image_id': torch.tensor([id]),
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': orig_size,
            'size': size
        }

        return target

    def get_image_coord_target_seq(self, scene, index_inScene, index):
        img_root = os.path.join(
            self.data_root, self.scene_list[scene], "color")
        mask_root = os.path.join(
            self.data_root, self.scene_list[scene], "instance-filt")
        depth_root = os.path.join(
            self.data_root, self.scene_list[scene], "depth")
        intrinsic_path = os.path.join(
            self.data_root, self.scene_list[scene], "intrinsic", "intrinsic_depth.txt")
        pose_root = os.path.join(
            self.data_root, self.scene_list[scene], "pose")

        scene_aggre_label = self.convert_label_map(os.path.join(
            self.data_root, self.scene_list[scene], self.scene_list[scene]+"_vh_clean.aggregation.json"))

        img_seq = torch.zeros((self.seq_size, 3, 640, 480))
        coord_seq = torch.zeros((self.seq_size, 3, 640, 480))
        target_seq = []
        for i in range(self.seq_size):
            img_frame = self.get_image(os.path.join(
                img_root, str(index_inScene-i)+".jpg"))
            target_frame = self.get_target(os.path.join(
                mask_root, str(index_inScene-i)+".png"), index-i, scene_aggre_label)
            depth_path = os.path.join(
                depth_root, str(index_inScene-i)+".png")
            intrinsic_path = intrinsic_path

            ######################################## some bug in data set fuck ###################################
            pose_path = os.path.join(pose_root, str(index_inScene-i)+".txt")

            pose = np.loadtxt(pose_path)
            # try to find available pose
            start_add_i = 1
            while (np.isinf(pose).any() or np.isnan(pose).any()):
                if(index_inScene-i+start_add_i < self.scene_data_num[scene]):
                    pose_path = os.path.join(pose_root, str(
                        index_inScene-i+start_add_i)+".txt")
                    pose = np.loadtxt(pose_path)
                elif(index_inScene-i-start_add_i >= 0):
                    pose_path = os.path.join(pose_root, str(
                        index_inScene-i-start_add_i)+".txt")
                    pose = np.loadtxt(pose_path)
                start_add_i += 1
            ######################################## some bug in data set fuck ###################################

            coord_frame = self.get_coord(
                depth_path, intrinsic_path, pose)
            img_frame, coord_frame, target_frame = self._transforms(
                img_frame, coord_frame, target_frame)
            coord_seq[i] = coord_frame
            img_seq[i] = img_frame
            target_seq.append(target_frame)

        return img_seq, coord_seq, target_seq

    def get_coord(self, depth_path, intrinsic_path, pose):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        intrinsic = np.loadtxt(intrinsic_path)

        u = range(0, depth_img.shape[1])
        v = range(0, depth_img.shape[0])

        u, v = np.meshgrid(u, v)
        u = u.astype(float)
        v = v.astype(float)

        Z = depth_img.astype(float) / 1000
        X = (u - intrinsic[0, 2]) * Z / intrinsic[0, 0]
        Y = (v - intrinsic[1, 2]) * Z / intrinsic[1, 1]

        X = np.ravel(X)
        Y = np.ravel(Y)
        Z = np.ravel(Z)

        # valid = Z > 0

        # X = X[valid]
        # Y = Y[valid]
        # Z = Z[valid]

        coordinate = np.vstack((X, Y, Z, np.ones(len(X))))
        coordinate = np.dot(pose, coordinate)
        coordinate = np.transpose(coordinate[:3])
        coordinate = coordinate.reshape(
            depth_img.shape[0], depth_img.shape[1], -1)
        return torch.tensor(coordinate.transpose())

    def get_coord_seq(self, scene, index_inScene):
        depth_root = os.path.join(
            self.data_root, self.scene_list[scene], "depth")
        intrinsic_path = os.path.join(
            self.data_root, self.scene_list[scene], "intrinsic", "intrinsic_depth.txt")
        pose_root = os.path.join(
            self.data_root, self.scene_list[scene], "pose")

        coord_seq = torch.zeros((self.seq_size, 3, 640, 480))
        for i in range(self.seq_size):
            depth_path = os.path.join(
                depth_root, str(index_inScene-i)+".png")
            intrinsic_path = intrinsic_path
            pose_path = os.path.join(
                pose_root, str(index_inScene-i)+".txt")
            coord_seq[i] = torch.from_numpy(self.get_coord(
                depth_path, intrinsic_path, pose_path))

        return coord_seq

    def getSceneId(self, index):
        data_num_sum = 0
        for i, data_num in enumerate(self.scene_data_num):
            data_num -= (self.seq_size-1)
            data_num_sum += data_num
            if(data_num_sum > index):
                return i, index - (data_num_sum-data_num) + (self.seq_size-1)

    def __getitem__(self, index):
        scene, img_id_in_scene = self.getSceneId(index)
        imgs, coords, targets = self.get_image_coord_target_seq(
            scene, img_id_in_scene, index)
        # coords = self.get_coord_seq(scene, img_id_in_scene)

        return imgs, coords, targets

    def __len__(self):
        return sum(self.scene_data_num)-((self.seq_size-1)*len(self.scene_data_num))


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [(480, 640)]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.RandomSelect(
            T.RandomResize(scales),
            # T.Compose([
            #     T.RandomResize([400, 500, 600]),
            #     T.RandomSizeCrop(384, 600),
            #     T.RandomResize(scales, max_size=1333),
            # ])
            # ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([(480, 640)]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.scannet_path)
    assert root.exists(), f'provided ScanNet path {root} does not exist'
    # PATHS = {
    #     "train": (root / "color", root / "depth", root / "intrinsic", root / "pose", root / "instance-filt", root / "scene0415_01_vh_clean.aggregation.json", root / "scannetv2-labels.combined.tsv"),
    #     "val": (root / "color", root / "depth", root / "intrinsic", root / "pose", root / "instance-filt", root / "scene0415_01_vh_clean.aggregation.json", root / "scannetv2-labels.combined.tsv")
    # }
    PATHS = {
        "train": (root / "train.txt", root / "scannetv2-labels.combined.tsv"),
        "val": (root / "val.txt", root / "scannetv2-labels.combined.tsv")
    }

    data_list, tsv_map = PATHS[image_set]
    # image_folder, depth_folder, intrinsic_folder, pose_folder, mask_folder, label_json, tsv_map = PATHS[
    #     image_set]
    dataset = ScanNetDetection(root, data_list, tsv_map, transforms=make_coco_transforms(image_set),
                               cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    # dataset = ScanNetDetection(image_folder, depth_folder, intrinsic_folder, pose_folder, mask_folder, label_json, tsv_map, transforms=make_coco_transforms(image_set),
    #                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    return dataset
