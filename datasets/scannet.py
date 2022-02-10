from collections import OrderedDict
from unittest import result
import torch
from torchvision.datasets.vision import VisionDataset
import datasets.utils as utils
import datasets.transforms as T
from pathlib import Path


class ScanNet(VisionDataset):
    """ScanNet dataset http://www.scan-net.org/
    Keyword arguments:
    - root_dir (``string``): Path to the base directory of the dataset
    - scene_file (``string``): Path to file containing a list of scenes to be loaded
    - transform (``callable``, optional): A function/transform that takes in a 
    PIL image and returns a transformed version of the image. Default: None.
    - target_transform (``callable``, optional): A function/transform that takes 
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its path.
    By default, ``default_loader`` is used.
    - color_mean (``list``): A list of length 3, containing the R, G, B channelwise mean.
    - color_std (``list``): A list of length 3, containing the R, G, B channelwise standard deviation.
    - load_depth (``bool``): Whether or not to load depth images (architectures that use depth 
    information need depth to be loaded).
    - seg_classes (``string``): The palette of classes that the network should learn.
    """

    def __init__(self, root_dir, scene_file, tsv_map, mode='train', transforms=None, loader=utils.scannet_loader, load_mode='rgb', seg_classes='nyu40', num_frames=3):
        super(ScanNet, self).__init__(root_dir)
        self.transforms = transforms
        self.root_dir = root_dir
        self.mode = mode
        self.loader = loader
        self.length = 0
        self.load_mode = load_mode
        self.seg_classes = seg_classes
        self.num_frames = num_frames
        # color_encoding has to be initialized AFTER seg_classes
        self.color_encoding = self.get_color_encoding()
        self.preprocessing_map = utils.get_preprocessing_map(tsv_map)

        # Get the list of scenes, and generate paths
        scene_list = []
        try:
            with open(scene_file, 'r') as f:
                scenes = f.readlines()
                for scene in scenes:
                    scene = scene.strip().split()
                    scene_list.append(scene[0])
        except Exception as e:
            raise e

        # Get train data and labels filepaths
        self.data = []
        self.depth = []
        self.labels = []
        self.instances = []
        self.poses = []
        self.intrinsic = []
        self.num_sceneData = []
        for scene in scene_list:
            color_images, depth_images, labels, instances, poses, intrinsic = utils.get_filenames_scannet(
                self.root_dir, scene)
            self.data += color_images
            self.depth += depth_images
            self.labels += labels
            self.instances += instances
            self.poses += poses
            self.intrinsic.append(intrinsic)
            self.length += len(color_images)
            self.num_sceneData.append(len(color_images))

    def get_len_for_cocoApi(self):
        """ Returns the length of the dataset. """
        return self.length

    def get_item_for_cocoApi(self, index):

        data_path, label_path, instance_path = self.data[index], self.labels[index], self.instances[index]
        path_set = dict({'rgb': data_path, 'depth': None, 'pose': None,
                        'label': label_path, 'instance': instance_path, 'intrinsic': None})
        rgb, depth, coords, target = self.loader(
            index, path_set, 'load_target_only', self.preprocessing_map, self.seg_classes)
        _, _, _, target = self.transforms(rgb, depth, coords, target)

        return target

    def get_data_Id(self, index):
        """ Cover the sequential id to the dataset id. """
        sum = 0
        for i, num_data in enumerate(self.num_sceneData):
            num_data -= self.num_frames - 1
            sum += num_data
            if sum > index:
                return i, index + (i + 1) * (self.num_frames - 1)

    def get_data_sequence(self, sceneId, data_start_index):
        data_seq = []
        target_seq = []
        intrinsic_path = self.intrinsic[sceneId]
        for i in range(self.num_frames):
            index = data_start_index - i
            data_path, depth_path, label_path, instance_path, pose_path = self.data[
                index], self.depth[index], self.labels[index], self.instances[index], self.poses[index]
            path_set = dict({'rgb': data_path, 'depth': depth_path, 'pose': pose_path,
                            'label': label_path, 'instance': instance_path, 'intrinsic': intrinsic_path})
            rgb, depth, coords, target = self.loader(
                index, path_set, self.load_mode, self.preprocessing_map, self.seg_classes)
            rgb, depth, coords, target = self.transforms(rgb, depth, coords, target)

            # concatenate all sources into data
            if self.load_mode == 'rgb':
                data = rgb
            elif self.load_mode == 'depth':
                data = torch.cat((rgb, depth), 0)
            else:
                assert self.load_mode == 'coords'
                data = torch.cat((rgb, depth, coords), 0)

            data_seq.append(data)
            target_seq.append(target)

        return torch.stack(data_seq, dim=0), target_seq

    def __getitem__(self, index):
        """ Returns element at index in the dataset.
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth of the image
        """
        scene_id, data_start_id = self.get_data_Id(index)
        data, targets = self.get_data_sequence(scene_id, data_start_id)

        return data, targets

    def __len__(self):
        """ Returns the sequential length of the dataset. """
        return self.length - (self.num_frames - 1) * len(self.num_sceneData)

    def get_color_encoding(self):
        if self.seg_classes.lower() == 'nyu40':
            """Color palette for nyu40 labels """
            return OrderedDict([
                ('unlabeled', (0, 0, 0)),
                ('wall', (174, 199, 232)),
                ('floor', (152, 223, 138)),
                ('cabinet', (31, 119, 180)),
                ('bed', (255, 187, 120)),
                ('chair', (188, 189, 34)),
                ('sofa', (140, 86, 75)),
                ('table', (255, 152, 150)),
                ('door', (214, 39, 40)),
                ('window', (197, 176, 213)),
                ('bookshelf', (148, 103, 189)),
                ('picture', (196, 156, 148)),
                ('counter', (23, 190, 207)),
                ('blinds', (178, 76, 76)),
                ('desk', (247, 182, 210)),
                ('shelves', (66, 188, 102)),
                ('curtain', (219, 219, 141)),
                ('dresser', (140, 57, 197)),
                ('pillow', (202, 185, 52)),
                ('mirror', (51, 176, 203)),
                ('floormat', (200, 54, 131)),
                ('clothes', (92, 193, 61)),
                ('ceiling', (78, 71, 183)),
                ('books', (172, 114, 82)),
                ('refrigerator', (255, 127, 14)),
                ('television', (91, 163, 138)),
                ('paper', (153, 98, 156)),
                ('towel', (140, 153, 101)),
                ('showercurtain', (158, 218, 229)),
                ('box', (100, 125, 154)),
                ('whiteboard', (178, 127, 135)),
                ('person', (120, 185, 128)),
                ('nightstand', (146, 111, 194)),
                ('toilet', (44, 160, 44)),
                ('sink', (112, 128, 144)),
                ('lamp', (96, 207, 209)),
                ('bathtub', (227, 119, 194)),
                ('bag', (213, 92, 176)),
                ('otherstructure', (94, 106, 211)),
                ('otherfurniture', (82, 84, 163)),
                ('otherprop', (100, 85, 144)),
            ])
        elif self.seg_classes.lower() == 'scannet20':
            return OrderedDict([
                ('unlabeled', (0, 0, 0)),
                ('wall', (174, 199, 232)),
                ('floor', (152, 223, 138)),
                ('cabinet', (31, 119, 180)),
                ('bed', (255, 187, 120)),
                ('chair', (188, 189, 34)),
                ('sofa', (140, 86, 75)),
                ('table', (255, 152, 150)),
                ('door', (214, 39, 40)),
                ('window', (197, 176, 213)),
                ('bookshelf', (148, 103, 189)),
                ('picture', (196, 156, 148)),
                ('counter', (23, 190, 207)),
                ('desk', (247, 182, 210)),
                ('curtain', (219, 219, 141)),
                ('refrigerator', (255, 127, 14)),
                ('showercurtain', (158, 218, 229)),
                ('toilet', (44, 160, 44)),
                ('sink', (112, 128, 144)),
                ('bathtub', (227, 119, 194)),
                ('otherfurniture', (82, 84, 163)),
            ])


def make_transforms(image_set):
    # Mean color, standard deviation (R, G, B)
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]

    normalize = T.Compose([
        T.Normalize(color_mean, color_std)
    ])

    scales = [(240, 320)]  # (h, w)

    if image_set == 'train':
        return T.Compose([
            T.RandomResize(scales),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize(scales),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.scannet_path)
    assert root.exists(), f'provided ScanNet path {root} does not exist'

    PATHS = {
        "train": (root / "train.txt", root / "scannet-labels.combined.tsv"),
        "val": (root / "val.txt", root / "scannet-labels.combined.tsv")
    }
    scene_list, tsv_map = PATHS[image_set]
    dataset = ScanNet(root, scene_list, tsv_map, mode=image_set, transforms=make_transforms(
        image_set), loader=utils.scannet_loader, load_mode=args.load_mode, seg_classes=args.seg_classes, num_frames=args.num_frames)
    return dataset
