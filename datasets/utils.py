import os
from PIL import Image
from natsort import natsorted
import numpy as np
import imageio
import csv

import torch
from torch._C import StringType
import torchvision.transforms as transforms


def get_filenames_scannet(base_dir, scene_id):
    """Helper function that returns a list of scannet images and the corresponding 
    segmentation labels, given a base directory name and a scene id.

    Args:
    - base_dir (``string``): Path to the base directory containing ScanNet data, in the 
    directory structure specified in https://github.com/angeladai/3DMV/tree/master/prepare_data
    - scene_id (``string``): ScanNet scene id

    """

    if not os.path.isdir(base_dir):
        raise RuntimeError('\'{0}\' is not a directory.'.format(base_dir))

    color_images = []
    depth_images = []
    labels = []
    instances = []
    poses = []
    intrinsic = os.path.join(base_dir, scene_id, "intrinsic", "intrinsic_depth.txt")

    # for debug only, can be ignored
    invalid_list = []

    # Explore the directory tree to get a list of all files
    for path, _, files in os.walk(os.path.join(base_dir, scene_id, 'color')):
        files = natsorted(files)
        for file in files:
            filename, _ = os.path.splitext(file)
            depthfile = os.path.join(base_dir, scene_id, 'depth', filename + '.png')
            labelfile = os.path.join(base_dir, scene_id, 'label-filt', filename + '.png')
            instancefile = os.path.join(base_dir, scene_id, 'instance-filt', filename + '.png')
            posefile = os.path.join(base_dir, scene_id, 'pose', filename + '.txt')
            # Add this file to the list of train samples, only if its corresponding depth and label
            # files exist.
            if os.path.exists(depthfile) and os.path.exists(labelfile) and os.path.exists(instancefile) and os.path.exists(posefile):
                if is_pose_invalid(posefile):
                    invalid_list.append(filename)
                    continue
                color_images.append(os.path.join(base_dir, scene_id, 'color', filename + '.jpg'))
                depth_images.append(depthfile)
                labels.append(labelfile)
                instances.append(instancefile)
                poses.append(posefile)

    # Assert that we have the same number of color, depth images as labels
    assert (len(color_images) == len(depth_images) == len(labels) == len(instances) == len(poses))

    return color_images, depth_images, labels, instances, poses, intrinsic


def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (``string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        def name_cond(filename): return True
    else:
        def name_cond(filename): return name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        def ext_cond(filename): return True
    else:
        def ext_cond(filename): return filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def is_pose_invalid(pose_path):
    pose = np.loadtxt(pose_path)
    return (np.isinf(pose).any() or np.isnan(pose).any())


def load_rgb(data_path: StringType):
    """Loads a sample RGB image given their path as PIL images.

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - color_mean (``list``): R, G, B channel-wise mean
    - color_std (``list``): R, G, B channel-wise stddev

    Returns the image as PIL images.

    """

    # Load image
    data = np.array(imageio.imread(data_path))
    # Reshape data from H x W x C to C x H x W
    data = np.moveaxis(data, 2, 0)

    return torch.Tensor(data.astype(np.float32))


def load_depth(depth_path: StringType):
    """Loads a sample depth image given their path as PIL images.

    Keyword arguments:
    - depth_path (``string``): The filepath to the depth png.

    Returns the depth as PIL images.

    """

    # Load depth
    depth = torch.Tensor(np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0)
    depth = torch.unsqueeze(depth, 0)

    return depth


def load_depth_coords(pose_path: StringType, depth_path: StringType, intrinsic_path: StringType, load_mode: StringType):
    """Loads a sample depth image given their path as PIL images, and convert depth image to 3D coordinates.

    Keyword arguments:
    - pose_path (``string``): The filepath to the camera pose.
    - depth_path (``string``): The filepath to the depth png.
    - intrinsic (``string``): The filepath to the camera intrinsic.

    Returns the depth and the coordinates as PIL images.

    """

    assert load_mode == "coords" 
    
    # Load intrinsic
    intrinsic = torch.Tensor(np.loadtxt(intrinsic_path, dtype=np.float32))
    
    # Load depth
    depth = torch.Tensor(np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0)
    h, w = depth.shape[0], depth.shape[1] 

    # Load pose
    pose = torch.Tensor(np.loadtxt(pose_path, dtype=np.float32))

    # transform
    u = torch.arange(0, w)
    v = torch.arange(0, h)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')

    X = (grid_u - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    Y = (grid_v - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    X, Y, Z = torch.ravel(X), torch.ravel(Y), torch.ravel(depth)

    homo_coords = pose @ torch.stack((X, Y, Z, torch.ones_like(X)), dim=0)
    coordinates = homo_coords[:3] / homo_coords[3]

    return torch.unsqueeze(depth, 0), coordinates.reshape(3, h, w)


def load_label(label_path: StringType, preprocessing_map, seg_classes='nyu40'):
    """Loads a label image given their path as PIL images. (nyu40 classes)

    Keyword arguments:
    - label_path (``string``): The filepath to the ground-truth image.
    - preprocessing_map (``dict``): The map to convert raw category to nyu40
    - seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')

    Returns the label as PIL images.

    """

    # Load label
    label = torch.Tensor(np.array(imageio.imread(label_path)).astype(np.uint8))
    label = rawCategory_to_nyu40(label, preprocessing_map)
    if seg_classes.lower() == 'scannet20':
        # Remap classes from 'nyu40' to 'scannet20'
        label = nyu40_to_scannet20(label)
    return label


def load_target(image_id: int, label_path: StringType, instance_path: StringType, preprocessing_map: dict, seg_classes='nyu40'):
    """Loads a label image given their path as PIL images. (nyu40 classes)

    Keyword arguments:
    - label_path (``string``): The filepath to the ground-truth image.
    - instance_path (``string``): The filepath to the ground-truth image.
    - preprocessing_map (``dict``): The map to convert raw category to nyu40
    - id (``int``): The image index.
    - seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')

    Returns the label as PIL images.

    """

    # Load label
    label_masks = load_label(label_path, preprocessing_map, seg_classes)

    # Load instance
    instance_masks = torch.Tensor(np.array(imageio.imread(instance_path)))

    # Generate bbox and masks
    insts = np.unique(instance_masks)

    bboxes = []
    labels = []
    masks = []
    area = []
    for inst in insts: 
        regions = torch.where(instance_masks == inst)
        label = label_masks[regions][0].to(torch.int64)
        if label in [0]:
            continue
        bbox = torch.tensor([torch.min(regions[1]), torch.min(regions[0]),
                            torch.max(regions[1]), torch.max(regions[0])], dtype=torch.int)
        mask = (instance_masks == inst).clone().detach().to(torch.int)

        bboxes.append(bbox)
        labels.append(label)
        masks.append(mask)
        area.append(torch.sum(mask))
    bboxes = torch.stack(bboxes, 0)
    labels = torch.stack(labels, 0)
    masks = torch.stack(masks, 0)
    area = torch.stack(area, 0)
    size = orig_size = torch.tensor((label_masks.shape[0], label_masks.shape[1]))

    target = {
        'image_id': torch.tensor(image_id),
        'insts': torch.tensor(insts),
        'boxes': bboxes,
        'labels': labels,
        'masks': masks,
        'area': area,
        'orig_size': orig_size,
        'size': size,
        'iscrowd': torch.zeros_like(labels)
    }

    return target


def scannet_loader(image_id: int, path_set: dict, load_mode: StringType, preprocessing_map, seg_classes='nyu40'):
    """Loads a sample and label image given their path as PIL images. (nyu40 classes)

    Keyword arguments:
    - path_set (``dict``): The dict of filepath.
    - load_mode (``string``): The mode to data loading type.
    - color_mean (``list``): R, G, B channel-wise mean
    - color_std (``list``): R, G, B channel-wise stddev
    - seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')

    Returns the image and the label as PIL images.

    """

    # Load target
    target = load_target(image_id, path_set['label'], path_set['instance'], preprocessing_map, seg_classes)

    # Load data
    if load_mode == 'rgb':
        rgb = load_rgb(path_set['rgb'])
        return rgb, None, None, target
    
    elif load_mode == 'depth':
        rgb = load_rgb(path_set['rgb'])
        depth = load_depth(path_set['depth'])
        return rgb, depth, None, target
    
    elif load_mode == 'coords':
        rgb = load_rgb(path_set['rgb'])
        depth, coords = load_depth_coords(path_set['pose'], path_set['depth'], path_set['intrinsic'], load_mode)
        return rgb, depth, coords, target
        
    else:
        assert load_mode == 'load_target_only'
        return None, None, None, target


def get_preprocessing_map(label_map):
    mapping = dict()
    with open(label_map) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if type(row["nyu40id"]) != int:
                mapping[int(row["id"])] = 40
            else:
                mapping[int(row["id"])] = int(row["nyu40id"])
    return mapping


def rawCategory_to_nyu40(label, map: dict):
    """Remap a label image from the 'raw_category' class palette to the 'nyu40' class palette """

    nyu40_label = label
    keys = torch.unique(label).numpy().astype(np.int)
    for key in keys:
        if key != 0:
            nyu40_label[label == key] = map[key]
    return nyu40_label


def nyu40_to_scannet20(label):
    """Remap a label image from the 'nyu40' class palette to the 'scannet20' class palette """

    # Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
    # Because, these classes from 'nyu40' are absent from 'scannet20'. Our label files are in
    # 'nyu40' format, hence this 'hack'. To see detailed class lists visit:
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt ('nyu40' labels)
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt ('scannet20' labels)
    # The remaining labels are then to be mapped onto a contiguous ordering in the range [0,20]

    # The remapping array comprises tuples (src, tar), where 'src' is the 'nyu40' label, and 'tar' is the
    # corresponding target 'scannet20' label
    remapping = [(0, 0), (13, 0), (15, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (25, 0), (26, 0), (27, 0), (29, 0), (30, 0),
                 (31, 0), (32, 0), (35, 0), (37, 0), (38, 0), (40, 0), (14, 13), (16, 14), (24, 15), (28, 16), (33, 17), (34, 18), (36, 19), (39, 20)]
    for src, tar in remapping:
        label[label == src] = tar
    return label


def create_label_image(output, color_palette):
    """Create a label image, given a network output (each pixel contains class index) and a color palette.

    Args:
    - output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer, 
    corresponding to the class label of that pixel.
    - color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
    """

    label_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_palette):
        label_image[output == idx] = color
    return label_image


def remap(image, old_values, new_values):
    assert isinstance(image, Image.Image) or isinstance(
        image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
    assert type(new_values) is tuple, "new_values must be of type tuple"
    assert type(old_values) is tuple, "old_values must be of type tuple"
    assert len(new_values) == len(
        old_values), "new_values and old_values must have the same length"

    # If image is a PIL.Image convert it to a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Replace old values by the new ones
    tmp = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values
        # equal to 0
        if new != 0:
            tmp[image == old] = new

    return Image.fromarray(tmp)


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

            w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

            propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

            w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq
