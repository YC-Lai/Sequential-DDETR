"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from types import coroutine
from typing import Iterable

import PIL
import torch
import util.misc as utils
from datasets.data_prefetcher import data_prefetcher
from util import box_ops
import torch.nn.functional as F
import matplotlib.pyplot as plt

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# def plot_results(pil_img, prob, boxes):
def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        print(ymax-ymin)
        ax.add_patch(plt.Rectangle((ymin, xmin), ymax - ymin, xmax - xmin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        # ax.text(xmin, ymin, text, fontsize=15,
        #         bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("test.png")
    
# plot_results(im, scores, boxes)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_ops.box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(b.get_device())
    return b

@torch.no_grad()
def test(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
             data_loader: Iterable, device: torch.device,
             comet_logger=None, val_step=0):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = '\nTest:'
    print_freq = len(data_loader)
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, coords, depths, targets  = prefetcher.next()

    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())

    # coco_evaluator = ScannetEvaluator(base_ds, iou_types)
    # # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    # for samples, coords, targets in metric_logger.log_every(data_loader, 10, header):
    for _ in metric_logger.log_every(range(len(data_loader)), 10, header):
        outputs = model(samples, coords, depths)


        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}


        
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # # add comet logger
        # comet_logger.log_metric("val_class error", loss_dict_reduced['class_error'], step=val_step)
        # comet_logger.log_metric("val_bbox loss", loss_dict_reduced['loss_bbox'], step=val_step)
        # comet_logger.log_metric("val_mask loss", loss_dict_reduced['loss_dice'], step=val_step)

        # samples, coords, targets = prefetcher.next()
        # val_step += 1   

        # # flatten targets into the list of (batch_size * n_frames)
        flat_targets = [t for target in targets for t in target]

        orig_target_sizes = torch.stack([t["orig_size"] for t in flat_targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # if 'segm' in postprocessors.keys():
        target_sizes = torch.stack([t["size"] for t in flat_targets], dim=0)
        results = postprocessors['segm'](results, outputs, target_sizes, target_sizes)


        outputs["pred_logits"] = outputs["pred_logits"].reshape(3, -1, outputs["pred_logits"].shape[-1])
        outputs["pred_boxes"] = outputs["pred_boxes"].reshape(3, -1, outputs["pred_boxes"].shape[-1])
        
        probas = outputs["pred_logits"].softmax(-1)[:, :, 1:]

        keep = probas.max(-1).values > 0.2

        bboxs_scaled = []
        mask_scaled = []
        prob_scaled = []
        for i in range(3):
            target_size = flat_targets[i]["orig_size"]
            bboxs_scaled.append(rescale_bboxes(outputs["pred_boxes"][i, keep[i]], target_size))
            # print(outputs["pred_masks"][i].shape)
            # output_masks = F.interpolate(outputs["pred_masks"][i], size=target_size, mode="bilinear", align_corners=False)
            mask_scaled.append(F.interpolate(outputs["pred_masks"][i, keep[i]].unsqueeze(0), size=tuple(target_size.tolist()), mode="bilinear", align_corners=False).byte())
            prob_scaled.append(probas[i, keep[i]])
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # unorm = UnNormalize(mean, std)
        # image = unorm(samples.tensors)
        # print(image.shape)

        for i in range(3):
            image = PIL.Image.open(flat_targets[i]["image_name"])
            plot_results(image, prob_scaled[i], bboxs_scaled[i])
        exit()
        # print(samples.tensors.shape)

        # print(outputs["pred_logits"].shape)
        # print(outputs["pred_boxes"].shape)
        # print(outputs["pred_masks"].shape)

        # res = {target['image_id'].item(): output for target, output in zip(flat_targets, results)}

        # print(results[0]["masks"].shape)
        # print(results[0]["label"])
        
        # if coco_evaluator is not None:
        #     print("coco")
        #     coco_evaluator.update(res)
        # exit()
        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](
        #         outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name

        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged testing stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]
    # return stats, coco_evaluator
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, val_step