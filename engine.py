# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator

from util.misc import NestedTensor

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,criterion_detr: torch.nn.Module,
                    criterion_verbs:torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args):
    max_norm = args.clip_max_norm
    model.train()
    criterion.train()
    criterion_detr.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outs = model(samples)    # pred_logits(100,classes+1), pred_boxes(100,4), aux_outputs(5 groups)
        output_detr = outs['detr']
        outputs = outs['hoi']
        verbs = outs['verbs']
        # cnn verbs loss
        losses_verbs = criterion_verbs(verbs,targets)
        # hoi
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses_hoi = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # detr
        loss_dict_detr = criterion_detr(output_detr, targets)
        weight_dict_detr = criterion_detr.weight_dict
        losses_detr = sum(loss_dict_detr[k] * weight_dict_detr[k] for k in loss_dict_detr.keys() if k in weight_dict_detr)
        # all
        losses = args.loss_hoi_weight*losses_hoi + args.loss_detr_weight*losses_detr + args.loss_verbs_weight*losses_verbs
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {f'{k}_scaled': v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # detr log
        loss_dict_reduced_detr = utils.reduce_dict(loss_dict_detr)
        loss_dict_reduced_unscaled_detr = {f'{k}_unscaled_detr': v
                                      for k, v in loss_dict_reduced_detr.items()}
        loss_dict_reduced_scaled_detr = {f'{k}_scaled_detr': v * weight_dict_detr[k]
                                    for k, v in loss_dict_reduced_detr.items() if k in weight_dict_detr}
        losses_reduced_scaled_detr = sum(loss_dict_reduced_scaled_detr.values())
        loss_value_detr = losses_reduced_scaled_detr.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss_detr=loss_value_detr, **loss_dict_reduced_scaled_detr, **loss_dict_reduced_unscaled_detr)
        metric_logger.update(loss_cnn_verbs = losses_verbs)


        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_w_accum(model: torch.nn.Module, criterion: torch.nn.Module,criterion_detr: torch.nn.Module,
                    criterion_verbs:torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args):
    max_norm = args.clip_max_norm
    model.train()
    criterion.train()
    criterion_detr.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        B = samples.tensors.shape[0]
        b = B // 2
        loss_dicts = list()
        optimizer.zero_grad()
        for i in range(2):
            b_samples = NestedTensor(samples.tensors[i*b:(i+1)*b], samples.mask[i*b:(i+1)*b])
            # b_samples = NestedTensor(samples.tendors[i*b:(i+1)*b], samples.tensors[i*b:(i+1)*b])
            b_targets = []
            for i in range(b*i, b* (i+1)):
                b_targets.append(targets[i])
            # b_targets = [{k: v for k, v in t.items()} for t in targets]
            b_outs = model(b_samples)
            b_output_detr = b_outs['detr']
            b_outputs = b_outs['hoi']
            b_verbs = b_outs['verbs']
            # cnn verbs loss
            b_losses_verbs = criterion_verbs(b_verbs, b_targets)
            # hoi
            b_loss_dict = criterion(b_outputs, b_targets)
            weight_dict = criterion.weight_dict
            b_losses_hoi = sum(b_loss_dict[k] * weight_dict[k] for k in b_loss_dict.keys() if k in weight_dict)
            # detr
            b_loss_dict_detr = criterion_detr(b_output_detr, b_targets)
            weight_dict_detr = criterion_detr.weight_dict
            b_losses_detr = sum(b_loss_dict_detr[k] * weight_dict_detr[k] for k in b_loss_dict_detr.keys() if k in weight_dict_detr)

            # all
            b_losses = (args.loss_hoi_weight * b_losses_hoi + args.loss_detr_weight * b_losses_detr + args.loss_verbs_weight * b_losses_verbs) / 2

            if torch.isnan(b_losses).sum() != 0:
                print('sum_losses',b_losses)
                print('b_losses_hoi',b_losses_hoi)
                print('b_losses_detr',b_losses_detr)
                print('b_losses_verbs',b_losses_verbs)
                sys.exit(1)
            b_losses.backward()
            loss_dicts.append(b_loss_dict)

        loss_dict = {k: (loss_dicts[0][k] + loss_dicts[1][k]) / 2 for k in loss_dicts[0].keys()}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {f'{k}_scaled': v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # detr log
        loss_dict_reduced_detr = utils.reduce_dict(b_loss_dict_detr)
        loss_dict_reduced_unscaled_detr = {f'{k}_unscaled_detr': v
                                      for k, v in loss_dict_reduced_detr.items()}
        loss_dict_reduced_scaled_detr = {f'{k}_scaled_detr': v * weight_dict_detr[k]
                                    for k, v in loss_dict_reduced_detr.items() if k in weight_dict_detr}
        losses_reduced_scaled_detr = sum(loss_dict_reduced_scaled_detr.values())
        loss_value_detr = losses_reduced_scaled_detr.item()


        if not (math.isfinite(loss_value) and math.isfinite(loss_value_detr) and math.isfinite(b_losses_verbs)):
            print("Loss is {}, stopping training".format(loss_value))
            print("DETR Loss is {}, stopping training".format(loss_value_detr))
            print("Verb Loss is {}, stopping training".format(b_losses_verbs))
            print('HOI',loss_dict_reduced)
            print('DETR',loss_dict_reduced_detr)
            sys.exit(1)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss_detr=loss_value_detr, **loss_dict_reduced_scaled_detr, **loss_dict_reduced_unscaled_detr)
        metric_logger.update(loss_cnn_verbs = b_losses_verbs)


        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        output_list = model(samples)
        outputs = output_list['hoi']
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat)

    stats = evaluator.evaluate()

    return stats
