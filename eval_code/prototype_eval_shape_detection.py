#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import logging
import json
import sys

import numpy as np
from sklearn.metrics import auc
import skimage.io as skio
import pandas

import iou


IOU_THRESHOLD_DEFAULT = 0.5
logging.basicConfig(level=logging.INFO)

BG_LABEL = 0







def coco_panoptic_metrics(df: pandas.DataFrame, iou_threshold: float = 0.5):
    idx = df.index.searchsorted(iou_threshold)
    # if idx > len(df) then df.iloc[idx] will fail
    if idx >= len(df):
        return 0., 0., 0.

    iou_sum = np.sum(df.index[idx:])
    tp = df["True Positives"].iloc[idx]
    fp = df["False Positives"].iloc[idx]
    fn = df["False Negatives"].iloc[idx]
    pq = iou_sum / (tp + 0.5 * fp + 0.5 * fn)
    sq = (iou_sum / tp) if tp != 0 else 0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return pq, sq, rq



def print_scores_summary(iou_ref: np.array, iou_contender: np.array, df: pandas.DataFrame, iou_threshold = 0.5, file = None):
    nlabel_gt = iou_ref.size
    nlabel_contender = iou_contender.size
    if file == None:
        file = sys.stdout

    idx = df.index.searchsorted(iou_threshold)
    f1_auc = 0.
    # if idx > len(df) then df.iloc[idx] will fail
    if idx < len(df):
        print(f"F1@IoU>={iou_threshold:0.3}: {df['F-score'].iloc[idx]:.3f}", file=file)
        # Is the first IoU exactly equal to our threshold?
        # (this means our first element is exactly on our left edge)
        first_value_on_edge = df.index[idx] == iou_threshold
        num_vals = len(df) - idx + (1 if not first_value_on_edge else 0)
        if num_vals > 1:
        # We need at least 2 points to compute the AUC
            ious = np.zeros(num_vals, dtype=np.float32)
            fscores = np.zeros(num_vals, dtype=np.float32)
            # Add an extra point on the left by taking the best value for IoU > Thresh
            # (unless our first point is on the edge -- this is useless then)
            offset = 0
            if not first_value_on_edge:
                ious[0] = iou_threshold
                fscores[0] = df["F-score"].iloc[idx]
                offset = 1
        ious[offset:] = df.index[idx:]
        fscores[offset:] = df["F-score"].iloc[idx:]
        f1_auc = auc(ious, fscores)
    print(f"Number of labels in GT: {nlabel_gt}", file = file)
    print(f"Number of labels in contender: {nlabel_contender}", file = file)
    print("AUC F1@IoU>{}: {:.3f}".format(iou_threshold, f1_auc), file=file)

    pq, sq, rq = coco_panoptic_metrics(df, 0.5)
    print(f"COCO Panoptic eval @ IoU>0.5 (default): PQ: {pq:.3f} = SQ {sq:.3f} + RQ {rq:.3f}", file=file)
    if iou_threshold != 0.5:
        pq, sq, rq = coco_panoptic_metrics(df, iou_threshold)
        print(f"COCO Panoptic eval @ IoU>{iou_threshold:0.3}: PQ: {pq:.3f} = SQ {sq:.3f} + RQ {rq:.3f}", file=file)

    THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
    idx = df.index.searchsorted(THRESHOLDS)
    # num_thr = len(THRESHOLDS)
    msk = idx < len(df)
    idx = idx[msk]
    num_valid_idx = np.count_nonzero(msk)

    # Create a larger dataframe to store the results
    subset = pandas.DataFrame(columns=df.columns, index=np.array(THRESHOLDS))
    subset.index.name = "IoU"

    # Copy valid elements if there are some
    if num_valid_idx > 0:
        subset.iloc[:num_valid_idx] = df.iloc[idx]

    # Assign default value to remaining elements
    for extra_idx in range(num_valid_idx, len(THRESHOLDS)):
        subset.iloc[extra_idx] = {
            "Precision" : 0.,
            "Recall" : 0.,
            "F-score" : 0.,
            "True Positives": 0,
            "False Positives": nlabel_contender,
            "False Negatives": nlabel_gt, 
        }

    print(subset.round(2), file = file)
    return subset


def shape_detection(input_gt_path, input_contenders_path, input_mask, output_dir,
            iou_threshold,
            ignore_label_0_gt=False, ignore_label_0_pred=False):
    if not (0.5 <= iou_threshold < 1.0):
        raise ValueError(f"iou_threshold parameter must be >= 0.5 and < 1.")

    # Load input images
    # ref = cv2.imread(input_gt_path, cv2.IMREAD_UNCHANGED)
    ref = skio.imread(input_gt_path)
    if ref is None:
        raise ValueError(f"input file {input_gt_path} cannot be read.")

    # Load mask image
    msk_bg = None
    if input_mask:
        # msk_bg = cv2.imread(input_mask, cv2.IMREAD_UNCHANGED)
        msk_bg = skio.imread(input_mask)
        if msk_bg is None:
            raise ValueError(f"mask file {input_mask} cannot be read.")
        if msk_bg.shape != ref.shape:
            raise ValueError("GT and MASK image do not have the same shapes: {} vs {}", ref.shape, msk_bg.shape)
        # Create boolean mask
        msk_bg = msk_bg==0

    # Mask input image if needed
    if msk_bg is not None:
        ref = iou.mask_label_image(ref, msk_bg, bg_label=BG_LABEL)

    contenders = []
    for p in input_contenders_path:
        p = Path(p)
        # contender = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        contender = skio.imread(str(p))

        if contender is None:
            raise ValueError(f"input file {p} cannot be read.")

        if contender.shape != ref.shape:
            raise ValueError("GT and PRED label maps do not have the same shapes: {} vs {}", ref.shape, contender.shape)

        # Mask predicted image if needed
        if msk_bg is not None:
            contender = iou.mask_label_image(contender, msk_bg, bg_label=BG_LABEL)

        contenders.append((str(p.stem), contender))

    # Create output dir early
    os.makedirs(output_dir, exist_ok=True)

    odir = Path(output_dir)
    recalls = []
    precisions = []
    # coco_metrics - structure: dict[str, dict(str, float)]
    # contender_name -> {"pq": pq_float_value, "sq": sq_float_value, "rq": rq_float_value}
    coco_metrics = {}
    for name, contender_img in contenders:
        logging.info("Processing: %s", name)
        intersections = iou.intersections(ref, contender_img,
            ignore_label_0_a=ignore_label_0_gt,
            ignore_label_0_b=ignore_label_0_pred)

        # Divided by marginal area only, not union
        # Usefull to say this is a bad metric
        # recall, precision = iou.compute_IoUs(intersections, mode="marginal")
        # iou.viz_iou(ref, recall, Path(odir, "viz_marginal_recall_{}.jpg".format(name)))
        # iou.viz_iou(contender_img, precision, Path(odir, "viz_marginal_precision_{}.jpg".format(name)))

        # # Scaled by area/areamax
        # # wtf?
        # # recall, precision = iou.compute_IoUs(intersections, scaling="marginal")
        # # iou.viz_iou(ref, recall, Path(odir, "viz_scaled_recall_{}.jpg".format(name)))
        # # iou.viz_iou(contender_img, precision, Path(odir, "viz_scaled_precision_{}.jpg".format(name)))


        # intersection over union
        # the good stuff
        recall, precision = iou.compute_IoUs(intersections)
        iou.viz_iou(ref, recall, Path(odir, "viz_iou=0.5_recall_{}.jpg".format(name)))
        iou.viz_iou(contender_img, precision, Path(odir, "viz_iou=0.5_precision_{}.jpg".format(name)))
        # custom lower bound for visualization
        if iou_threshold != 0.5:
            iou.viz_iou(ref, recall,
                Path(odir, f"viz_iou={iou_threshold:0.3f}_recall_{name}.jpg"),
                lower_bound=iou_threshold)
            iou.viz_iou(contender_img, precision,
                Path(odir, f"viz_iou={iou_threshold:0.3f}_precision_{name}.jpg"),
                lower_bound=iou_threshold)

        # # Number of nodes at a distance <= 1
        # # graph point of view, very strict
        # num_matches_gt, num_matches_pred = iou.compute_num_matches(intersections, max_distance=1)
        # iou.viz_matches(ref, num_matches_gt, Path(odir, f"viz_links1_marginal_overseg_{name}.jpg"))
        # iou.viz_matches(contender_img, num_matches_pred, Path(odir, f"viz_links1_marginal_underseg_{name}.jpg"))

        # # Number of nodes at a distance <= 2
        # # FIXME buggy
        # # num_matches_gt, num_matches_pred = iou.compute_num_matches(intersections, max_distance=2)
        # # iou.viz_matches(ref, num_matches_gt, Path(odir, f"viz_links2_marginal_overseg_{name}.jpg"))
        # # iou.viz_matches(contender_img, num_matches_pred, Path(odir, f"viz_links2_marginal_underseg_{name}.jpg"))

        # # IoU with the 2nd best match
        # # useless
        # # rec2, prec2 = iou.compute_2ndbest_iou(intersections)
        # # iou.viz_2ndbest(ref, rec2, Path(odir, "viz_2ndbest_recall_{}.jpg".format(name)))
        # # iou.viz_2ndbest(contender_img, prec2, Path(odir, "viz_2ndbest_precision_{}.jpg".format(name)))

        # # Difference of best and 2nd best IoU
        # # stricter that prec/recall
        # rec2, prec2 = iou.compute_fx_1st_to_2nd_best(intersections, fx="difference")
        # iou.viz_iou(ref, rec2, Path(odir, "viz_diff12_recall_{}.jpg".format(name)))
        # iou.viz_iou(contender_img, prec2, Path(odir, "viz_diff12_precision_{}.jpg".format(name)))

        # # 1 - ratio between 2nd best and 1st best IoU
        # # useless
        # # rec2, prec2 = iou.compute_fx_1st_to_2nd_best(intersections, fx="ratio")
        # # iou.viz_iou(ref, rec2, Path(odir, "viz_ratio12_recall_{}.jpg".format(name)))
        # # iou.viz_iou(contender_img, prec2, Path(odir, "viz_ratio12_precision_{}.jpg".format(name)))

        # # Show exact 1-n or m-1 matches
        # # goal: look for recoverable cases with minimal manual annotation
        # # FIXME this is buggy
        # rec2, prec2 = iou.identify_acceptable_overunderseg(intersections, mode="many_to_one")
        # iou.viz_iou(ref, rec2, Path(odir, "viz_manytoone_gt_{}.jpg".format(name)))
        # iou.viz_iou(contender_img, prec2, Path(odir, "viz_manytoone_pred_{}.jpg".format(name)))
        # rec2, prec2 = iou.identify_acceptable_overunderseg(intersections, mode="one_to_many")
        # iou.viz_iou(ref, rec2, Path(odir, "viz_onetomany_gt_{}.jpg".format(name)))
        # iou.viz_iou(contender_img, prec2, Path(odir, "viz_onetomany_pred_{}.jpg".format(name)))

        df = iou.compute_matching_scores(recall, precision)
        df.to_csv(Path(odir, f"{name}_figure.csv"))

        pq, sq, rq = coco_panoptic_metrics(df, iou_threshold)
        coco_metrics_current = {"iou_threshold": iou_threshold, "PQ": pq, "SQ": sq, "RQ": rq}
        with open(Path(odir, f"{name}_coco_panoptic_metrics.json"), "w") as f:
            json.dump(coco_metrics_current, f)
        coco_metrics[name] = coco_metrics_current

        with open(Path(odir, f"{name}_summary.txt"), "w") as f:
            print_scores_summary(recall, precision, df, iou_threshold=iou_threshold)
            _subset = print_scores_summary(recall, precision, df, iou_threshold=iou_threshold, file=f)

        iou.plot_scores(df, out = Path(odir, f"{name}_figure.pdf"))
        recalls.append(recall)
        precisions.append(precision)


    if len(contenders) == 2:
        A_recall, B_recall = recalls
        A_precision, B_precision = precisions
        (A_name, A), (B_name, B) = contenders
        A_recall_map = A_recall[ref]
        A_precision_map = A_precision[A]
        B_recall_map = B_recall[ref]
        B_precision_map = B_precision[B]
        iou.diff(A_recall_map, B_recall_map, out_path=Path(odir, "compare_recall.png"))
        iou.diff(A_precision_map, B_precision_map, out_path=Path(odir, "compare_precision.png"))

    print("All done.")



def main():
    parser = argparse.ArgumentParser(description='Evaluate the detection of shapes.')
    parser.add_argument('input_gt_path', help='Path to the input label map (TIFF 16 bits) for ground truth.')
    parser.add_argument('input_contenders_path', help='Path to the contenders label map (TIFF 16 bits) for predictions.', nargs='+')
    parser.add_argument('-m', '--input-mask', help='Path to an mask image (pixel with value 0 will be discarded in the evaluation).')
    parser.add_argument('-o', '--output-dir', help='Path to the output directory where results will be stored.')
    parser.add_argument('--iou-threshold', type=float, help='Threshold value (float) for IoU: 0.5 <= t < 1.'
                        f' Default={IOU_THRESHOLD_DEFAULT}', default=IOU_THRESHOLD_DEFAULT)
    parser.add_argument('--ignore_label_0_gt',
        help='Activate to set all intersection values to 0 when comparing against GT label 0.',
        action="store_true")
    parser.add_argument('--ignore_label_0_pred',
        help='Activate to set all intersection values to 0 when comparing against PRED label 0.',
        action="store_true")
    args = parser.parse_args()
    shape_detection(
        args.input_gt_path,
        args.input_contenders_path,
        args.input_mask,
        args.output_dir,
        args.iou_threshold,
        ignore_label_0_gt=args.ignore_label_0_gt,
        ignore_label_0_pred=args.ignore_label_0_pred,
    )

if __name__ == "__main__":
    main()
