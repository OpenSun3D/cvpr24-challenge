# Based on https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts
# Adapted, April 2024

import argparse
import os
import numpy as np
from benchmark_eval.eval_utils.eval_script_inst import evaluate

def main(pred_dir, gt_dir):
    scene_desc_names = sorted(el[:-4] for el in os.listdir(gt_dir) if el.endswith('.txt'))

    preds = {}
    for scene_desc_name in scene_desc_names[:]:

        # Load predictions based on the requested submission format
        file_path = os.path.join(pred_dir, scene_desc_name+'.txt')  # {visit_id}_{desc_id}.txt file
        scene_pred_mask_list = np.loadtxt(file_path, dtype=str)  # (num_masks, 2)

        if scene_pred_mask_list.shape == (2,):
            scene_pred_mask_list = scene_pred_mask_list[np.newaxis, ...]

        assert scene_pred_mask_list.shape[1] == 2, 'Each line should have 2 values: instance mask path and confidence score.'
        pred_masks = []
        pred_scores = []
        for mask_path, conf_score in scene_pred_mask_list: 
            # Read mask and confidence score for each instance mask
            pred_mask = np.loadtxt(os.path.join(pred_dir, mask_path), dtype=int) # Values: 0 for the background, 1 for the instance
            pred_masks.append(pred_mask)
            pred_scores.append(float(conf_score))

        assert len(pred_masks) == len(pred_scores), 'Number of masks and confidence scores should be the same.'
        
        # Aggregate masks and scores for each scene - pred_class is always 1 (we only have one semantic class, 'functionality', referring to the query functionality)
        preds[scene_desc_name] = {
            'pred_masks': np.vstack(pred_masks).T if len(pred_masks) > 0 else np.zeros((1, 1)),
            'pred_scores': np.vstack(pred_scores) if len(pred_masks) > 0 else np.zeros(1),
            'pred_classes': np.ones(len(pred_masks), dtype=np.int64) if len(pred_masks) > 0 else np.ones(1, dtype=np.int64)
        }

    # Run evaluation script
    ap_dict = evaluate(preds, gt_dir)
    del ap_dict['classes']
    print(ap_dict)
 
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_dir",
        help="Specify the predictions directory. Predictions must be in the submission format, containing '<visit_id>_<desc_id>.txt' files and 'predicted_masks/' folder including all masks."
    )

    parser.add_argument(
        "--gt_dir",
        help="Specify the GT annotations directory. It must contain <visit_id>_<desc_id>.txt files for gt annotations, see https://github.com/OpenSun3D/cvpr24-challenge/blob/main/challenge_track_2/benchmark_data/gt_development_scenes"
    )

    args = parser.parse_args()

    main(args.pred_dir, args.gt_dir)
