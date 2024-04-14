####
#### Basic Usage: python asset_prepare.py --split {Training, Validation} --visit_id_csv ./data_splits/{train_laser_scans.csv, val_laser_scans.csv} 
#### -- Example: python asset_prepare.py --split Training --visit_id_csv ./data_splits/train_laser_scans.csv --dataset_assets mesh annotation

import argparse
import os
import pandas as pd
import copy
from data_downloader.download_utils.download_data import download_assets_for_visit_id, download_assets_for_video_id, \
    default_raw_dataset_assets


DEBUG = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["challenge_dev_set"]
        # choices=["Training", "Validation"], #challenge_dev_set, challenge_test_set
    )

    parser.add_argument(
        "--download_dir",
        default="data",
    )

    # parser.add_argument(
    #     "--visit_id_csv", # two types available {visit_id, video_id, #laser_scans} , {visit_id, #laser_scans}
    # )
    parser.add_argument(
        "--download_only_one_video",
        action="store_true"
    )

    parser.add_argument(
        "--dataset_assets",
        nargs='+',
        choices=default_raw_dataset_assets
    )

    parser.add_argument(
        "--continue_video_id",
        default="0",
    )

    args = parser.parse_args()

    # assert args.visit_id_csv is not None, \
    #     'visit_id_csv must be specified'
    
    continue_video_id = int(args.continue_video_id)

    split = None
    if args.split == "challenge_dev_set":
        split = "Training"
    else:
        split = None
    # split_folder = "train/" if split == "Training" else "val/"
    split_folder = "dev/" if split == "Training" else "test/"

    download_dir = os.path.abspath(args.download_dir)
    assets_folder_path = os.path.join(download_dir, split_folder)
    os.makedirs(assets_folder_path, exist_ok=True)

    video_id_csv = None
    if args.split == "challenge_dev_set" and args.download_only_one_video:
        video_id_csv = "challenge_file_lists/dev_set_only_one_video.csv"
    elif args.split == "challenge_dev_set":
        video_id_csv = "challenge_file_lists/dev_set.csv"

    df = pd.read_csv(video_id_csv)

    if continue_video_id > 0:
        row_index = df[df['video_id'] == continue_video_id].index.tolist()
        if len(row_index) > 1:
            print(f"Error: Multiple video_ids found")
        
        row_index = row_index[0]
        df = df.truncate(before = row_index)

    video_ids_download_set = None
    if "video_id" in df.columns:
        video_ids_download_set = df["video_id"].values.tolist()
        df = df.drop(columns=['video_id'])
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

    df_metadata = pd.read_csv("challenge_file_lists/metadata.csv")

    visit_dataset_assets = []
    video_dataset_assets = []
    for asset in args.dataset_assets:
        if asset == 'laser_scan_5mm' or asset == 'crop_mask':
            visit_dataset_assets.append(asset)
        else:
            video_dataset_assets.append(asset)

    number_of_visits = len(df.index)

    # iterate for each visit_id/video_id
    prev_visit_id = None
    for index, row in df.iterrows():
        print(f"Venue {index} / {number_of_visits}")

        visit_id = int(row['visit_id'])

        current_video_id_list = df_metadata[df_metadata['visit_id'] == visit_id]['video_id'].values.tolist()

        if video_ids_download_set is not None:
            current_video_id_list_ = copy.deepcopy(current_video_id_list)
            for video_id in current_video_id_list_:
                if video_id not in video_ids_download_set:
                    current_video_id_list.remove(video_id)

        print(f"Visit_id: { row['visit_id'] }, Number of video id's: { len(current_video_id_list) } -> video_id_list: {current_video_id_list}")

        current_asset_download_dir = os.path.join(assets_folder_path, str(visit_id))

        if str(visit_id) != prev_visit_id:
            if visit_dataset_assets:
                download_assets_for_visit_id(str(visit_id), current_asset_download_dir, split, visit_dataset_assets, df_metadata)
            prev_visit_id = str(visit_id)
        
        for video_id in current_video_id_list:
            if video_dataset_assets:
                download_assets_for_video_id(str(visit_id), video_id, current_asset_download_dir, split, video_dataset_assets, df_metadata)
            
        print("\n")

    print("Done.")

