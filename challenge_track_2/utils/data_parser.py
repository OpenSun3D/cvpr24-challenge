# Helpers for parsing the data

import numpy as np
import cv2
import os
import open3d as o3d
import glob
import utils.homogenous as hm
from utils.rigid_interpolation import rigid_interp_split

def decide_pose(pose):
    """
    Args:
        pose: np.array (4, 4)
    Returns:
        index: int (0, 1, 2, 3)
        for upright, left, upside-down and right
    """
    # pose style
    z_vec = pose[2, :3]
    z_orien = np.array(
        [
            [0.0, -1.0, 0.0],  # upright
            [-1.0, 0.0, 0.0],  # left
            [0.0, 1.0, 0.0],  # upside-down
            [1.0, 0.0, 0.0],
        ]  # right
    )
    corr = np.matmul(z_orien, z_vec)
    corr_max = np.argmax(corr)
    return corr_max


def rotate_pose(im, rot_index):
    """
    Args:
        im: (m, n)
    """
    h, w, d = im.shape
    if d == 3:
        if rot_index == 0:
            new_im = im
        elif rot_index == 1:
            new_im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif rot_index == 2:
            new_im = cv2.rotate(im, cv2.ROTATE_180)
        elif rot_index == 3:
            new_im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_im

def st2_camera_intrinsics(filename, format="tuple"):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)

    if format == "tuple":
        return (w, h, fx, fy, hw, hh)
    elif format == "matrix":
        return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
    else:
        raise ValueError(f"Unknown format {format}")

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]

    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))

    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)

    return (ts, Rt)


class DataParser:

    rgb_assets = [
        "wide", "lowres_wide", "vga_wide", "ultrawide"
    ]

    rgb_assets_to_depth_path = {
        "wide": "highres_depth",
        "lowres_wide": "lowres_depth"
    }

    def __init__(self, data_root_path, split = "train"):
        if split not in ["train", "val"]:
            raise ValueError(f"Unknown split {split}")
        
        self.data_root_path = os.path.join(data_root_path, split)
    
    def get_camera_trajectory(self, visit_id, video_id):
        traj_file = os.path.join(self.data_root_path, visit_id, video_id, "lowres_wide.traj")
        with open(traj_file) as f:
            traj = f.readlines()

        # convert traj to json dict
        poses_from_traj = {}
        for line in traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = np.array(TrajStringToMatrix(line)[1].tolist())

        return poses_from_traj

    def get_laser_scan(self, visit_id):
        laser_scan_path = os.path.join(self.data_root_path, visit_id, visit_id + ".ply")

        pcd = o3d.io.read_point_cloud(laser_scan_path)

        return pcd
    
    def get_laser_scan_path(self, visit_id):
        laser_scan_path = os.path.join(self.data_root_path, visit_id, visit_id + ".ply")

        return laser_scan_path

    def get_mesh_reconstruction(self, visit_id, video_id, format="point_cloud"):
        mesh_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_3dod_mesh.ply")
        
        mesh = None 

        if format == "point_cloud":
            mesh = o3d.io.read_point_cloud(mesh_path)
        elif format == "mesh":
            mesh = o3d.io.read_triangle_mesh(mesh_path)
        else: 
            raise ValueError(f"Unknown mesh format {format}")
        
        return mesh
    
    def get_mesh_reconstruction_path(self, visit_id, video_id):
        mesh_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_3dod_mesh.ply")
        
        return mesh_path
    
    def get_highres_reconstruction(self, visit_id, video_id):
        highres_recon_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_highres_recon.ply")
        
        pcd = o3d.io.read_point_cloud(highres_recon_path) 

        return pcd
    
    def get_highres_reconstruction_path(self, visit_id, video_id):
        highres_recon_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_highres_recon.ply")
        
        return highres_recon_path
    
    # def get_3dod_annotations(self, visit_id, video_id):
    #     annotations_3dod_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_3dod_annotation.json")
        
    #     pcd = o3d.io.read_point_cloud(hires_recon_path) 

    #     return pcd
    

    def get_frame_id_and_intrinsic(self, visit_id, video_id, asset_type, format="rgb"):

        if format == "rgb":
            if asset_type not in self.rgb_assets:
                raise ValueError(f"Unknown asset type {asset_type}")
            
            frames_path = os.path.join(self.data_root_path, visit_id, video_id, asset_type)
        elif format == "depth":
            if asset_type not in self.rgb_assets_to_depth_path.keys():
                raise ValueError(f"Unknown asset type {asset_type}")
            
            frames_path = os.path.join(self.data_root_path, visit_id, video_id, self.rgb_assets_to_depth_path[asset_type])
        else:
            raise ValueError(f"Unknown format {format}")

        intrinsics_path = os.path.join(self.data_root_path, visit_id, video_id, asset_type + "_intrinsics")

        frames = sorted(glob.glob(os.path.join(frames_path, "*.png")))
        frame_ids = [os.path.basename(x) for x in frames]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        frame_ids = [x for x in frame_ids]
        frame_ids.sort()

        # get frame paths
        frame_paths = {}
        for frame_id in frame_ids:
            frame_paths[frame_id] = os.path.join(frames_path, f"{video_id}_{frame_id}.png")
        
        # get intrinsics
        intrinsics = {}
        for frame_id in frame_ids:
            intrinsic_fn = os.path.join(intrinsics_path, f"{video_id}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(intrinsics_path,
                                            f"{video_id}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(intrinsics_path,
                                            f"{video_id}_{float(frame_id) + 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                raise ValueError(f"Intrinsics of frame_id {frame_id} do not exist")

            intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)

        return frame_ids, frame_paths, intrinsics
    

    def get_nearest_pose(self, 
                         desired_timestamp,
                         poses_from_traj, 
                         use_interpolation = False, 
                         time_distance_threshold = np.inf,
                         frame_distance_threshold = np.inf):

        max_pose_timestamp = max(float(key) for key in poses_from_traj.keys())
        min_pose_timestamp = min(float(key) for key in poses_from_traj.keys()) 

        if float(desired_timestamp) < min_pose_timestamp or \
            float(desired_timestamp) > max_pose_timestamp:
            return None

        if desired_timestamp in poses_from_traj.keys():
            H = poses_from_traj[desired_timestamp]
        else:
            if use_interpolation:
                greater_closest_timestamp = min(
                    [x for x in poses_from_traj.keys() if float(x) > float(desired_timestamp) ], 
                    key=lambda x: abs(float(x) - float(desired_timestamp))
                )
                smaller_closest_timestamp = min(
                    [x for x in poses_from_traj.keys() if float(x) < float(desired_timestamp) ], 
                    key=lambda x: abs(float(x) - float(desired_timestamp))
                )
                # print(desired_timestamp)
                # print(greater_closest_timestamp)
                # print(smaller_closest_timestamp)

                if abs(float(greater_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold or \
                    abs(float(smaller_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold:
                    # print("Skipping frame.")
                    return None
                
                H0 = poses_from_traj[smaller_closest_timestamp]
                H1 = poses_from_traj[greater_closest_timestamp]
                H0_t = hm.trans(H0)
                H1_t = hm.trans(H1)

                # print(np.linalg.norm(H0_t - H1_t) / (float(greater_closest_timestamp) - float(smaller_closest_timestamp)))
                if np.linalg.norm(H0_t - H1_t) > frame_distance_threshold:
                    # print("Skipping frame.")
                    return None

                H = rigid_interp_split(
                    float(desired_timestamp), 
                    poses_from_traj[smaller_closest_timestamp], 
                    float(smaller_closest_timestamp), 
                    poses_from_traj[greater_closest_timestamp], 
                    float(greater_closest_timestamp)
                )
            else:
                closest_timestamp = min(
                    poses_from_traj.keys(), 
                    key=lambda x: abs(float(x) - float(desired_timestamp))
                )

                if abs(float(closest_timestamp) - float(desired_timestamp)) > time_distance_threshold:
                    # print("Skipping frame.")
                    return None

                H = poses_from_traj[closest_timestamp]

        desired_pose = H

        assert desired_pose.shape == (4, 4)

        return desired_pose

    def get_estimated_transform(self, visit_id, video_id):
        estimated_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_estimated_transform.npy")
        
        estimated_transform = np.load(estimated_transform_path) 

        return estimated_transform
    
    def get_estimated_transform_path(self, visit_id, video_id):
        estimated_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_estimated_transform.npy")
        
        return estimated_transform_path
    
    def get_refined_transform(self, visit_id, video_id):
        refined_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_refined_transform.npy")
        
        refined_transform = np.load(refined_transform_path) 

        return refined_transform
    
    def get_refined_transform_path(self, visit_id, video_id):
        refined_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_refined_transform.npy")
        
        return refined_transform_path
    
    def get_crop_mask(self, visit_id, return_indices=False):
        crop_mask_path = os.path.join(self.data_root_path, visit_id, f"{visit_id}_crop_mask.npy")

        crop_mask = np.load(crop_mask_path)
        
        if return_indices:
            return np.where(crop_mask)[0]
        else:
            return crop_mask
    
    def get_cropped_laser_scan(self, visit_id, laser_scan):
        filtered_idx_list = self.get_crop_mask(visit_id, return_indices=True)

        laser_scan_points = np.array(laser_scan.points)
        laser_scan_colors = np.array(laser_scan.colors)
        laser_scan_points = laser_scan_points[filtered_idx_list]
        laser_scan_colors = laser_scan_colors[filtered_idx_list]

        cropped_laser_scan = o3d.geometry.PointCloud()
        cropped_laser_scan.points = o3d.utility.Vector3dVector(laser_scan_points)
        cropped_laser_scan.colors = o3d.utility.Vector3dVector(laser_scan_colors)
        
        return cropped_laser_scan