# Helpers for parsing the data

import numpy as np
import cv2
import os
import open3d as o3d
import glob
import imageio
import utils.homogenous as hm
from utils.rigid_interpolation import rigid_interp_split, rigid_interp_geodesic

def decide_pose(pose):
    """
    Determines the orientation of a 3D pose based on the alignment of its z-vector with predefined orientations.

    Args:
        pose (np.ndarray): A 4x4 NumPy array representing a 3D pose transformation matrix.

    Returns:
        (int): Index representing the closest predefined orientation:
             0 for upright, 1 for left, 2 for upside-down, and 3 for right.
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
    Rotates an image by a specified angle based on the rotation index.

    Args:
        im (numpy.ndarray): The input image to be rotated. It should have shape (height, width, channels).
        rot_index (int): Index representing the rotation angle:
                         0 for no rotation, 1 for 90 degrees clockwise rotation,
                         2 for 180 degrees rotation, and 3 for 90 degrees counterclockwise rotation.

    Returns:
        (numpy.ndarray): The rotated image.
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
    """
    Parses a file containing camera intrinsic parameters and returns them in the specified format.

    Args:
        filename (str): The path to the file containing camera intrinsic parameters.
        format (str, optional): The format in which to return the camera intrinsic parameters.
                                Supported formats are "tuple" and "matrix". Defaults to "tuple".

    Returns:
        (Union[tuple, numpy.ndarray]): Camera intrinsic parameters in the specified format.

            - If format is "tuple", returns a tuple \\(w, h, fx, fy, hw, hh\\).
            - If format is "matrix", returns a 3x3 numpy array representing the camera matrix.
    
    Raises:
        ValueError: If an unsupported format is specified.
    """
    w, h, fx, fy, hw, hh = np.loadtxt(filename)

    if format == "tuple":
        return (w, h, fx, fy, hw, hh)
    elif format == "matrix":
        return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
    else:
        raise ValueError(f"Unknown format {format}")

def convert_angle_axis_to_matrix3(angle_axis):
    """
    Converts a rotation from angle-axis representation to a 3x3 rotation matrix.

    Args:
        angle_axis (numpy.ndarray): A 3-element array representing the rotation in angle-axis form [angle, axis_x, axis_y, axis_z].

    Returns:
        (numpy.ndarray): A 3x3 rotation matrix representing the same rotation as the input angle-axis.

    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def TrajStringToMatrix(traj_str):
    """ 
    Converts a line from the camera trajectory file into translation and rotation matrices

    Args:
        traj_str (str): A space-delimited file where each line represents a camera pose at a particular timestamp. The file has seven columns:

            - Column 1: timestamp
            - Columns 2-4: rotation (axis-angle representation in radians)
            - Columns 5-7: translation (usually in meters)

    Returns:
        (tuple): Tuple containing:

               - ts (str): Timestamp.
               - Rt (numpy.ndarray): Transformation matrix representing rotation and translation.

    Raises:
        AssertionError: If the input string does not have exactly seven columns.
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
    """
    A class for parsing data files in the SceneFun3D dataset
    """

    rgb_assets = [
        "wide", "lowres_wide", "vga_wide", "ultrawide"
    ]

    rgb_assets_to_depth_path = {
        "wide": "highres_depth",
        "lowres_wide": "lowres_depth"
    }

    def __init__(self, data_root_path, split = "train"):
        """
        Initialize the DataParser instance with the root path and split.

        Args:
            data_root_path (str): The root path where data is located.
            split (str, optional): The split of the data (e.g., "train", "val"). Defaults to "train".

        Raises:
            ValueError: If an unknown split is specified.
        """
        if split not in ["train", "val", "test", "dev"]:
            raise ValueError(f"Unknown split {split}")
        
        self.data_root_path = os.path.join(data_root_path, split)
    
    def get_camera_trajectory(self, visit_id, video_id):
        """
        Retrieve the camera trajectory from a file and convert it into a dictionary whose keys are the timestamps and values are the corresponding camera poses.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (dict): A dictionary where keys are timestamps (rounded to 3 decimal places) and values are 4x4 transformation matrices representing camera poses.
        """
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
        """
        Load a point cloud from a .ply file containing laser scan data.

        Args:
            visit_id (str): The identifier of the scene.

        Returns:
            (open3d.geometry.PointCloud): A point cloud object containing the laser scan data (i.e., XYZRGB point cloud).
        """
        laser_scan_path = os.path.join(self.data_root_path, visit_id, visit_id + "_laser_scan.ply")

        pcd = o3d.io.read_point_cloud(laser_scan_path)

        return pcd
    
    def get_laser_scan_path(self, visit_id):
        """
        Get the file path of the laser scan.

        Args:
            visit_id (str): The identifier of the scene.

        Returns:
            (str): The file path of the .ply file containing the laser scan.
        """
        laser_scan_path = os.path.join(self.data_root_path, visit_id, visit_id + "_laser_scan.ply")

        return laser_scan_path


    def get_mesh_reconstruction(self, visit_id, video_id, format="point_cloud"):
        """
        Load mesh reconstruction data based on the iPad video sequence from a .ply file.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            format (str, optional): The format of the mesh reconstruction data to load. 
                                    Supported formats are "point_cloud" and "mesh". 
                                    Defaults to "point_cloud".

        Returns:
            (Union[open3d.geometry.PointCloud, open3d.geometry.TriangleMesh]): 
                The loaded mesh reconstruction data in the specified format.

        Raises:
            ValueError: If an unsupported 3D data format is specified.
        """
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
        """
        Get the file path of the mesh reconstruction data based on the iPad video sequence.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (str): The file path of the .ply file containing the mesh reconstruction data.
        """
        mesh_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_3dod_mesh.ply")
        
        return mesh_path


    def get_highres_reconstruction(self, visit_id, video_id):
        """
        Load high-resolution 3D reconstruction data based on the iPad hires frames from a .ply file.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (open3d.geometry.PointCloud): A point cloud object containing the high-resolution 3D reconstruction data.
        """
        highres_recon_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_highres_recon.ply")
        
        pcd = o3d.io.read_point_cloud(highres_recon_path) 

        return pcd


    def get_highres_reconstruction_path(self, visit_id, video_id):
        """
        Get the file path of the high-resolution reconstruction data based on the iPad hires frames.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (str): The file path of the .ply file containing the high-resolution 3D reconstruction data.
        """
        highres_recon_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_highres_recon.ply")
        
        return highres_recon_path
    

    def get_frame_id_and_intrinsic(self, visit_id, video_id, asset_type, format="rgb"):
        """
        Retrieve frame IDs, frame paths, and camera intrinsics for a given visit, video, and asset type.

        Args:
            visit_id (str): The identifier of the visit.
            video_id (str): The identifier of the video within the visit.
            asset_type (str): The type of asset, such as "rgb" or "depth". 
                                Supported asset types are ["wide", "lowres_wide", "vga_wide", "ultrawide"] if format="rgb" and ["wide", "lowres_wide"] if format="depth"
            format (str, optional): The format of the asset data to retrieve. 
                                    Supported formats are "rgb" and "depth". 
                                    Defaults to "rgb".

        Returns:
            (tuple): A tuple containing:

                - frame_ids (list): A list of frame IDs.
                - frame_paths (dict): A dictionary mapping frame IDs to their corresponding file paths.
                - intrinsics (dict): A dictionary mapping frame IDs to their camera intrinsics.

        Raises:
            ValueError: If an unknown asset type or format is specified, or if the intrinsics file of a frame does not exist.
        """

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
                         time_distance_threshold = np.inf,
                         use_interpolation = False,
                         interpolation_method = 'split',
                         frame_distance_threshold = np.inf):
        """
        Get the nearest pose to a desired timestamp from a dictionary of poses.

        Args:
            desired_timestamp (float): The timestamp of the desired pose.
            poses_from_traj (dict): A dictionary where keys are timestamps and values are 4x4 transformation matrices representing poses.
            time_distance_threshold (float, optional): The maximum allowable time difference between the desired timestamp and the nearest pose timestamp. Defaults to np.inf.
            use_interpolation (bool, optional): Whether to use interpolation to find the nearest pose. Defaults to False.
            interpolation_method (str, optional): Supports two options, "split" or "geodesic_path". Defaults to "split".

                - "split": performs rigid body motion interpolation in SO(3) x R^3
                - "geodesic_path": performs rigid body motion interpolation in SE(3)
            frame_distance_threshold (float, optional): The maximum allowable distance in terms of frame difference between the desired timestamp and the nearest pose timestamp. Defaults to np.inf.

        Returns:
            (Union[numpy.ndarray, None]): The nearest pose as a 4x4 transformation matrix if found within the specified thresholds, else None.

        Raises:
            ValueError: If an unsupported interpolation method is specified.

        Note:
            If `use_interpolation` is True, the function will perform rigid body motion interpolation between two nearest poses to estimate the desired pose. 
            The thresholds `time_distance_threshold` and `frame_distance_threshold` are used to control how tolerant the function is towards deviations in time and frame distance.
        """

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

                if abs(float(greater_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold or \
                    abs(float(smaller_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold:
                    # print("Skipping frame.")
                    return None
                
                H0 = poses_from_traj[smaller_closest_timestamp]
                H1 = poses_from_traj[greater_closest_timestamp]
                H0_t = hm.trans(H0)
                H1_t = hm.trans(H1)

                if np.linalg.norm(H0_t - H1_t) > frame_distance_threshold:
                    # print("Skipping frame.")
                    return None

                if interpolation_method == "split":
                    H = rigid_interp_split(
                        float(desired_timestamp), 
                        poses_from_traj[smaller_closest_timestamp], 
                        float(smaller_closest_timestamp), 
                        poses_from_traj[greater_closest_timestamp], 
                        float(greater_closest_timestamp)
                    )
                elif interpolation_method == "geodesic_path":
                    H = rigid_interp_geodesic(
                        float(desired_timestamp), 
                        poses_from_traj[smaller_closest_timestamp], 
                        float(smaller_closest_timestamp), 
                        poses_from_traj[greater_closest_timestamp], 
                        float(greater_closest_timestamp)
                    )
                else:
                    raise ValueError(f"Unknown interpolation method {interpolation_method}")

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
        """
        Load the estimated transformation matrix from a .npy file.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (numpy.ndarray): The estimated transformation matrix loaded from the file.
        """
        estimated_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_estimated_transform.npy")
        estimated_transform = np.load(estimated_transform_path) 
        return estimated_transform

    def get_estimated_transform_path(self, visit_id, video_id):
        """
        Get the file path of the estimated transformation matrix.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (str): The file path of the .npy file containing the estimated transformation matrix.
        """
        estimated_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_estimated_transform.npy")
        return estimated_transform_path

    def get_refined_transform(self, visit_id, video_id):
        """
        Load the refined transformation matrix from a .npy file.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (numpy.ndarray): The refined transformation matrix loaded from the file.
        """
        refined_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_refined_transform.npy")
        refined_transform = np.load(refined_transform_path) 
        return refined_transform

    def get_refined_transform_path(self, visit_id, video_id):
        """
        Get the file path of the refined transformation matrix.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (str): The file path of the .npy file containing the refined transformation matrix.
        """
        refined_transform_path = os.path.join(self.data_root_path, visit_id, video_id, f"{video_id}_refined_transform.npy")
        return refined_transform_path
    
    def read_rgb_frame(self, full_frame_path, normalize=False):
        """
        Read an RGB frame from the specified path.

        Args:
            full_frame_path (str): The full path to the RGB frame file.
            normalize (bool, optional): Whether to normalize the pixel values to the range [0, 1]. Defaults to False.

        Returns:
            (numpy.ndarray): The RGB frame as a NumPy array with the RGB color values.

        """
        color = imageio.v2.imread(full_frame_path)

        if normalize:
            color = color / 255.

        return color
    
    def read_depth_frame(self, full_frame_path, conversion_factor=1000):
        """
        Read a depth frame from the specified path and convert it to depth values.

        Args:
            full_frame_path (str): The full path to the depth frame file.
            conversion_factor (float, optional): The conversion factor to convert pixel values to depth values. Defaults to 1000 to convert millimeters to meters.

        Returns:
            (numpy.ndarray): The depth frame as a NumPy array with the depth values.
        """
        
        depth = imageio.v2.imread(full_frame_path) / conversion_factor

        return depth

    def get_crop_mask(self, visit_id, return_indices=False):
        """
        Load the crop mask from a .npy file.

        Args:
            visit_id (str): The identifier of the scene.
            return_indices (bool, optional): Whether to return the indices of the cropped points. Defaults to False.

        Returns:
            numpy.ndarray or tuple: The crop mask loaded from the file. If `return_indices` is True, returns a tuple containing the indices of the cropped points.
        """
        crop_mask_path = os.path.join(self.data_root_path, visit_id, f"{visit_id}_crop_mask.npy")
        crop_mask = np.load(crop_mask_path)
        
        if return_indices:
            return np.where(crop_mask)[0]
        else:
            return crop_mask

    def get_cropped_laser_scan(self, visit_id, laser_scan):
        """
        Crop a laser scan using a crop mask.

        Args:
            visit_id (str): The identifier of the scene.
            laser_scan (open3d.geometry.PointCloud): The laser scan point cloud to be cropped.

        Returns:
            (open3d.geometry.PointCloud): The cropped laser scan point cloud.
        """
        filtered_idx_list = self.get_crop_mask(visit_id, return_indices=True)

        laser_scan_points = np.array(laser_scan.points)
        laser_scan_colors = np.array(laser_scan.colors)
        laser_scan_points = laser_scan_points[filtered_idx_list]
        laser_scan_colors = laser_scan_colors[filtered_idx_list]

        cropped_laser_scan = o3d.geometry.PointCloud()
        cropped_laser_scan.points = o3d.utility.Vector3dVector(laser_scan_points)
        cropped_laser_scan.colors = o3d.utility.Vector3dVector(laser_scan_colors)
        
        return cropped_laser_scan