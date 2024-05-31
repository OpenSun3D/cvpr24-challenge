# Helpers for visualization
#
# SceneFun3D Toolkit
#

import open3d as o3d
import numpy as np
import copy
from utils.pc_process import pc_estimate_normals
import pyviz3d.visualizer as viz
from utils.viz_constants import SCANNET_COLOR_MAP_200, VIZ_TOOL_OPTIONS

def viz_3d(to_plot_list, show_coordinate_system=False):
    """
    Visualize 3D geometries using Open3D.

    This function takes a list of 3D geometries (e.g., point clouds) and visualizes them using Open3D's visualization tools.
    An optional coordinate system can be displayed for reference.

    Args:
        to_plot_list (list): A list of Open3D geometry objects to be visualized. These can include point clouds (o3d.geometry.PointCloud), meshes (o3d.geometry.TriangleMesh), etc.
        show_coordinate_system (bool, optional): A flag to indicate whether to show the coordinate system in the visualization. Default is False.

    Returns:
        None
            This function does not return any value. It opens a visualizer window displaying the geometries.
    
    Example:
    --------
    >>> import open3d as o3d
    >>> pcd = o3d.geometry.PointCloud()
    >>> pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    >>> viz_3d([pcd], show_coordinate_system=True)
    """

    if show_coordinate_system:
        coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame()

        o3d.visualization.draw_geometries([*to_plot_list, coordinate_system])
    else:
        o3d.visualization.draw_geometries(to_plot_list)

def viz_masks(laser_scan_pcd, mask_indices, mask_labels = None, use_normals=True, viz_tool="pyviz3d"):
    """
    Visualize point cloud masks.

    Args:
        laser_scan_pcd (open3d.geometry.PointCloud): The original point cloud to be visualized.
        mask_indices (list of lists of int or list of np.array of int): A list where each element is a list (or a np.array) of indices representing a mask in the point cloud.
        mask_labels (list of str, optional): Labels for each mask to be used with Pyviz3D. Must be the same length as mask_indices. Default is None.
        use_normals (bool, optional): Whether to use normals in the visualization. If True and the point cloud does not have normals, they will be estimated. Default is True.
        viz_tool (str, optional): The visualization tool to use, either 'pyviz3d' or 'open3d'. Default is 'pyviz3d'.

    Returns:
        None
            The function visualizes the point cloud masks and does not return any value.
    """
    if viz_tool not in VIZ_TOOL_OPTIONS:
        assert False, f"Unknown viz tool option {viz_tool}. Visualization tool option must only be 'open3d' or 'pyviz3d'."

    if mask_labels is not None and viz_tool == "open3d":
        assert False, f"The mask_labels input is only supported for the visualization option 'pyviz3d'"

    if mask_labels is not None:
        assert len(mask_indices) == len(mask_labels), f"The length of mask_labels must be equal to the length of mask_indices. Each label must correspond to a single mask."

    pcd = copy.deepcopy(laser_scan_pcd)
    if not pcd.has_normals() and use_normals:
        pcd = pc_estimate_normals(pcd, radius = 0.1, max_nn = 50)

    # center the point cloud
    pcd_points = np.array(pcd.points) 
    pcd_points -= np.mean(pcd_points, axis=0)
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

    if viz_tool == "pyviz3d":
        assert use_normals, f"use_normals must be set to True if the visualization tool option is 'pyviz3d'."
        v = viz.Visualizer()
        v.add_points('RGB Color', np.array(pcd.points), np.array(pcd.colors)*255., np.array(pcd.normals), point_size=8, visible=False)

    pcd_colors = np.array(pcd.colors) 
    new_pcd_colors = np.ones((pcd_colors.shape))*0+ (0.77, 0.77, 0.77)

    if viz_tool == "pyviz3d":
        v.add_points('Background Color', np.array(pcd.points), new_pcd_colors*255., np.array(pcd.normals), point_size=8, visible=True)

    COLOR_MAP = list(SCANNET_COLOR_MAP_200.values())

    for annot_i, mask_idx in enumerate(mask_indices):
        cur_color = COLOR_MAP[annot_i % len(COLOR_MAP)]

        new_pcd_colors[mask_idx] = cur_color 
        new_pcd_colors[mask_idx] /= 255.

        if viz_tool == "pyviz3d":
            mask_points = np.array(pcd.points)[mask_idx]
            mask_colors = new_pcd_colors[mask_idx]
            mask_normals = np.array(pcd.normals)[mask_idx]

            if mask_labels is not None:
                cur_label = mask_labels[annot_i]
            else:
                cur_label = f"label_{annot_i}"

            v.add_points(cur_label, mask_points, mask_colors*255., mask_normals, point_size=8, visible=True)
        
    if viz_tool == "pyviz3d":
        v.save('pyviz3d_output')
    else:
        pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
        viz_3d([pcd])



def viz_registration_result(source, target, transformation=np.eye(4), show_coordinate_system=False):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    to_plot_list = [source_temp, target_temp]
    viz_3d(to_plot_list, show_coordinate_system)

