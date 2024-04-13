import open3d as o3d
import numpy as np
import copy

def viz_3d(to_plot_list, show_coordinate_system=False):

    if show_coordinate_system:
        coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame()
        
        # to_plot_list_ = copy.deepcopy(to_plot_list)
        # to_plot_list_.append(coordinate_system)
        # o3d.visualization.draw_geometries(to_plot_list_)
        o3d.visualization.draw_geometries([*to_plot_list, coordinate_system])
    else:
        o3d.visualization.draw_geometries(to_plot_list)

def viz_registration_result(source, target, transformation=np.eye(4), show_coordinate_system=False):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    to_plot_list = [source_temp, target_temp]
    viz_3d(to_plot_list, show_coordinate_system)


