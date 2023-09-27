import numpy as np
import os
import json
import numpy as np
import open3d as o3d
#from open3d.geometry import PointCloud

frame_count = 1

#读取数据
def read_data():
    path = r'D:/VR_24/Segmentation/4cams-data-ply-0907/'
    files = os.listdir(path)

    point_files = [f for f in files if 'point_cloud' in f and 'points' in f]

    pcd0, pcd1, skeletons = [], [], []
    for i in range(frame_count):

        with open(os.path.join(path, f'frame-{i}-skeleton_0.json'), 'r') as fr:
            skeleton = json.load(fr)
        # with open(os.path.join(path, f'frame-{i}-skeleton_1.json'), 'r') as fr:
        #     skeleton1 = json.load(fr)

        # pcd0.append(point_cloud0)
        # pcd1.append(point_cloud1)
        skeletons.append(skeleton)
    #
    # return pcd0, pcd1, skeletons
    return skeletons
#filter的代码，逻辑上应该没问题，输入是点云数据，已经两个相邻的skeleton joint的坐标以及半径
#判断每个点到圆柱体axis的距离，保留距离小于radius的points
def filter_points_by_cylinder(ply_file, point1, point2, radius):
    # Load the PLY file as a point cloud
    point_cloud = o3d.io.read_point_cloud(ply_file)

    # Convert the point cloud to a NumPy array
    point_cloud_data = np.asarray(point_cloud.points)

    # Calculate the direction vector of the cylinder axis
    axis_vector = point2 - point1
    axis_length = np.linalg.norm(axis_vector)
    axis_direction = axis_vector / axis_length

    # Calculate the distance from the cylinder axis for each point
    vector_to_points = point_cloud_data - point1
    perpendicular_distances = np.linalg.norm(np.cross(vector_to_points, axis_direction), axis=1)

    # Create a mask to filter out points that are outside the cylinder
    mask = perpendicular_distances <= radius

    # Apply the mask to the point cloud to obtain the filtered points
    filtered_points = point_cloud.select_by_index(np.where(mask)[0])

    return filtered_points
#为了方便，两个相机的骨骼数据存一块了，取一台相机对应的骨骼和点云数据测试就行
skeletons = read_data()

for i in range(frame_count):
    # point_cloud0 = pcd0[i]
    # point_cloud1 = pcd1[i]
    #这就是那两个骨骼数据
    skeleton0, skeleton1, skeleton2, skeleton3 = skeletons[i]
    #这个skeleton0和skeleton1是32维的dict，每行代表一个joint的坐标
    #具体信息kinect官网上应该可以查到哪个点对应哪个点
    skeleton_joint_0 = np.asarray(skeleton0[6]['position'])
    skeleton_joint_1 = np.asarray(skeleton0[9]['position'])
    #输入的点云数据的路径：
    PtCl_path = r'D:\VR_24\Segmentation\4cams-data-ply-0907\frame-0-camera-3.ply'
    filtered_result = filter_points_by_cylinder(PtCl_path,skeleton_joint_0,skeleton_joint_1,0.1)

    #o3d.visualization.draw_geometries(filtered_result)
    print(np.asarray(filtered_result))
    print()