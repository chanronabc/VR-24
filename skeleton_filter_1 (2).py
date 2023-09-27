import numpy as np
import os
import json
import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
import skimage
from skimage import io, color
from skimage.metrics import structural_similarity as ssim

frame_count = 1

BASE_PATH = ".\\4cams-data-ply-0907"
import random

def downsample_point_cloud(point_cloud, downsample_percentage):
    """
    Downsample a point cloud to a given percentage of its original size.

    Args:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        downsample_percentage (float): The percentage of points to keep (0.0 to 1.0).

    Returns:
        open3d.geometry.PointCloud: The downsampled point cloud.
    """
    # Calculate the number of points to keep based on the percentage
    total_points = len(point_cloud.points)
    points_to_keep = int(downsample_percentage * total_points)

    # Randomly select the points to keep
    random_indices = random.sample(range(total_points), points_to_keep)
    downsampled_points = [point_cloud.points[i] for i in random_indices]
    downsampled_colors = [point_cloud.colors[i] for i in random_indices]  # If you have color information

    # Create a new point cloud with the selected points
    downsampled_point_cloud = o3d.geometry.PointCloud()
    downsampled_point_cloud.points = o3d.utility.Vector3dVector(downsampled_points)
    downsampled_point_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors)  # If you have color information

    return downsampled_point_cloud


def take_picture(point_cloud, p):
  """
  Use Open3D to take a picture of the point cloud."""
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  vis.add_geometry(point_cloud)

  # 设置相机位置
  camera = vis.get_view_control()
  camera.set_lookat([0, 0, 0])  # 设置观察点位置
  camera.set_up([0, 0, 1])      # 设置相机的上向量
  camera.set_front([100, -1, 0])  # 设置相机的前向向量
  camera.set_zoom(4)          # 设置缩放级别

  # 渲染图像
  vis.poll_events()
  vis.update_renderer()

  # 保存渲染的图像
  image_path = p
  vis.capture_screen_image(image_path)

  # while True:
  #   # 渲染图像
  #   vis.poll_events()
  #   vis.update_renderer()
  #   # params = camera.convert_to_pinhole_camera_parameters()
  #   # print(params)
  #   # camera_position = params.intrinsic.intrinsic_matrix
  #   # print(camera.get_lookat())
  # 关闭渲染器窗口
  vis.destroy_window()


def get_image_ssim(gt_image, pred_image):
  # Load the GT and predicted images
  gt_image = io.imread(gt_image)
  pred_image = io.imread(pred_image)

  # Convert images to grayscale if they are not already
  if len(gt_image.shape) == 3:
      gt_image = color.rgb2gray(gt_image)
  if len(pred_image.shape) == 3:
      pred_image = color.rgb2gray(pred_image)

  # Calculate SSIM
  ssim_value = ssim(gt_image, pred_image, data_range=1.0)
  # print(ssim_value)
  return ssim_value



#读取数据
def read_data():
    path = BASE_PATH
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

    #visualization
    with open(f"{BASE_PATH}\\frame-0-skeleton_0.json", "r") as json_file:
        skeleton_data = json.load(json_file)
    skeleton_joint_pos = []
    connections = []
    joint_connections = [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6], [6,7], [7,8], [8,9], [7,10], [2,11], [11,12], [12,13], [13,14], [14,15], [15,16], [14,17], [0,18], [18,19], [19, 20], [20,21], [0,22], [22,23], [23, 24], [24, 25], [3, 26], [26,27], [26,28], [26,29], [26, 30], [26,31]]
    for frame_data in skeleton_data:
        for joint_info in frame_data:
            positions = joint_info['position']
            joint_pos_np = np.asarray(positions)
            skeleton_joint_pos.append(joint_pos_np)
        connections.extend(joint_connections)
        break
    skeleton_joint_pos_array = np.asarray(skeleton_joint_pos)
    skeleton_cloud = o3d.geometry.PointCloud()
    skeleton_cloud.points = o3d.utility.Vector3dVector(skeleton_joint_pos_array)
    point_cloud = o3d.io.read_point_cloud(f'{BASE_PATH}\\frame-0-camera-0.ply')
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(skeleton_joint_pos_array)
    line_set.lines = o3d.utility.Vector2iVector(connections)
    #o3d.visualization.draw_geometries([point_cloud, skeleton_cloud, line_set])

    #输入的点云数据的路径：
    PtCl_path = f'{BASE_PATH}\\frame-0-camera-0.ply'

    point_cloud = o3d.io.read_point_cloud(PtCl_path)
    print("原始点云 point count: ", np.asarray(point_cloud))


    # 这个skeleton0和skeleton1是32维的dict，每行代表一个joint的坐标
    # 具体信息kinect官网上应该可以查到哪个点对应哪个点
    skeleton_joint_0 = np.asarray(skeleton0[0]['position'])
    skeleton_joint_1 = np.asarray(skeleton0[31]['position'])

    # 以 skeleton_joint_0 和 skeleton_joint_1 为圆柱体的两端，过滤出人体
    filtered_result = filter_points_by_cylinder(PtCl_path,skeleton_joint_0,skeleton_joint_1,500)
    print("过滤出人体后 point count:", np.asarray(filtered_result))
    o3d.visualization.draw_geometries([filtered_result, skeleton_cloud, line_set])
    # 存储整个人的照片为 ground truth.png
    take_picture(filtered_result, 'gt.png')

    # # ======= 1. 全身下采样 ==========
    filtered_result = filter_points_by_cylinder(PtCl_path,skeleton_joint_0,skeleton_joint_1,500)
    downsampled = downsample_point_cloud(filtered_result, i * 0.1) # 按 10% 下采样
    print("after downsample point count:", np.asarray(downsampled))

    # 1.1 存储全身下采样的照片（@boyan TODO 2: 调整 Open3d take picture camera 位置，找5个典型 从不同角度，不同远近给画面拍照）
    take_picture(downsampled, 'down.png')

    # 1.2 比较全身下采样的照片，关于原始照片的 SSIM（@Boyan TODO 3：调整 Open3D take picture 参数只把人切出来比，去掉白色背景）
    ssim = get_image_ssim('gt.png', 'down.png')
    print(ssim)

    # # ======= 1. 全身下采样 end ==========

    # @Boyan TODO 1
    # ======= 2. 单个关节下采样 ==========
    # 2.1 先把五个 body part 的点分别拆出来，也就是找到对应  skeleton 下标始尾
    # body_part_start_ends = [ # https://learn.microsoft.com/en-us/azure/kinect-dk/body-joints
    #     [(5, 9), (12, 16), (18, 21), (22, 25)], # 12-肩膀，16-手指尖
    # ]
    # for body_part_start_end in body_part_start_ends:
    #     skeleton_joint_0 = np.asarray(skeleton0[body_part_start_end[0]]['position'])
    #     skeleton_joint_1 = np.asarray(skeleton0[body_part_start_end[1]]['position'])
    #
    #     # 预览一下拆出来的效果，最后一个参数 50 是圆柱的半径
    #     filtered_result = filter_points_by_cylinder(PtCl_path,skeleton_joint_0,skeleton_joint_1,50)
    #     o3d.visualization.draw_geometries([filtered_result, skeleton_cloud, line_set])

    # 2.2 先不做：再对不同 body part 下采样
    # 1) 先按这个做：下采样这一部分，然后只评估这一部分，画面中其他删掉的下采样 SSIM（问题：选取角度和距离，观察每个 body part 的时候不一样？）
    # 2) 先不管：下采样这一部分，其他 body part 不下采样，评估整个画面SSIM（问题：是否能有效反映motivation）
    # 2.3 再拼起来？晚上 discuss 一下，确定方案

    # 3. 下采样完成后，算 SSIM
