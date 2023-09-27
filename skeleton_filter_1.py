import numpy as np
import os
import json
import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
import skimage
from skimage import io, color
from skimage.metrics import structural_similarity as SSIM
import matplotlib.pyplot as plt

frame_count = 5

BASE_PATH = ".\\data_ply_0924"
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


def take_picture(point_cloud, p, camera_parameters):
    """
    Use Open3D to take a picture of the point cloud."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 设置相机位置
    camera = vis.get_view_control()
    camera.set_lookat(camera_parameters["lookat"])  # 设置观察点位置
    camera.set_up(camera_parameters["set_up"])      # 设置相机的上向量
    camera.set_front(camera_parameters["set_front"])  # 设置相机的前向向量
    camera.set_zoom(camera_parameters["set_zoom"])          # 设置缩放级别

    vis.add_geometry(point_cloud)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    for i in range(4):
        # point_cloud_rotate = point_cloud.get_rotation_matrix_from_xyz((np.pi/2 *1, 0, 0))
        # point_cloud.rotate(point_cloud_rotate, center=(0.0,0.0,0.0))
        point_cloud.scale(0.8, center=(0.0,0.0,0.0))
        vis.update_geometry(point_cloud)
        # 渲染图像
        vis.poll_events()
        vis.update_renderer()

        image_path = p

        # 保存渲染的图像
        vis.capture_screen_image(image_path+ "_" + str(i) + ".png")

    # vis.update_geometry(point_cloud)
    # # 渲染图像
    # vis.poll_events()
    # vis.update_renderer()
    #
    # image_path = p
    #
    # # 保存渲染的图像
    # vis.capture_screen_image(image_path+ "_" + str(0) + ".png")

    # while True:
    #   # 渲染图像
    # vis.poll_events()
    # vis.update_renderer()
    # params = camera.convert_to_pinhole_camera_parameters()
    # print(params)
    # camera_position = params.intrinsic.intrinsic_matrix
    # print(camera.get_lookat())
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
  ssim_value = SSIM(gt_image, pred_image, data_range=1.0)
  # print(ssim_value)
  return ssim_value


#读取数据
def read_data():
    path = BASE_PATH
    files = os.listdir(path)

    point_files = [f for f in files if 'point_cloud' in f and 'points' in f]

    pcd0, pcd1, skeletons = [], [], []
    for i in range(5):

        with open(os.path.join(path, f'frame-0-skeleton_{1}.json'), 'r') as fr:
            skeleton = json.load(fr)
        # with open(os.path.join(path, f'frame-{i}-skeleton_1.json'), 'r') as fr:
        #     skeleton1 = json.load(fr)

        # pcd0.append(point_cloud0)
        # pcd1.append(point_cloud1)
        if isinstance(skeleton, list):
            for j in range(len(skeleton)):
                skeletons.append(skeleton[j])
        else:
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


def find_trunk(point_cloud, other_part):
    new_point_indexes = []
    other_part_point = []
    points = point_cloud.points
    for i in range(len(other_part)):
        new_point_indexes.append([])
        other_part_point.append(np.asarray(other_part[i].points))
        points = np.asarray(points)
        # for j, point in enumerate(point_cloud):
        #     min_distance = min(np.linalg.norm(point - other_part[i], axis=1))
        #     if min_distance > distance_threshold:
        #         new_point_indexes.append(j)

        for j, point in enumerate(points):
            if point not in other_part_point[i]:
                new_point_indexes[i].append(j)
        # print(new_point_indexes)

    new_index = []
    for i in range(len(new_point_indexes)):
        for j in range(len(new_point_indexes[i])):
            new_index.append(new_point_indexes[i][j])

    new_points = [point_cloud.points[j] for j in new_index]
    new_colors = [point_cloud.colors[j] for j in new_index]  # If you have color information

    # Create a new point cloud with the selected points
    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(new_points)
    new_point_cloud.colors = o3d.utility.Vector3dVector(new_colors)  # If you have color information

    return new_point_cloud


# def crop_background(Image, image_name):
#     Image = skimage.color.rgb2gray(Image)
#     contours = skimage.measure.find_contours(Image, 0.5)
#     fig, ax1 = plt.subplots(1, 1)
#     rows, cols = Image.shape
#     ax1.axis([0, rows, cols, 0])
#     for n, contour in enumerate(contours):
#         ax1.plot(contour[:, 1], contour[:, 0], linewidth=1, color='black')
#     ax1.axis('image')
#     plt.axis('off')
#
#     # 保存图片
#     plt.rcParams['savefig.dpi'] = 100
#     # 将轮廓保存为无背景的镂空图
#     plt.savefig(image_name, transparent=True, bbox_inches='tight', pad_inches=0.0)


def find_background_position(Image):
    '''

    :param Image:
    :return: four parameter, the minimum and maximum column and row in the Image except whit background.

    '''

    min_x, min_y = 99999, 99999
    max_x, max_y = 0, 0
    # 扫描整张图片，保留非白色的最大与最小行、列
    for i in range(len(Image)):
        for j in range(len(Image[i])):
            item = Image[i][j]
            # 判断像素点颜色是否接近白色
            if item[0] <= 220 or item[1] <= 220 or item[2] <= 220:
                if j < min_x:
                    min_x = j
                if j > max_x:
                    max_x = j
                if i < min_y:
                    min_y = i
                if i > max_y:
                    max_y = i
    print(str(min_x) + " " + str(max_x) + " "+ str(min_y )+ " "+ str(max_y))
    return min_x, max_x, min_y, max_y


def crop_background(Image, min_x, max_x, min_y, max_y):
    new_image = np.zeros([max_y - min_y, max_x - min_x, 3])

    # 根据之前记录的最小与最大行列，把位于原Image的数据记录入新Image的数据
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            new_image[j-min_y ][i-min_x] = Image[j][i]
    return new_image


def compare_Images(Image1_name, Image2_name):
    Image1 = skimage.io.imread(Image1_name+".png")
    min_x1, max_x1, min_y1, max_y1 = find_background_position(Image1)
    Image2 = skimage.io.imread(Image2_name+".png")
    min_x2, max_x2, min_y2, max_y2 = find_background_position(Image2)

    # 比较两个图形的最大于最小行列，保留较大边框的那一个行列
    if min_x1 > min_x2:
        min_x = min_x2
    else:
        min_x = min_x1
    if max_x1 > max_x2:
        max_x = max_x1
    else:
        max_x = max_x2
    if min_y1 > min_y2:
        min_y = min_y2
    else:
        min_y = min_y1
    if max_y1 > max_y2:
        max_y = max_y1
    else:
        max_y = max_y2

    Image1 = crop_background(Image1, min_x, max_x, min_y, max_y)
    Image2 = crop_background(Image2, min_x, max_x, min_y, max_y)

    skimage.io.imsave(Image1_name+".png", Image1)
    skimage.io.imsave(Image2_name+".png", Image2)
    ssim = get_image_ssim(Image1_name+".png", Image2_name+".png")
    return ssim


#为了方便，两个相机的骨骼数据存一块了，取一台相机对应的骨骼和点云数据测试就行
skeletons = read_data()

SSIM_result = []
for i in range(frame_count):
    # point_cloud0 = pcd0[i]
    # point_cloud1 = pcd1[i]
    #这就是那两个骨骼数据
    skeleton0, skeleton1, skeleton2, skeleton3, skeleton4 = skeletons

    #visualization
    with open(f"{BASE_PATH}\\frame-0-skeleton_{1}.json", "r") as json_file:
        skeleton_data = json.load(json_file)
    skeleton_joint_pos = []
    connections = []
    joint_connections = [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6], [6,7], [7,8], [8,9], [7,10], [2,11],
                         [11,12], [12,13], [13,14], [14,15], [15,16], [14,17], [0,18], [18,19], [19, 20],
                         [20,21], [0,22], [22,23], [23, 24], [24, 25], [3, 26], [26,27], [26,28], [26,29],
                         [26, 30], [26,31]]
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
    point_cloud = o3d.io.read_point_cloud(f'{BASE_PATH}\\frame-0-point_cloud_{5}.ply')
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(skeleton_joint_pos_array)
    line_set.lines = o3d.utility.Vector2iVector(connections)
    # o3d.visualization.draw_geometries([point_cloud, skeleton_cloud, line_set])

    #输入的点云数据的路径：
    # PtCl_path = f'{BASE_PATH}\\frame-0-point_cloud_1.ply'
    PtCl_path = f'{BASE_PATH}\\frame-0-point_cloud_{5}.ply'

    point_cloud = o3d.io.read_point_cloud(PtCl_path)
    print("原始点云 point count: ", np.asarray(point_cloud))

    # 这个skeleton0和skeleton1是32维的dict，每行代表一个joint的坐标
    # 具体信息kinect官网上应该可以查到哪个点对应哪个点
    skeleton_joint_0 = np.asarray(skeleton_data[0][0]['position'])
    skeleton_joint_1 = np.asarray(skeleton_data[0][31]['position'])

    # 以 skeleton_joint_0 和 skeleton_joint_1 为圆柱体的两端，过滤出人体
    filtered_result = filter_points_by_cylinder(PtCl_path,skeleton_joint_0,skeleton_joint_1,500)
    print("过滤出人体后 point count:", np.asarray(filtered_result))
    # o3d.visualization.draw_geometries([filtered_result, skeleton_cloud, line_set])
    # 存储整个人的照片为 ground truth.png

    camera_parameters = {"window_name":"ground_truth", "lookat": [0,0,0], "set_up":[0,0,1], "set_front":[100,-1,0], "set_zoom":8.0}
    take_picture(filtered_result, 'gt', camera_parameters)

    # # ======= 1. 全身下采样 ==========
    filtered_result = filter_points_by_cylinder(PtCl_path,skeleton_joint_0,skeleton_joint_1,500)
    downsampled = downsample_point_cloud(filtered_result, 0.05) # 按 10% 下采样
    print("after downsample point count:", np.asarray(downsampled))

    # 1.1 存储全身下采样的照片（@boyan TODO 2: 调整 Open3d take picture camera 位置，找5个典型 从不同角度，不同远近给画面拍照）

    take_picture(downsampled, 'down', camera_parameters)
    # 1.2 比较全身下采样的照片，关于原始照片的 SSIM_result（@Boyan TODO 3：调整 Open3D take picture 参数只把人切出来比，去掉白色背景）
    ssim0 = compare_Images("gt_0", "down_0")
    SSIM_result.append(ssim0)
    # # ======= 1. 全身下采样 end ==========

    # @Boyan TODO 1
    # ======= 2. 单个关节下采样 ==========
    # 2.1 先把五个 body part 的点分别拆出来，也就是找到对应  skeleton 下标始尾
    body_part_start_ends = [ # https://learn.microsoft.com/en-us/azure/kinect-dk/body-joints
        (5, 9), (12, 16), (18, 21), (22, 25), (30, 0), (28, 3)]
        # 3-脖子 5-左肩膀 9-左手指尖 12-右肩膀，16-右手指尖 18-左髋部 21-左足 22-右髋部 25-右足

    body_part = []

    for body_part_start_end in body_part_start_ends:
        skeleton_joint_0 = np.asarray(skeleton1[body_part_start_end[0]]['position'])
        skeleton_joint_1 = np.asarray(skeleton1[body_part_start_end[1]]['position'])

        # 预览一下拆出来的效果，最后一个参数 50 是圆柱的半径
        filtered_result = filter_points_by_cylinder(PtCl_path,skeleton_joint_0,skeleton_joint_1,60)
        body_part.append(filtered_result)
        # o3d.visualization.draw_geometries([filtered_result, skeleton_cloud, line_set])

    skeleton_joint_0 = np.asarray(skeleton1[25]['position'])
    skeleton_joint_1 = np.asarray(skeleton1[26]['position'])
    filtered_result = filter_points_by_cylinder(PtCl_path, skeleton_joint_0, skeleton_joint_1, 300)
    body_trunk = find_trunk(filtered_result, body_part)
    body_part.append(body_trunk)
    # o3d.visualization.draw_geometries([body_part[0], body_part[1], body_part[2], body_part[3], body_part[4], skeleton_cloud, line_set])

    Arms_0 = body_part[0] + body_part[1]
    Legs_0 = body_part[2] + body_part[3]
    Head_0 = body_part[4] + body_part[5]
    Trunk_0 = body_part[6]
    # point_cloud1 = body_part[0] + body_part[1] + body_part[2] + body_part[3] + body_part[4] + body_part[5]
    # print("原始点云 point count: ", np.asarray(point_cloud1))

    take_picture(Arms_0, "Arms_gt", camera_parameters)
    take_picture(Legs_0, "Legs_gt", camera_parameters)
    take_picture(Head_0, "Head_gt", camera_parameters)
    take_picture(Trunk_0, "Trunk_gt", camera_parameters)

    body_part[0] = downsample_point_cloud(body_part[0], 0.05)  # 左臂
    body_part[1] = downsample_point_cloud(body_part[1], 0.05)  # 右臂
    body_part[2] = downsample_point_cloud(body_part[2], 0.05)  # 左腿
    body_part[3] = downsample_point_cloud(body_part[3], 0.05)  # 右腿
    body_part[4] = downsample_point_cloud(body_part[4], 0.05)  # 头部
    body_part[5] = downsample_point_cloud(body_part[5], 0.05)  # 躯干

    Arms_1 = body_part[0] + body_part[1]
    Legs_1 = body_part[2] + body_part[3]
    Head_1 = body_part[4] + body_part[5]
    Trunk_1 = body_part[6]

    # o3d.visualization.draw_geometries([body_part[0], body_part[1], body_part[2], body_part[3], body_part[4], skeleton_cloud, line_set])
    # point_cloud2 = body_part[0] + body_part[1] + body_part[2] + body_part[3] + body_part[4] + body_part[5]
    # print("after downsample point count:", np.asarray(point_cloud2))

    take_picture(Arms_1, "Arms_down", camera_parameters)
    take_picture(Legs_1, "Legs_down", camera_parameters)
    take_picture(Head_1, "Head_down", camera_parameters)
    take_picture(Trunk_1, "Trunk_down", camera_parameters)
    # body_part.append(find_trunk(PtCl_path, body_part))
    ssim1 = compare_Images("Arms_gt_0", "Arms_down_0")
    ssim2 = compare_Images("Legs_gt_0", "Legs_down_0")
    ssim3 = compare_Images("Head_gt_0", "Head_down_0")
    ssim4 = compare_Images("Trunk_gt_0", "Trunk_down_0")
    SSIM_result.append([ssim1, ssim2, ssim3, ssim4])

print(SSIM_result)
SSIM_result = np.asarray(SSIM_result)
SSIM_result = np.transpose(SSIM_result)
SSIM_Average = np.zeros((5, 1))
print(SSIM_result)
for i in range(len(SSIM_Average)):
    SSIM_Average[i] = np.mean(SSIM_result[i])
print(SSIM_Average)
    # 2.2 再对不同 body part 下采样
    # 1) 先按这个做：下采样这一部分，然后只评估这一部分，画面中其他删掉的下采样 SSIM（问题：选取角度和距离，观察每个 body part 的时候不一样？）

    # 先不做2.3 再拼起来？晚上 discuss 一下，确定方案

    # 3. 下采样完成后，算 SSIM
