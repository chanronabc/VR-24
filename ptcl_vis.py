import open3d as o3d

# Read the PLY file
ply_file = r"D:\Azure Kinect SDK\redandblack_vox10_1080.ply"
point_cloud = o3d.io.read_point_cloud(ply_file)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])