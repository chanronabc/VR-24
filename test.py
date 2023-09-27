import numpy as np
import open3d as o3d

RED = [1.0, 0.0, 0.0, 1.0]
GREEN = [0.0, 0.5, 0.0, 1.0]  # dark green is easier to see on white when small
BLUE = [0.0, 0.0, 1.0, 1.0]

def add_cloud(scene, name, npts, center, radius, color, size):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)

    material = o3d.visualization.Material()
    material.po_color = color
    material.point_size = size
    scene.add_geometry(name, cloud, material)

def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background_color([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)

    add_cloud(widget3d.scene, "green", 5000, [0, 0, 0], 10.0, GREEN, 2)
    add_cloud(widget3d.scene, "red", 500, [-2, -2, -2], 5.0, RED, 4)
    add_cloud(widget3d.scene, "blue", 250, [1, 1, 1], 5.0, BLUE, 6)

    widget3d.setup_camera(60, widget3d.scene.bounding_box, [0, 0, 0])

    app.run()

if __name__ == "__main__":
    main()