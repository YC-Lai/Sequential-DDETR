from cProfile import label
from turtle import pos
import util
import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import imageio
from PIL import Image
from numpy.linalg import inv, norm
from plyfile import PlyData
import matplotlib.pyplot as plt

mesh_file = '/home/aicenteruav/Sequential-DDETR/data/scannet/scene0006_00/scene0006_00_vh_clean_2.ply'
label_file = '/home/aicenteruav/Sequential-DDETR/data/scannet/scene0006_00/label-filt/0.png'
depth_file = '/home/aicenteruav/Sequential-DDETR/data/scannet/scene0006_00/depth/0.png'
intrinsic_file = '/home/aicenteruav/Sequential-DDETR/data/scannet/scene0006_00/intrinsic/intrinsic_depth.txt'
pose_file = '/home/aicenteruav/Sequential-DDETR/data/scannet/scene0006_00/pose/0.txt'
colors = util.create_color_palette()
num_colors = len(colors)
depth_img = np.array(imageio.imread(depth_file)).astype(np.float32) / 1000.0
label_img = np.array(imageio.imread(label_file)).astype(np.uint8)
intrinsic = np.loadtxt(intrinsic_file, dtype=np.float32)
pose = np.loadtxt(pose_file, dtype=np.float32)
camera_position = inv(pose)
# print(camera_position)
camera_position = -inv(camera_position[:3, :3])@camera_position[:3, 3]

# h, w = depth_img.shape[0], depth_img.shape[1]
# label_img_resize = np.array(Image.fromarray(label_img).resize((h, w)))

# u = np.arange(0, w)
# v = np.arange(0, h)
# grid_u, grid_v = np.meshgrid(u, v, indexing='xy')

# X = (grid_u - intrinsic[0, 2]) * depth_img / intrinsic[0, 0]
# Y = (grid_v - intrinsic[1, 2]) * depth_img / intrinsic[1, 1]
# X, Y, Z = np.ravel(X), np.ravel(Y), np.ravel(depth_img)

# homo_coords = pose @ np.stack((X, Y, Z, np.ones_like(X)), axis=0)
# coordinates = homo_coords[:3] / homo_coords[3]
# all_point = np.ravel(label_img_resize)
# print(all_point.shape)
# print(coordinates.shape)
# exit()
inversez_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
rotationz_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
rotationy_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
mesh = o3d.io.read_triangle_mesh(mesh_file)
displaysList = [mesh]
points_camera = np.array([[0, 0, 1], [480, 0, 1], [480, 640, 1], [0, 640, 1]])
points_camera = np.linalg.pinv(intrinsic[:3, :3])@points_camera.T
points_camera = inversez_matrix@points_camera
points_camera = rotationz_matrix@points_camera
points_camera = rotationy_matrix@points_camera
points_camera = camera_position.reshape(3, 1) + pose[:3, :3]@points_camera
points_camera = points_camera.T
all_points = np.ones((points_camera.shape[0]+1, points_camera.shape[1]))
all_points[:-1, :] = points_camera

all_points[-1, :] = camera_position
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(all_points),
    lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
)
color = [0, 0, 0]
colors = np.tile(color, (8, 1))
line_set.colors = o3d.utility.Vector3dVector(colors)
displaysList.append(line_set)
o3d.visualization.draw_geometries(displaysList)

# print(vertices.shape)
# o3d.visualization.draw_geometries([mesh])
with open(mesh_file, 'rb') as f:
    plydata = PlyData.read(f)
    num_verts = plydata['vertex'].count
    # vertices = plydata['vertex']
# num_verts = vertices.shape[0]
    vertexs = np.ones((num_verts, 3))
    vertexs[:, 0] = plydata['vertex']['x']
    vertexs[:, 1] = plydata['vertex']['y']
    vertexs[:, 2] = plydata['vertex']['z']
    # for i in range(num_verts):
    #     vertexs[i][0] = plydata['vertex'][i][0]
    #     vertexs[i][1] = plydata['vertex'][i][1]
    #     vertexs[i][2] = plydata['vertex'][i][2]
camera_position = np.repeat(camera_position[:, None].T, num_verts, axis=0)
# point_camera_depth = norm((vertexs[:, :3] - camera_position[:, :3]), axis=1)
# point_camera_depth = point_camera_depth.T
vertex_img = np.zeros((480, 640, 3), dtype=np.uint8)
vertexs_camera = vertexs.T - camera_position.T
vertexs_camera = np.linalg.pinv(pose[:3, :3])@vertexs_camera
# vertexs_camera = inv(pose) @ vertexs.T
vertexs_camera = np.linalg.pinv(inversez_matrix) @ np.linalg.pinv(rotationz_matrix) @np.linalg.pinv(rotationy_matrix) @ vertexs_camera
vertexs_projection = intrinsic[:3, :3] @ vertexs_camera

vertexs_pixels = vertexs_projection[:3, :] / vertexs_projection[2, :]
vertexs_pixels = vertexs_pixels[:2]
# points_depth = vertexs
pixel_count = 0
label_dict = {}
for i, vertex_each in enumerate(vertexs_pixels.T):
    vertex_each_v, vertex_each_u = round(vertex_each[1]), round(vertex_each[0])
    if(vertex_each_u >= 0 and vertex_each_u < 480 and vertex_each_v >= 0 and vertex_each_v < 640):
        Depth_Point = vertexs_camera[2][i]
        Depth_Img = depth_img[vertex_each_u, vertex_each_v]

#         print(Depth_Point)
        if(abs(Depth_Point - Depth_Img) < 1):
            # print("get depth")
            vertex_img[vertex_each_u, vertex_each_v][0] = plydata['vertex']['red'][i]
            vertex_img[vertex_each_u, vertex_each_v][1] = plydata['vertex']['green'][i]
            vertex_img[vertex_each_u, vertex_each_v][2] = plydata['vertex']['blue'][i]
            if(i not in label_dict):
                label_dict[i] = [2]
            else:
                label_dict[i].append[2]
            pixel_count += 1
keys = label_dict.keys()
keys.sort()
label_dict = []
print(pixel_count)
plt.imsave("testnew.png", vertex_img)
# print(pixel_count)
