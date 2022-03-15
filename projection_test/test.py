import cv2
from distutils import core
from gc import garbage

from cv2 import line
import util
import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import imageio
from PIL import Image
import PIL
from numpy.linalg import inv, norm
from plyfile import PlyData
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from scipy import ndimage
import sys
# sys.path.remove('/home/aicenteruav/catkin_ws/devel/lib/python2.7/dist-packages')


def load_mesh(scene_name: Path):
    mesh_suffix = "scene0694_01_vh_clean_2.ply"

    mesh_path = scene_name / mesh_suffix

    return o3d.io.read_triangle_mesh(str(mesh_path))


def load_vertex(scene_name: Path):
    mesh_suffix = "scene0694_01_vh_clean_2.ply"

    mesh_path = scene_name / mesh_suffix

    with open(mesh_path, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count

        vertexs = np.ones((num_verts, 3))
        vertexs[:, 0] = plydata['vertex']['x']
        vertexs[:, 1] = plydata['vertex']['y']
        vertexs[:, 2] = plydata['vertex']['z']

    return vertexs, plydata


def get_preprocessing_map(label_map):
    mapping = dict()
    with open(label_map) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:

            if not row["nyu40id"].isnumeric():
                mapping[int(row["id"])] = 40
            else:
                mapping[int(row["id"])] = int(row["nyu40id"])

    return mapping


def rawCategory_to_nyu40(label, map: dict):
    """Remap a label image from the 'raw_category' class palette to the 'nyu40' class palette """

    nyu40_label = label
    keys = np.unique(label).astype(np.int)
    print(np.unique(label))
    for key in keys:
        if key != 0:
            nyu40_label[label == key] = map[key]
    return nyu40_label


def load_label(scene_name: Path, img_index: int, size):
    label_path = scene_name / "label-filt" / (str(img_index)+".png")

    label_img = imageio.imread(label_path)
    label = np.array(label_img).astype(np.uint16)

    label = cv2.resize(label, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    tsv_map = get_preprocessing_map("/home/aicenteruav/Sequential-DDETR/data/scannet/scannet-labels.combined.tsv")

    return rawCategory_to_nyu40(label, tsv_map)


def load_depth(scene_name: Path, img_index: int):
    depth_path = scene_name / "depth" / (str(img_index)+".png")

    return np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0


def load_color(scene_name: Path, img_index: int):
    color_path = scene_name / "color" / (str(img_index)+".jpg")

    return np.array(imageio.imread(color_path))


def load_intrinsic(scene_name: Path, intrinsic_type: str):
    intrinsic_path = scene_name / "intrinsic" / ("intrinsic_"+intrinsic_type+".txt")

    return np.loadtxt(intrinsic_path, dtype=np.float32)


def load_pose(scene_name: Path, img_index: int):
    pose_path = scene_name / "pose" / (str(img_index)+".txt")
    pose_matrix = np.loadtxt(pose_path, dtype=np.float32)

    # calculate the true camera translation
    inv_pose = np.linalg.pinv(pose_matrix)
    camera_translation = -np.linalg.pinv(inv_pose[:3, :3])@inv_pose[:3, 3]

    return pose_matrix, camera_translation


# TODO check if ever frame is using same modified matrix
def rotation_matrix():

    inversez_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    rotationz_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotationy_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    return rotationy_matrix@rotationz_matrix@inversez_matrix


def visualizeCameraPosition(mesh, intrinsic, pose_matrix, translation, depth_img):

    camera_corner = np.array([[0, 0, 1],
                              [0, depth_img.shape[0], 1],
                              [depth_img.shape[1], depth_img.shape[0], 1],
                              [depth_img.shape[1], 0, 1]])

    camera_corner_3d = np.linalg.pinv(intrinsic[:3, :3])@camera_corner.T
    # camera_corner_3d = rotation_matrix()@camera_corner_3d
    camera_corner_3d = translation.reshape(3, 1) + pose_matrix[:3, :3]@camera_corner_3d
    camera_corner_3d = camera_corner_3d.T

    # 5 = 4 corners + 1 center
    visualized_points = np.ones((5, 3))
    visualized_points[:-1, :] = camera_corner_3d
    visualized_points[-1, :] = translation

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(visualized_points),
        lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )

    color = [1, 0, 0]
    colors = np.tile(color, (8, 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def projectToFrame(vertexs, intrinsic, pose_matrix, translation):

    vertexs_translated = vertexs.T - translation.T
    vertexs_camera_coordinate = np.linalg.pinv(pose_matrix[:3, :3])@vertexs_translated
    # vertexs_camera_coordinate = np.linalg.pinv(rotation_matrix()) @ vertexs_camera_coordinate
    vertexs_camera_projection = intrinsic[:3, :3] @ vertexs_camera_coordinate

    vertexs_pixels = vertexs_camera_projection[:3, :] / vertexs_camera_projection[2, :]
    vertexs_pixels = vertexs_pixels[:2]

    return vertexs_pixels, vertexs_camera_coordinate


def visualizeProjection(vertex_frame, vertex_CC, vertexs, vertexs_ply, depth_img):
    # vertex no. 0 ~ 100
    vertex_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
    for i, vertex_each in enumerate(vertex_frame.T):
        vertex_each_v, vertex_each_u = round(vertex_each[1]), round(vertex_each[0])
        if(vertex_each_u >= 0 and vertex_each_u < 480 and vertex_each_v >= 0 and vertex_each_v < 640):
            Depth_Point = vertex_CC[2][i]
            Depth_Img = depth_img[vertex_each_u, vertex_each_v]

            if(abs(Depth_Point - Depth_Img) < 1 and Depth_Point > 0):
                vertex_img[vertex_each_u, vertex_each_v][0] = vertexs_ply['vertex']['red'][i]
                vertex_img[vertex_each_u, vertex_each_v][1] = vertexs_ply['vertex']['green'][i]
                vertex_img[vertex_each_u, vertex_each_v][2] = vertexs_ply['vertex']['blue'][i]

            # vertex[50].color = labelTocolor(label)

    plt.imsave("testnewcolor.png", vertex_img)
    


def label_mesh(mesh_labels, vertex_frame, vertex_CC, mesh,
               depth_img, label_img, label_colors, linesets, 
               viz, ids, color_img):
    print(np.unique(label_img))
    vertex_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    for i, vertex_each in enumerate(vertex_frame.T):
        vertex_each_v, vertex_each_u = round(vertex_each[0]), round(vertex_each[1])
        if(vertex_each_u >= 0 and vertex_each_u < 480 and vertex_each_v >= 0 and vertex_each_v < 640):
            Depth_Point = vertex_CC[2][i]
            Depth_Img = depth_img[vertex_each_u, vertex_each_v]

            if(abs(Depth_Point - Depth_Img) < 1 and Depth_Point > 0):
                if(label_img[vertex_each_u, vertex_each_v] != 0 and label_img[vertex_each_u, vertex_each_v] != 40):
                    vertex_img[vertex_each_u, vertex_each_v] = label_colors[label_img[vertex_each_u, vertex_each_v]]
                    mesh.vertex_colors[i] = np.array(label_colors[label_img[vertex_each_u, vertex_each_v]])/255
                mesh_labels[i][label_img[vertex_each_u, vertex_each_v]] += 1
        # else:
            # mesh.vertex_colors[i] = [1, 1, 1]

    # displaysList = [mesh, linesets]



    viz.update_geometry(mesh)
    viz.update_geometry(linesets)
    viz.poll_events()
    viz.update_renderer()
    # viz.add_geometry(linesets)
   


    
    # viz.remove_geometry(mesh)
    # viz.remove_geometry(mesh)

    # o3d.visualization.draw_geometries(displaysList)
    # plt.imsave(f"testnew{ids}.png", vertex_img)
    # plt.imsave("testnewL.png", label_img)
    plt.imsave(f"testnewD{ids}.png", 5*depth_img)
    plt.imsave(f"color{ids}.png", color_img)
    viz.capture_screen_image(f"mesh{ids}.png")
    return mesh


def init_viz():


    viz = o3d.visualization.Visualizer()
    viz.create_window()
    
    
    return viz


if __name__ == '__main__':

    viz = init_viz()
    scene_name = Path("/home/aicenteruav/Sequential-DDETR/data/scannet/scene0694_01/")
    scene_mesh = load_mesh(scene_name)
    scene_mesh_vertexs, scene_mesh_ply = load_vertex(scene_name)
    # prepare some data
    for text_id in range(0, 400, 20):
        # text_id = 50

        print(text_id)

        single_depth_img = load_depth(scene_name, text_id)
        single_color_img = load_color(scene_name, text_id)
        color_intrinsic = load_intrinsic(scene_name, "color")
        depth_intrinsic = load_intrinsic(scene_name, "depth")

        # COLLECT MESH LABEL COLOR
        mesh_labels = np.zeros((scene_mesh_vertexs.shape[0], 41))

        single_label_img = load_label(scene_name, text_id, single_depth_img.shape)

        pose_matrix, single_camera_translation = load_pose(scene_name, text_id)

        label_colors = util.create_color_palette()

        # visualize
        # visualizeCameraPosition(scene_mesh, depth_intrinsic, pose_matrix,
        # single_camera_translation, single_depth_img)

        camera_translation_stack = np.repeat(single_camera_translation[:, None].T,
                                            scene_mesh_vertexs.shape[0], axis=0)

        vertex_projection, vertex_camera_coord = projectToFrame(scene_mesh_vertexs, depth_intrinsic,
                                                                pose_matrix, camera_translation_stack)

        # visualizeProjection(vertex_projection, vertex_camera_coord, scene_mesh_vertexs,
        # scene_mesh_ply, single_depth_img)
        
        lines = visualizeCameraPosition(scene_mesh, depth_intrinsic, pose_matrix,
                                        single_camera_translation, single_depth_img)
        

        viz.add_geometry(scene_mesh)
        viz.add_geometry(lines)
    
        viz_extrinsic = np.array([[0.29230591640903653, -0.62232938153940409, 0.72612904645459286, 0.0],
                                [-0.95257587001974819, -0.12230626737306148, 0.27864024981576002, 0.0],
                                [-0.084595881036805376, -0.77314120174393253, -0.62856679762580092, 0.0],
                                [1.78451689354381, 3.6328365570750529, 5.954964044176613, 1.0]]).T


        control = viz.get_view_control()
        param = control.convert_to_pinhole_camera_parameters()
        param.extrinsic = viz_extrinsic
        control.convert_from_pinhole_camera_parameters(param)





        scene_mesh = label_mesh(mesh_labels, vertex_projection, vertex_camera_coord, scene_mesh,
                                single_depth_img, single_label_img, label_colors, lines, viz, text_id, single_color_img)
        
        viz.remove_geometry(lines)

        # main()
