import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture

import cv2
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box, plot_crop
from rtree import index

REAL = 1
FAKE = 0

SPATIAL_THRESHOLD_DEGREE = 5

def main(args):
    #---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)
    mode = args.mode
    dataset_folder_path = args.inputDir
    spatial_threshold_degree = args.spatialThresholdDegree

    dataset_folder_list = os.walk(dataset_folder_path).next()[1]
    train_folder_path = args.outputDir
    train_folder_list = os.walk(train_folder_path).next()[1]
    for fname in dataset_folder_list:
        source_path = os.path.join(dataset_folder_path, fname)
        hs_warp_folder = os.path.join(source_path, "hs_raw_warp")
        rs_color_folder = os.path.join(source_path, "rs_raw_color")
        rs_depth_folder = os.path.join(source_path, "rs_raw_depth")

        hs_warp_list = sorted(glob(os.path.join(hs_warp_folder, '*.jpg')))
        rs_color_list = sorted(glob(os.path.join(rs_color_folder, '*.jpg')))
        rs_depth_list = sorted(glob(os.path.join(rs_depth_folder, '*.jpg')))

        if not hs_warp_list or not rs_color_list or not rs_depth_list:
            print("skip ", source_path)
            continue
        elif len(hs_warp_list) != len(rs_color_list) or len(hs_warp_list) != len(rs_depth_list):
            print("skip ", source_path)
            continue
        else:
            pass

        if mode == FAKE:
            new_fname = "1_" + fname[1:] + "_3_1_4"
        elif mode == REAL:
            new_fname = "1_" + fname[1:] + "_1_1_1"

        if new_fname not in train_folder_list:
            new_path = os.path.join(train_folder_path, new_fname)
            os.mkdir(new_path)
            new_depth_path = os.path.join(new_path, "depth")
            os.mkdir(new_depth_path)
            new_profile_path = os.path.join(new_path, "profile")
            os.mkdir(new_profile_path)
            new_rs_path = os.path.join(new_path, "rs")
            os.mkdir(new_rs_path)
            print("create new folder: {}".format(fname))
        else:
            print("skip {}".format(new_fname))
            continue

        spatial_coordinate_idx = index.Index(properties=p)
        count_num = 1
        total_num = len(hs_warp_list)
        for j in range(total_num):
            if j % 10 == 0:
                print("has processed {} of {} images".format(j, total_num))

            hs_warp_image = imread(hs_warp_list[j])
            [h, w, c] = hs_warp_image.shape
            if c>3:
                hs_warp_image = hs_warp_image[:,:,:3]

            # the core: regress position map
            if args.isDlib:
                max_size = max(hs_warp_image.shape[0], hs_warp_image.shape[1])
                if max_size> 1000:
                    hs_warp_image = rescale(hs_warp_image, 1000./max_size)
                    hs_warp_image = (hs_warp_image*255).astype(np.uint8)
                hs_pos = prn.process(hs_warp_image) # use dlib to detect face
            else:
                if hs_warp_image.shape[0] == hs_warp_image.shape[1]:
                    hs_warp_image = resize(hs_warp_image, (256,256))
                    hs_pos = prn.net_forward(hs_warp_image/255.) # input hs_warp_image has been cropped to 256x256
                else:
                    box = np.array([0, hs_warp_image.shape[1]-1, 0, hs_warp_image.shape[0]-1]) # cropped with bounding box
                    hs_pos = prn.process(hs_warp_image, box)

            hs_warp_image = hs_warp_image/255.
            if hs_pos is None:
                continue
            hs_vertices = prn.get_vertices(hs_pos)

            camera_matrix, euler_pose = estimate_pose(hs_vertices)
            # check similarity with previous pose
            hit = spatial_coordinate_idx.nearest((euler_pose[0], euler_pose[1], euler_pose[2], euler_pose[0], euler_pose[1], euler_pose[2]), 1, objects=True)
            hit = [i for i in hit]
            if hit:
                nearest_euler_pose = np.array(hit[0].bbox[:3])
                current_euler_pose = np.array(euler_pose)
                dist = np.linalg.norm(current_euler_pose - nearest_euler_pose)
                if dist > SPATIAL_THRESHOLD_DEGREE:
                    print("Get a new euler pose {}".format(euler_pose))
                    spatial_coordinate_idx.insert(0,(euler_pose[0], euler_pose[1], euler_pose[2], euler_pose[0], euler_pose[1], euler_pose[2]))
                else:
                    continue
            else:
                print("First euler_pose: {}".format(euler_pose))  
                spatial_coordinate_idx.insert(0,(euler_pose[0], euler_pose[1], euler_pose[2], euler_pose[0], euler_pose[1], euler_pose[2]))

            ##############################################
            #            
            ##############################################

            if mode == FAKE:
                imsave(os.path.join(new_profile_path, ('%04d' % count_num) + '.jpg'), plot_crop(hs_warp_image, hs_vertices))
                rs_depth_image = imread(rs_depth_list[j])
                imsave(os.path.join(new_depth_path, ('%04d' % count_num) + '.jpg'), plot_crop(rs_depth_image, hs_vertices))
            elif mode == REAL:
                rs_color_image = imread(rs_color_list[j])
                [h, w, c] = rs_color_image.shape
                if c>3:
                    rs_color_image = rs_color_image[:,:,:3]

                # the core: regress position map
                if args.isDlib:
                    max_size = max(rs_color_image.shape[0], rs_color_image.shape[1])
                    if max_size> 1000:
                        rs_color_image = rescale(rs_color_image, 1000./max_size)
                        rs_color_image = (rs_color_image*255).astype(np.uint8)
                    rs_pos = prn.process(rs_color_image) # use dlib to detect face
                else:
                    if rs_color_image.shape[0] == rs_color_image.shape[1]:
                        rs_color_image = resize(rs_color_image, (256,256))
                        rs_pos = prn.net_forward(rs_color_image/255.) # input rs_color_image has been cropped to 256x256
                    else:
                        box = np.array([0, rs_color_image.shape[1]-1, 0, rs_color_image.shape[0]-1]) # cropped with bounding box
                        rs_pos = prn.process(rs_color_image, box)

                rs_color_image = rs_color_image/255.
                if rs_pos is None:
                    continue
                rs_vertices = prn.get_vertices(rs_pos)

                rs_depth_image = imread(rs_depth_list[j])
                imsave(os.path.join(new_profile_path, ('%04d' % count_num) + '.jpg'), plot_crop(hs_warp_image, rs_vertices))
                imsave(os.path.join(new_depth_path, ('%04d' % count_num) + '.jpg'), plot_crop(rs_depth_image, rs_vertices))
                imsave(os.path.join(new_rs_path, ('%04d' % count_num) + '.jpg'), plot_crop(rs_color_image, rs_vertices))

            count_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='/home/yangchihyuan/anaconda3/dataset/all_dataset/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='/home/yangchihyuan/anaconda3/dataset/phase3/train/', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=False, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=True, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=True, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--isTxt', default=False, type=ast.literal_eval,
                        help='whether to save txt')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')
    parser.add_argument('--mode', default=1, type=int,
                        help='real=1, fake=0')
    parser.add_argument('--spatialThresholdDegree', default=5, type=int,
                        help='threshold degree for pose selection')

    p = index.Property()
    p.dimension = 3

    main(parser.parse_args())
