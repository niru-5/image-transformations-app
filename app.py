import cv2
import streamlit as st
import numpy as np
import argparse
from utls.cv_utils import *
from utls.display_utils import *

def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",type=str, default='./sample_data/video.mp4',
                        help='provide the path to data')
    return parser

def main():

    args = get_arg_parser().parse_args()

    imgs_flag = False
    video_flag = False
    if args.data.endswith('.mp4'):
        # print (args.data)
        cap, total_frames = open_video(args)
        read_video(cap,total_frames)
        video_flag = True
    elif args.data.endswith('.jpg') or args.data.endswith('.png'):
        frame = cv2.imread(args.data)
        # open_image()
    else:
        img_paths = sorted(glob.glob(args.data))
        imgs_flag = True
        # open_folder()


    if video_flag:
        val = create_video_slider(total_frames)
        frame = read_video(cap, val)

    if imgs_flag:
        val = create_imgs_slider(img_paths)
        frame = cv2.imread(img_paths[val])

    width, height = get_width_height()

    options = build_side_bar()

    frame = cv2.resize(frame, (width, height))
    frame = rotate_frame(frame, options['rotate_option'])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if options['gray_scale']:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if options['hsv'] and not options['gray_scale']:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    if options['hist_normalize']:
        frame = normalize_hist(frame)

    if options['color_apply']:
        frame = apply_color_transformations(frame, options['contrast'], options['brightness'])

    if options['kernel_apply']:
        frame = apply_kernel(frame, options['kernel_size'], options['kernel_type'])

    if options['thresh_apply']:
        frame = apply_threshold(frame, options['thresh_val'], options['thresh_max'], options['thresh_option'], options['thresh_pref'])

    if options['erosion_apply']:
        frame = transformations(frame, options['transformation_type'], options['tf_kernel_size'], options['tf_kernel_type'])



    frame = placing_text(frame, options['text'], position=(options['pos_x'], options['pos_y']))

    display_image(frame, "Image")

    if options['display_hist']:
        hist = calculcate_hist(frame)
        display_hist(hist)

if __name__ == "__main__":
    main()