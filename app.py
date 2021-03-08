import cv2
import streamlit as st
import numpy as np
import random
import io
import argparse
from utls.cv_utils import *

def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",type=str, default='/hdd/mai/computer_vision/assignment_1/video_3.mp4',
                        help='provide the path to data')
    return parser

def main():

    args = get_arg_parser().parse_args()

    if args.data.endswith('.mp4'):
        # print (args.data)
        cap, total_frames = open_video(args)
        read_video(cap,total_frames)
        video_flag = True
    elif args.data.endswith('.jpg') or args.data.endswith('.png'):
        print(args.data)
        # open_image()
    else:
        print(args.data)
        # open_folder()


    if video_flag:
        val = create_video_slider(total_frames)
        frame = read_video(cap, val)
    width, height = get_width_height()

    options = build_side_bar()

    frame = cv2.resize(frame, (width, height))
    frame = rotate_frame(frame, options['rotate_option'])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if options['gray_scale']:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if options['hsv']:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    if options['kernel_apply']:
        frame = apply_kernel(frame, options['kernel_size'], options['kernel_type'])

    if options['thresh_apply']:
        frame = apply_threshold(frame, options['thresh_val'], options['thresh_max'], options['thresh_option'])

    if options['erosion_apply']:
        frame = transformations(frame, options['transformation_type'], options['tf_kernel_size'], options['tf_kernel_type'])

    display_image(frame, "Image")


def apply_threshold(frame, thresh_val, thresh_max, thresh_option):
    if thresh_option == "binary":
        frame = threshold_img(frame, 'gray',thresh_val, thresh_max, cv2.THRESH_BINARY)
        return frame
    if thresh_option == "inverse binary":
        frame = threshold_img(frame, 'gray',thresh_val, thresh_max, cv2.THRESH_BINARY_INV)
        return frame
    if thresh_option == "truncated":
        frame = threshold_img(frame, 'gray',thresh_val, thresh_max, cv2.THRESH_TRUNC)
        return frame
    if thresh_option == "to zero":
        frame = threshold_img(frame, 'gray',thresh_val, thresh_max, cv2.THRESH_TOZERO)
        return frame
    if thresh_option == "inverse to zero":
        frame = threshold_img(frame, 'gray',thresh_val, thresh_max, cv2.THRESH_TOZERO_INV)
        return frame



def rotate_frame(frame, rotate_option):
    if rotate_option == "rotate_0":
        return frame
    elif rotate_option == "rotate_90":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
    elif rotate_option == "rotate_180":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame
    elif rotate_option == "rotate_270":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

def build_side_bar():
    options = {}
    options['rotate_option'] = st.sidebar.selectbox("select the rotate option",
                                         ("rotate_0", "rotate_90", "rotate_180", "rotate_270"))

    options['gray_scale'] = st.sidebar.checkbox("Convert to gray scale", False)
    options['hsv'] = st.sidebar.checkbox("Convert to HSV", False)

    st.sidebar.header("Apply Various Kernels")
    # kernel size, kernel type
    options['kernel_apply'] = st.sidebar.checkbox("Should the kernel be applied", False)
    options['kernel_size'] = st.sidebar.number_input("Choose kernel size", 3, 15, 5, 2)
    options['kernel_type'] = st.sidebar.selectbox("select the Kernel Type",
                                       ("average_blur", "gaussian_blur", "median_blur", "bilateral_filter"))

    st.sidebar.header("Apply Threshold functions")
    options['thresh_apply'] = st.sidebar.checkbox("Should the threshold be applied", False)
    options['thresh_val'] = st.sidebar.number_input("Choose Threshold value", 0, 255, 128, 1)
    options['thresh_max'] = st.sidebar.number_input("Choose Threshold max value", options['thresh_val'], 255, 255, 1)
    options['thresh_option'] = st.sidebar.selectbox("select the Kernel Type",
                                         ("binary", "inverse binary", "truncated", "to zero", "inverse to zero"))

    st.sidebar.header("Apply erosion/dilation")
    options['erosion_apply'] = st.sidebar.checkbox("Should the erosion/dilation be applied", False)
    options['tf_kernel_size'] = st.sidebar.number_input("Choose transformation kernel size", 3, 15, 5, 2)
    options['tf_kernel_type'] = st.sidebar.selectbox("select the Kernel Type",
                                          ("average", "ellipse", "cross"))
    options['transformation_type'] = st.sidebar.selectbox("select the transformation Type",
                                               ("erosion", "dilation", "opening", "closing"))
    return options

# @st.cache(suppress_st_warning=True)
def create_video_slider(total_frames):
    val = st.sidebar.slider("frame number", 0, total_frames, 0, 1)
    return val

def display_image(frame, image_text):
    st.image(
        frame, caption=image_text, use_column_width=True,
    )


def open_video(args):
    cap = cv2.VideoCapture(args.data)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, total_frames

def read_video(cap,val):
    # widht,height = get_width_height()
    cap.set(1, val)
    ret, frame = cap.read()
    return frame

# @st.cache(suppress_st_warning=True)
def get_width_height():
    width = st.sidebar.number_input("Frame width", 10, 1500, 600)
    height = st.sidebar.number_input("Frame height", 10, 1500, 600)
    return width, height

if __name__ == "__main__":
    main()