import cv2
import streamlit as st
import numpy as np
import random
import io
import argparse
from utls.cv_utils import *
import glob
import matplotlib.pyplot as plt

def create_imgs_slider(img_paths):
    val = st.sidebar.slider("frame number", 0, len(img_paths), 0, 1)
    return val

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

    options['display_hist'] = st.sidebar.checkbox("Display Histogram", False)
    options['hist_normalize'] = st.sidebar.checkbox("Perform Histogram Equalization", False)

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

    st.sidebar.header("Placing text")
    options['text'] = st.sidebar.text_input("Input your text here", "")
    options['pos_x'] = st.sidebar.number_input("X position with respect to image size", 0.0, 1.0, 0.5)
    options['pos_y'] = st.sidebar.number_input("Y position with respect to image size", 0.0, 1.0, 0.9)

    return options

# @st.cache(suppress_st_warning=True)
def create_video_slider(total_frames):
    val = st.sidebar.slider("frame number", 0, total_frames, 0, 1)
    return val

def display_image(frame, image_text):
    st.image(
        frame, caption=image_text, use_column_width=True,
    )

# @st.cache(suppress_st_warning=True)
def get_width_height():
    width = st.sidebar.number_input("Frame width", 10, 1500, 600)
    height = st.sidebar.number_input("Frame height", 10, 1500, 600)
    return width, height

def display_hist(hist):
    fig,ax = plt.subplots()
    ax.set_title("Histogram")
    if hist.shape[1] == 3:
        colors = ['r', 'g', 'b']
        labels = ['red', 'green', 'blue']
        for i, (col, label) in enumerate(zip(colors, labels)):
            ax.plot(hist[:, i], color=col, label=label)
    else:
        ax.plot(hist[:, 0], color='k', label='gray')
    ax.legend()
    st.pyplot(fig)