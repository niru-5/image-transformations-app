import cv2
import numpy as np

def placing_text(img, text, position=(0.5,0.9), color=(255,255,255), font=cv2.FONT_HERSHEY_PLAIN, size=1, thickness=None ):
    h,w = img.shape[:2]
    # print (img.shape)
    x_pos,y_pos = position
    img = cv2.putText(img, text, (int(w*x_pos),int(y_pos*h) ), font,size, color, thickness=thickness)
    return img

def convert_to_gray(img, option='gray'):
    if option == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    else:

        (b,g,r) = cv2.split(img)
        if option == 'blue':
            return b
        elif option == 'green':
            return g
        else:
            return r

def calculcate_hist(img):
    if len(img.shape) < 3:
        channels = 1
    else:
        _,_,channels = img.shape
    hist = np.zeros((256, channels))
    for i in range(channels):
        hist[:, i] = cv2.calcHist([img], [i], None, [256], [0, 256])[:, 0]
    return hist

# function for gaussian blur
def apply_kernel(img, kernel_size=5,type='filter2D', kernel = None):
    if type == 'filter2D':
        # kernel = np.ones((kernel_size,kernel_size),dtype=np.float32)/kernel_size
        return cv2.filter2D(img, -1, kernel)
    elif type == 'average_blur':
        return cv2.blur(img, (kernel_size,kernel_size))
    elif type == 'gaussian_blur':
        return cv2.GaussianBlur(img, (kernel_size,kernel_size),sigmaX=0, sigmaY=0)
    elif type == 'median_blur':
        return cv2.medianBlur(img, kernel_size)
    elif type == "bilateral_filter":
        return cv2.bilateralFilter(img, kernel_size, 75,75)

def threshold_img(img, img_type='gray', threshold_val=128, max_val=255, option=cv2.THRESH_BINARY):
    if img_type == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(img, threshold_val, max_val, option)
        return thresh
    elif img_type == 'color':
        (b,g,r) = cv2.split(img)
        (b_col,g_col,r_col) = threshold_val
        (b_max, g_max, r_max) = max_val
        ret_b, thresh_b = cv2.threshold(b, b_col, b_max, option)
        ret_g, thresh_g = cv2.threshold(g, g_col, g_max, option)
        ret_r, thresh_r = cv2.threshold(r, r_col, r_max, option)
        new_img = cv2.merge((thresh_b,thresh_g,thresh_r))
        return new_img

def transformations(img, type='erosion', kernel_size=5, kernel_type='average', iterations = 1):
    kernel = get_kernel(kernel_size, kernel_type)
    if type == 'erosion':
        img1 = cv2.erode(img, kernel, iterations=1)
        return img1
    elif type == 'dilation':
        return cv2.dilate(img, kernel, iterations=1)
    elif type == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif type == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def get_kernel(kernel_size, kernel_type):
    if kernel_type == 'average':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return kernel
    elif kernel_type == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return kernel
    elif kernel_type == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        return kernel

def sobel_edge_detector(frame, sobel_kernel_size=5, sobel_x_derivative_order=1, sobel_y_derivative_order=1, sobel_depth=cv2.CV_16S):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(frame, sobel_depth, sobel_x_derivative_order,0,ksize=sobel_kernel_size)
    grad_y = cv2.Sobel(frame, sobel_depth, 0, sobel_y_derivative_order, ksize=sobel_kernel_size)

    grad_x_abs = cv2.convertScaleAbs(grad_x)
    grad_y_abs = cv2.convertScaleAbs(grad_y)

    final_output = cv2.addWeighted(grad_x_abs,0.5, grad_y_abs,0.5,0)
    return final_output

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
