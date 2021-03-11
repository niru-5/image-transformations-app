# image-transformations-app
A streamlit app to visualize images/video and apply basic image transformations to get some insights.
The app is built on streamlit and on the backend, opencv functions are called to be applied on the image. 
![](display_images/output.gif)

## Usage
A video could be analysed with the following command
```
streamlit run app.py --data path_to_video_file
```
or a folder of images could also be viewed
```
streamlit run app.py --data path_to_folder_containing_images
```

## Requirements
There are a few requirements before we can fully start using the list
```
opencv
streamlit
numpy 
glob
matplotlib
argparse
```

## Features
The repo has the following features to play with
### Rotate
This functions provides the option of rotation across various orientations like 90,180,270

### Convert to grayscale
Conversion to grayscale

### Convert to HSV
Conversion to HSV

### Display Histogram and perform histogram equalization
This provides the display of the histogram, depending upon the channels of the image. 
It also provides an extra option of seeing what would histogram equalization would do. 

### Thresholding functions

### Applying various filters

### Applying Image transformations
