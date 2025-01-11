# Import the OpenCV library for computer vision tasks
from __learning__ import *


if __name__ == '__main__':
    # To apply LAB color space conversion and modify one of the channels, set 'convert' to True.
    # Pass 'l', 'a', or 'b' as the 'channel' parameter to specify which LAB channel should be zeroed out.
    # Use 'convert' set to False for a standard camera feed without any color space conversion.
    # camera(True, 'l')

    # Display the histogram for the specified image file.
    # The function will show the distribution of pixel intensities for the blue, green, and red channels.
    # Provide the path to the image as a string argument.
    # getHistogram("pictures/eagle.jpg")

    # Display the histogram for the specified image file.
    # Provide the path to the image as a string argument.
    # histogramsBB("pictures/eagle.jpg")

    # Provide the path to the image as a string argument.
    # smoothingFilter("pictures/eagle.jpg")

    # Sharpening Spatial Filter
    sharpeningFilter("pictures/doMoreQuality.jpg")

