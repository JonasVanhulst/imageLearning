# Import the OpenCV library for computer vision tasks
import cv2
import matplotlib.pyplot as plt

# Import the constants variables
from __CONSTANTS__ import *


def camera(convert: bool, channel: str):
    """
    Captures video from the default camera and displays it in a window. Optionally, the frame can be converted to the LAB color space,
    and one of the channels (L, a, or b) can be zeroed out based on the input.

    :param convert: A boolean value indicating whether to convert the frame from BGR to LAB color space.
                    If True, the frame will be converted, and one channel will be modified.
    :param channel: A string specifying the channel to be zeroed out when the frame is converted to LAB color space.
                    It should be one of 'l', 'a', or 'b' (case-insensitive), corresponding to the LAB color channels.

    :return: None. The function continuously displays the video feed from the camera in a window.
             The program can be exited by pressing the 'Esc' key.
    """

    # Open the default camera (index 0) for video capture
    cap: any = cv2.VideoCapture(0)

    # Check if the camera is opened correctly to prevent errors
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    try:
        # Continuously read frames from the camera until an error occurs or the program is stopped
        while True:
            # Read a frame from the camera and store it in the 'frame' variable
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from camera")
                break

            if convert:
                # Set the channel to lowercase
                channel.lower()

                # Convert the frame from BGR color space to LAB color space
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

                # Split the LAB image into its individual channels (L, a, b)
                l, a, b = cv2.split(frame)

                if channel == 'l':
                    # Set the L channel to zero, effectively removing luminance information
                    l = l * ZERO_CHANNEL
                elif channel == 'a':
                    # Set the A channel to zero, effectively removing luminance information
                    a = a * ZERO_CHANNEL
                elif channel == 'b':
                    # Set the B channel to zero, effectively removing luminance information
                    b = b * ZERO_CHANNEL

                # Merge the modified L channel with the original a and b channels
                frame = cv2.merge((l, a, b))

            # Display the captured frame in a window titled "Webcam Feed"
            cv2.imshow("Webcam Feed", frame)

            # Exit the program when the 'Esc' key is pressed
            if cv2.waitKey(1) == ESCAPE_KEY:
                break

    finally:
        # Release system resources to prevent memory leaks
        cap.release()
        cv2.destroyAllWindows()


def getHistogram(image: str):
    """
    Plots the histogram for the blue, green, and red color channels of an image.

    :param image: The file path to the image for which the histogram is to be calculated.
                      It should be a valid path to an image file (e.g., .jpg, .png).

    :return: None. The function displays a plot of the histograms for each color channel.
             The plot shows the distribution of pixel intensities for the blue, green, and red channels.
             :param image:
    """

    # Read the image from the specified file path
    img = cv2.imread(image)

    # Define the color channels for which to calculate the histogram
    color = ("b", "g", "r")  # Corresponds to blue, green, and red channels

    # Loop through each color channel to calculate and plot its histogram
    for i, col in enumerate(color):
        # Calculate the histogram for the current color channel
        # Parameters:
        # [img]: Source image
        # [i]: Index of the color channel (0 for blue, 1 for green, 2 for red)
        # None: No mask is used, so the histogram is calculated for the whole image
        # [256]: Number of bins for the histogram
        # [0,256]: Range of pixel intensity values
        hist = cv2.calcHist([img], [i], None, [MAX_PIXELS], [MIN_PIXELS, MAX_PIXELS])

        # Plot the histogram with the corresponding color
        plt.plot(hist, color=col)

        # Set the x-axis limits for the histogram (0 to 256 intensity values)
        plt.xlim([MIN_PIXELS, MAX_PIXELS])

    # Display the plotted histograms
    plt.show()
