from re import I
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def gaussian_blur(img,ksize):
    return cv2.GaussianBlur(img,(ksize,ksize),0)
def conv2canny(img,low_thresh, high_thresh):
    return cv2.Canny(img,low_thresh, high_thresh)

def ymask(img, poi):
    mask = np.zeros_like(img)
    ignore_mask_colour = 255
    cv2.fillPoly(mask,poi,ignore_mask_colour)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img,lines

def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def lanedetect(frame):

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame
    #print(img.shape)
    #img = frame[1:719,300:900]

    grayimg = grayscale(img)
    blurimg = gaussian_blur(grayimg,5)
    #kernel size = 5


    low_thresh = 100
    high_thresh = 200
    edgeimg = conv2canny(blurimg, low_thresh, high_thresh)
    k = np.where(edgeimg.sum(axis = 1) == max(edgeimg.sum(axis = 1)[0:650]))
    lowerlimitY = k[0][0]
    upperlimitY = 650        #cropping out the lower line


    poi = np.array([[0,lowerlimitY], [0,upperlimitY], [1200,upperlimitY], [1200,upperlimitY]])



    lowerLeftPoint = [300, 650]
    upperLeftPoint = [300, 475]
    upperRightPoint = [900, 475]
    lowerRightPoint = [900, 650]
    poi = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, lowerRightPoint]], dtype=np.int32)
    ymaskimg = ymask(edgeimg,poi)
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 20 
    max_line_gap = 300

    houged,lines = hough_lines(ymaskimg, rho, theta, threshold, min_line_len, max_line_gap)

    houged2 = cv2.cvtColor(houged,cv2.COLOR_BGR2GRAY)   #Converted to grey to reduce dimensions

    # ##   EXTRACTION OF POINTS FROM THE GRAPH   ##
    arr = houged2.sum(axis = 1)
    #print(type(arr))
    k0 = 0
    k1 = 0
    for i in range(len(np.diff(arr))):
        if(np.diff(arr)[i]>0):
            k0 = i
            break
    #print(k0)
    rev_arr = arr[::-1]
    #print(rev_arr)
    #print(np.diff(rev_arr))
    for i in range(len(np.diff(rev_arr))):
        if(np.diff(rev_arr)[i]>0):
            k1 = i
            break
    #print(720-k1)

    arr1 = houged2.sum(axis = 0)
    #plt.plot(arr1)
    #plt.show()
    k2 = 0
    k3 = 0
    #plt.plot(np.diff(arr1))
    #plt.show()

    for i in range(len(np.diff(arr1))):
        if(np.diff(arr1)[i]>0):
            k2 = i
            break
    #print(k2)
    rev_arr1 = arr1[::-1]
    for i in range(len(np.diff(rev_arr1))):
        if(np.diff(rev_arr1)[i]>0):
            k3 = i
            break
    #print(1280-k3)
    UpLt = k0
    RtLt = 1280-k3
    LtLt = k2
    #print(UpLt, RtLt, LtLt)

    added_image = cv2.addWeighted(frame,1,houged,1,0)
    return added_image,UpLt, RtLt, LtLt
    #cv2.imshow("final",added_image)


def detect_video(video_path):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        result1 = np.asarray(image)
        result = lanedetect(result1)
        #result = np.asarray(image)

        """FPS CALC"""

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
    
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    video_path = "project_video.mp4"
    detect_video(video_path)
    cv2.destroyAllWindows()
    