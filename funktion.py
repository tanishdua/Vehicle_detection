import cv2
import numpy as np
import lane_detect_pls as ldp
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

def actual_dist(right, left):
    width_in_reference_image = 302
    distance_in_reference_image = 4
    dist_act = (distance_in_reference_image*(right-left))/width_in_reference_image
    return(dist_act)

def horizontal_trans(right, left, right0, left0,tpf):
    hor_trans = ((right+left)/2 - (right0-left0)/2)*tpf
    return(hor_trans)

def collision(UpLt,RtLt,LtLt,d1,d0,right,left,right0,left0,bottom):
    wtext1 = ""
    wtext2 = ""
    top = 150
    speed = 7.5  #mps = 27 kmph
    tpf = 0.14   #averaged out value for time taken per frame
    #d0 = 0
    #d0 = d1
    #d1 = actual_dist()
    # UpLt = 473        ##########
    # RtLt = 843        ##########      From Lane Detect Pls
    # LtLt = 298        ##########
    middle_lane = (RtLt+LtLt)/2
    width_in_reference_image = 300
    distance_in_reference_image = 4
    #d1 = actual_dist()
    rel_vel = (d1 - d0)*1/tpf
    actual_vel = rel_vel + speed
    
    if(actual_vel < -5):
        wtext1 = 'Collision Imminent'
    elif(actual_vel > -5 and bottom<UpLt):
        wtext1 = 'Apply Brakes'
    else:
        wtext1 = 'Safe'
    k = horizontal_trans(right, left, right0, left0, tpf)
    if(middle_lane>(right+left)/2 and k>0):
        wtext2 = 'Look Left'
    elif(middle_lane<(right+left)/2 and k<0):
        wtext2 = 'Look Right'
    
    return wtext1,wtext2

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
        result,UpLt, RtLt, LtLt = ldp.lanedetect(result1)
        collision(UpLt, RtLt, LtLt)
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
	#video_path = "project_video.mp4"
    video_path = "D:\Documents\study\ML\innovative\yolo_lane\challenge_video.mp4"
    detect_video(video_path)
    cv2.destroyAllWindows()


