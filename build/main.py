import cv2
import preprocess
import calibrateCamera
import laneDetection
from moviepy.editor import VideoFileClip

def calibrate():
    #Insert atleast 20 images for calibration from your camera, needs to run only once.
    #fname = '..\\data\\calibration\\*.jpg'
    #objpoints, imgpoints = calibrateCamera.pointExtractor(fname)
    #return objpoints, imgpoints
    return 0

def pipeline(frame):
    image = frame

    #Disabled, techinically each frame needs to be undistored before being processed.
    #objpoints, imgpoints = [] #Add them manually
    #frame = calibrateCamera.calibrate(objpoints, imgpoints, frame)

    frame, invM = preprocess.warp(frame)
    frame = preprocess.grayscale(frame)
    frame = preprocess.threshold(frame)
    frame, left_curverad, right_curverad = laneDetection.search_around_poly(frame)
    frame = cv2.warpPerspective(frame, invM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    frame = cv2.addWeighted(frame, 0.3, image, 0.7, 0)

    #Add curvature and distance from the center
    curvature = (left_curverad + right_curverad) / 2
    car_pos = image.shape[1] / 2
    center = (abs(car_pos - curvature)*(3.7/650))/10
    curvature = 'Radius of Curvature: ' + str(round(curvature, 2)) + 'm'
    center = str(round(center, 3)) + 'm away from center'
    frame = cv2.putText(frame, curvature, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, center, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def debugFrames(file):
    cap = cv2.VideoCapture(file)
    if(cap.isOpened()==False):
        print('Error opening the file, check its format')
    cap.set(1, 100)
    res, frame = cap.read()
    #frame = pipeline(objpoints, imgpoints, frame) uncomment if using for
    frame = pipeline(frame)
    cv2.imshow('Frame', frame)
    cv2.waitKey(10000)

def processFrames(infile, outfile):
    output = outfile
    clip = VideoFileClip(infile)
    processingClip = clip.fl_image(pipeline)
    processingClip.write_videofile(output, audio=True)

def main(infile, outfile):
    #objpoints, imgpoints = calibrate() uncomment, provided you have calibration pictures
    processFrames(infile, outfile)

if __name__ == "__main__":
    infile = "..\\data\\dashcam_video_trim.mp4"
    outfile = "..\\data\\dashcam_video_trim_output.mp4"
    main(infile, outfile)