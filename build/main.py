import cv2
import preprocess
import laneDetection

def pipeline(frame):
    frame, invM = preprocess.warp(frame)
    frame = preprocess.grayscale(frame)
    frame = preprocess.threshold(frame)
    frame, left_curverad, right_curverad = laneDetection.search_around_poly(frame)
    return frame


def main(file):
    cap = cv2.VideoCapture(file)
    if(cap.isOpened()==False):
        print('Error opening the file, check its format')
    cap.set(1, 100)
    res, frame = cap.read()
    frame = pipeline(frame)
    cv2.imshow('Frame', frame)
    cv2.waitKey(10000)


if __name__ == "__main__":
    file = "..\\data\\dashcam_video_trim.mp4"
    main(file)