import numpy as np
import cv2


class VideoPlayer:
    def __init__(self, path, color=True):
        self.path = path
        self.color = color

    def playVideo(self):
        cap = cv2.VideoCapture(self.path)
        while cap.isOpened():
            ret, frame = cap.read()
            if self.color:
                cv2.imshow('frame', frame)

            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

vp = VideoPlayer('resources/video/field1/WideWide - Clip 001.mp4')
vp.playVideo()

