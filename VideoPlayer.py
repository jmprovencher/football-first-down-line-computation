import numpy as np
import cv2


class VideoPlayer:
    def __init__(self, path, color=True):
        print('Initializing video player...')
        self.path = path
        self.color = color
        print('ready!')

    def play_video(self):
        cap = cv2.VideoCapture(self.path)
        print('Playing video...')
        while cap.isOpened():
            ret, frame = cap.read()
            if self.color:
                cv2.imshow('Video Player', frame)

            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Video Player', gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Video stopped by user.')
                break

        cap.release()
        cv2.destroyAllWindows()

    def extract_frames(self):
        frames = []
        cap = cv2.VideoCapture(self.path)
        print('Extracting frames...')
        ret, frame = cap.read()
        frameCount = 0
        while ret:
            ret, frame = cap.read()
            if self.color:
                frames.append(frame)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Extraction stopped by user.')
                break
            frameCount+=1
        cap.release()
        cv2.destroyAllWindows()
        print('Total frames extracted: ',frameCount)
        print('Extraction done!')
        return frames




