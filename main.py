import cv2
from VideoPlayer import VideoPlayer

vp = VideoPlayer('resources/video/field1/WideWide - Clip 001.mp4')

frames = vp.extract_frames()

print(frames)

for frame in frames:
    cv2.imshow('Football Footage',frame)
    cv2.waitKey(0)
