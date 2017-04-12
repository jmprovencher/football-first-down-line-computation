import cv2
from VideoPlayer import VideoPlayer
from VideoWriter import VideoWriter

vp = VideoPlayer('resources/video/field1/WideWide - Clip 001.mp4')

frames = vp.extract_frames()

for frame in frames:
     cv2.imshow('Football Footage',frame)

vw = VideoWriter('test', frames)
