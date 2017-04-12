import cv2
from VideoPlayer import VideoPlayer
from VideoWriter import VideoWriter
from LineDrawer import LineDrawer

vp = VideoPlayer('resources/video/field1/WideWide - Clip 001.mp4')

frames = vp.extract_frames()
frames_with_line = []
for frame in frames:
    frames_with_line.append(LineDrawer(frame).draw_line())

vw = VideoWriter('test_footage', frames_with_line)
