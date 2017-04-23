import cv2
from VideoPlayer import VideoPlayer
from VideoWriter import VideoWriter
from LineDrawer import LineDrawer
from Model import Model
from ModelTransformer import ModelTransformer

vp = VideoPlayer('resources/video/field1/WideWide - Clip 001.mp4')
frames = vp.extract_frames()
frames_with_line = []

modelImage = cv2.imread('resources/model/model_cfl.png')
model = Model(modelImage)
modelTr = ModelTransformer(model, frames[13])

for frame in frames[14:]:
    modelTr.new_frame(frame)


#for frame in frames:
#    frames_with_line.append(LineDrawer(frame).draw_line())

#vw = VideoWriter('test_footage', frames_with_line)
