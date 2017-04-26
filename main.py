import cv2
import numpy as np
from VideoPlayer import VideoPlayer
from VideoWriter import VideoWriter
from LineDrawer import LineDrawer
from Model import Model
from LinePicker import LinePicker
from ModelTransformer import ModelTransformer

vp = VideoPlayer('resources/video/field1/WideWide - Clip 004.mp4')
frames = vp.extract_frames()
frames_with_line = []
field_lines_mask = []

modelImage = cv2.imread('resources/model/model_cfl.png')
model = Model(modelImage)
modelTr = ModelTransformer(model, frames[13], True)

point = (136, 260)
lp = LinePicker(frames[40])
first_point = lp.first_down_point
scrimmage = lp.scrimmage_point
print(first_point, scrimmage)
ld = LineDrawer(modelTr, first_point, scrimmage, model)

start_index = 70

all_homo = list()
for index, frame in enumerate(frames[start_index:]):
    modelTr.new_frame(frame)
    global_lines = modelTr.find_global_lines(frame)
    #mask = modelTr.line_mask(np.zeros(frame.shape), global_lines)
    #field_lines_mask.append(mask)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(1)
    all_homo.append(modelTr.H)


filter_size = 5
mean_homo = list()
for i in range(filter_size):
    mean_homo.append(all_homo[i])

for i in range(filter_size, len(all_homo) - filter_size):
    new_homo = all_homo[i]
    for j in range(filter_size):
        new_homo += all_homo[i - j]
        new_homo += all_homo[i + j]

    new_homo /= filter_size*2 + 1
    mean_homo.append(new_homo)

for i in range(len(all_homo)-filter_size, len(all_homo)):
    mean_homo.append(all_homo[i])

for index, frame in enumerate(frames[start_index:]):
    homo = mean_homo[index]
    output = ld.applyHomographyToPoint(frame, homo)
    frames_with_line.append(output)
    cv2.imshow('lines', output)
    print(index)
    cv2.waitKey(1)

cv2.destroyAllWindows()
vw = VideoWriter('footage_line_test', frames_with_line)
#vw = VideoWriter('mask_of_field_lines', field_lines_mask)
