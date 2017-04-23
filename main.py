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

frame_transformed = cv2.warpPerspective(frames[13], modelTr.H, (modelTr.cols, modelTr.rows))
cv2.imshow('model', cv2.addWeighted(modelTr.model, 1, frame_transformed, 1, 0))
cv2.waitKey()

for frame in frames[14:]:
    modelTr.new_frame(frame)
    frame_transformed = cv2.warpPerspective(frame, modelTr.H, (modelTr.cols, modelTr.rows))
    mask_transformed = cv2.warpPerspective(modelTr.line_mask(frame), modelTr.H, (modelTr.cols, modelTr.rows))
    cv2.imshow('model', cv2.addWeighted(modelTr.model, 1, frame_transformed, 1, 0))
    cv2.imshow('model_mask', cv2.addWeighted(modelTr.model, 1, mask_transformed, 1, 0))
    cv2.waitKey()

cv2.destroyAllWindows()


#for frame in frames:
#    frames_with_line.append(LineDrawer(frame).draw_line())

#vw = VideoWriter('test_footage', frames_with_line)
