import cv2
from VideoPlayer import VideoPlayer
from VideoWriter import VideoWriter
from LineDrawer import LineDrawer
from Model import Model
from LinePicker import LinePicker
from ModelTransformer import ModelTransformer

vp = VideoPlayer('resources/video/field3/WideWide - Clip 003.mp4')
frames = vp.extract_frames()
frames_with_line = []

modelImage = cv2.imread('resources/model/model_cfl.png')
model = Model(modelImage)
modelTr = ModelTransformer(model, frames[13], True)

# frame_transformed = cv2.warpPerspective(frames[13], modelTr.H, (modelTr.cols, modelTr.rows))
# cv2.imshow('model', cv2.addWeighted(modelTr.model, 1, frame_transformed, 1, 0))

model_frames = list()
mask_frames = list()

lp = LinePicker(frames[40])
first_point = lp.first_down_point
scrimmage = lp.scrimmage_point
print(first_point, scrimmage)
ld = LineDrawer(modelTr, first_point, scrimmage, model)

for index, frame in enumerate(frames[70:]):
    modelTr.new_frame(frame)
    output = ld.applyHomographyToPoint(frame, modelTr.H)
    frame_transformed = cv2.warpPerspective(frame, modelTr.H, (modelTr.cols, modelTr.rows))
    # mask, _ = modelTr.line_mask(frame)
    # mask_transformed = cv2.warpPerspective(modelTr.line_mask(frame), modelTr.H, (modelTr.cols, modelTr.rows))
    model_and_frame = cv2.addWeighted(modelTr.model, 1, frame_transformed, 1, 0)
    # model_and_mask = cv2.addWeighted(frame, 1, mask, 1, 0)
    # last_frame = frames[index-1]
    # last_model, _ = modelTr.line_mask(last_frame)
    # last_model_and_mask = cv2.addWeighted(last_frame, 1, last_model, 1, 0)
    frames_with_line.append(output)
    cv2.imshow('model', model_and_frame)
    # cv2.imshow('mask', model_and_mask)
    # cv2.imshow('last_mask', last_model_and_mask)
    model_frames.append(model_and_frame)
    # mask_frames.append(mask)
    cv2.waitKey(1)
    print(index)

cv2.destroyAllWindows()

# for frame in frames:
#    frames_with_line.append(LineDrawer(frame).draw_line())

# vw = VideoWriter('test_footage', frames_with_line)
vw = VideoWriter('aerial_footage', model_frames)
vw = VideoWriter('footage_line_test', frames_with_line)
