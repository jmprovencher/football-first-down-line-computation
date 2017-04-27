import cv2
import numpy as np
from VideoPlayer import VideoPlayer
from VideoWriter import VideoWriter
from LineDrawer import LineDrawer
from Model import Model
from LinePicker import LinePicker
from ModelTransformer import ModelTransformer
from HSVTrackbar import HSVPicker

if __name__ == '__main__':

    vp = VideoPlayer('resources/video/field4/WideWide - Clip 006.mp4')
    frames = vp.extract_frames()
    frames_with_line = []
    field_lines_mask = []

    lp = LinePicker(frames[60])
    first_point = lp.first_down_point
    scrimmage = lp.scrimmage_point
    print(first_point, scrimmage)
    HSV_LOW, HSV_HIGH = HSVPicker(frames[80]).getHSVMask()

    modelImage = cv2.imread('resources/model/model_cfl.png')
    model = Model(modelImage)
    modelTr = ModelTransformer(model, frames[13], [HSV_LOW, HSV_HIGH], True)

    ld = LineDrawer(modelTr, first_point, scrimmage, model, [HSV_LOW, HSV_HIGH])

    start_index = 14

    all_homo = list()
    for index, frame in enumerate(frames[start_index:]):
        try:
            modelTr.new_frame(frame)
            #global_lines = modelTr.find_global_lines(frame)
            #mask = modelTr.line_mask(np.zeros(frame.shape), global_lines)
            #field_lines_mask.append(mask)
            #cv2.imshow('mask', mask)
            #cv2.waitKey(1)
            all_homo.append(modelTr.H)
        except Exception as e:
            print(e)
            break


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

    for index in range(len(mean_homo)):
        homo = mean_homo[index]
        frame = frames[index]
        output = ld.applyHomographyToPoint(frame, homo)
        frames_with_line.append(output)
        cv2.imshow('lines', output)
        print(index)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    vw = VideoWriter('footage_line_test', frames_with_line)
    #vw = VideoWriter('mask_of_field_lines', field_lines_mask)
