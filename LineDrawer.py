import cv2
import numpy as np


class LineDrawer:
    def __init__(self, modelTr=None, point=None, model=None):
        print('Drawing...')
        self._modelTr = modelTr
        self._point = point
        self._model = model

        self.homogeneous_point = np.array([self._point], dtype=np.float32)
        self.point_in_model = cv2.perspectiveTransform(np.array([self.homogeneous_point]), self._modelTr.H)

    def draw_line(self, image, pt1, pt2):
        height, width, depth = image.shape
        line_mask = np.zeros((height, width), np.uint8)
        line_image = np.zeros((height, width, 3), np.uint8)
        cv2.line(line_image, pt1, pt2, (43, 124, 220), 10)
        cv2.line(line_mask, pt1, pt2, (255, 255, 255), 5)

        field_mask = self.mask_builder(image, 38, 88, 34, 101, 0, 174)
        field_mask_inv = cv2.bitwise_not(field_mask)

        lineToDraw = cv2.addWeighted(line_mask,1,field_mask_inv,-1,0)
        lineToDraw_inv = cv2.bitwise_not(lineToDraw)
        line = cv2.bitwise_and(line_image, line_image, mask=lineToDraw)
        # field = image - mask_inv_line
        field = cv2.bitwise_and(image, image, mask=lineToDraw_inv)
        output = cv2.addWeighted(line, 1, field, 1, 0)
        cv2.imshow('output', output)

        return output

    def applyHomographyToPoint(self, frame, H):
        start_line = (self.point_in_model[0][0][0].astype(int), self._model.params['upperLineY'])
        end_line = (self.point_in_model[0][0][0].astype(int), self._model.params['lowerLineY'])

        tupletest = (self.point_in_model[0][0][0], self.point_in_model[0][0][1])
        cv2.circle(self._modelTr.model, tupletest, 3, (43, 124, 220))
        res, inv_H = cv2.invert(H)
        model_line_points = np.array([start_line, end_line], dtype=np.float32)
        field_line_points = cv2.perspectiveTransform(np.array([model_line_points]), inv_H)
        start_line_field = (field_line_points[0][0][0], field_line_points[0][0][1])
        end_line_field = (field_line_points[0][1][0], field_line_points[0][1][1])
        return self.draw_line(frame, start_line_field, end_line_field)

    def mask_builder(self, frame, hl, hh, sl, sh, vl, vh):
        # load image, convert to hsv
        bgr = frame
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # set lower and upper bounds of range according to arguements
        lower_bound = np.array([hl, sl, vl], dtype=np.uint8)
        upper_bound = np.array([hh, sh, vh], dtype=np.uint8)
        return cv2.inRange(hsv, lower_bound, upper_bound)
