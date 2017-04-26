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

        # field_mask = self.mask_builder(image, 48, 53, 47, 79, 85, 175)
        # field_mask_inv = cv2.bitwise_not(field_mask)
        mask_inv_line = cv2.bitwise_not(line_mask)

        # lineToDraw = line_mask - field_mask_inv
        # field = image - mask_inv_line
        # result = cv2.add(lineToDraw, field)
        # cv2.imshow('test', result)

        alpha = 1  # Alpha defines the desired opacity of the line

        # Apply inversed mask of the line to get everything expect the line in image
        frame_not_line_region = cv2.bitwise_and(image, image, mask=mask_inv_line)

        rows, cols, channels = frame_not_line_region.shape
        roi = frame_not_line_region[0:rows, 0:cols]

        # Take only line from line image.
        line = cv2.bitwise_and(line_image, line_image, mask=line_mask)
        # cv2.imshow('line_image', line_image)
        # cv2.imshow('line_mask', line_image)
        # cv2.imshow('mask_inv', mask_inv_line)
        # cv2.imshow('roi', roi)
        # cv2.imshow('line', line)
        # Add line to image
        output = cv2.add(line, roi)

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
