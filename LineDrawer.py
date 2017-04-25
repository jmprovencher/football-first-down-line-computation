import cv2
import numpy as np


class LineDrawer:
    def __init__(self, frame, modelTr=None, point=None, model=None):
        print('Drawing...')
        self._overlay = frame.copy()
        self._line_overlay = frame.copy()
        self._output = frame.copy()
        self._modelTr = modelTr
        self._point = point
        self._model = model

    def draw_line(self, image, pt1, pt2):
        height, width, depth = image.shape
        line_mask = np.zeros((height, width), np.uint8)
        line_image = np.zeros((height, width, 3), np.uint8)
        cv2.line(line_image, pt1, pt2, (43, 124, 220), 10)
        cv2.line(line_mask, pt1, pt2, (255, 255, 255), 5)


        alpha = 1  # Alpha defines the desired opacity of the line
        mask_inv = cv2.bitwise_not(line_mask)

        # Apply inversed mask of the line to get everything expect the line in image
        frame_not_line_region = cv2.bitwise_and(image, image, mask=mask_inv)

        rows, cols, channels = frame_not_line_region.shape
        roi = frame_not_line_region[0:rows, 0:cols]

        # Take only line from line image.
        line = cv2.bitwise_and(line_image, line_image, mask=line_mask)

        # Add line to image
        self._output = cv2.add(line, roi)

        #cv2.imshow('model_with_line', line_mask)
        #cv2.waitKey(0)
        #cv2.imshow('output', self._output)

        return self._output

    def applyHomographyToPoint(self):
        homogeneous_point = np.array([self._point], dtype=np.float32)
        #homogeneous_point = np.array([(self._point[1], self._point[0])], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(np.array([homogeneous_point]), self._modelTr.H)

        start_line = (transformed_point[0][0][0].astype(int), self._model.params['upperLineY'])
        end_line = (transformed_point[0][0][0].astype(int), self._model.params['lowerLineY'])

        tupletest = (transformed_point[0][0][0], transformed_point[0][0][1])
        cv2.circle(self._modelTr.model, tupletest, 3, (43, 124, 220))
        #cv2.imshow('first_down',self._modelTr.model)
        #cv2.waitKey()
        res, inv_H = cv2.invert(self._modelTr.H)
        model_line_points = np.array([start_line,end_line], dtype=np.float32)
        field_line_points = cv2.perspectiveTransform(np.array([model_line_points]), inv_H)
        start_line_field = (field_line_points[0][0][0],field_line_points[0][0][1])
        end_line_field = (field_line_points[0][1][0],field_line_points[0][1][1])
        return self.draw_line(self._overlay,start_line_field,end_line_field)


