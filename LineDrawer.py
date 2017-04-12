import cv2
import numpy as np


class LineDrawer:
    def __init__(self, frame, mask=None):
        print('Drawing...')
        self._overlay = frame.copy()
        self._line_overlay = frame.copy()
        self._output = frame.copy()

    def draw_line(self):
        height, width, depth = self._overlay.shape
        line_mask = np.zeros((height, width), np.uint8)
        line_image = np.zeros((height, width,3), np.uint8)
        pt1 = (82, 105)
        pt2 = (279, 480)
        cv2.line(line_image, pt1, pt2, (43, 124, 220), 10)
        cv2.line(line_mask, pt1, pt2, (255, 255, 255), 10)


        alpha = 1  # Alpha defines the desired opacity of the line
        mask_inv = cv2.bitwise_not(line_mask)

        # Apply inversed mask of the line to get everything expect the line in image
        frame_not_line_region = cv2.bitwise_and(self._overlay, self._overlay, mask=mask_inv)

        rows, cols, channels = frame_not_line_region.shape
        roi = frame_not_line_region[0:rows, 0:cols]

        # Take only line from line image.
        line = cv2.bitwise_and(line_image, line_image, mask=line_mask)

        # Add line to image
        self._output = cv2.add(line, roi)


        return self._output
