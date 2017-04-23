import cv2
import numpy as np

refPT = list()


class ModelTransformer:

    def __init__(self, model, first_frame):
        self.refPT = list()
        self.last_frame = first_frame
        self.rows, self.cols, channels = first_frame.shape
        self.model = cv2.resize(model.image, (self.cols, self.rows))

        cv2.imshow('frame_hsv', self.line_mask(first_frame))
        cv2.imshow('frame', first_frame)
        cv2.setMouseCallback("frame", self.click_gather_point)
        cv2.waitKey()
        cv2.destroyAllWindows()
        frame_pts = self.refPT
        self.refPT = list()

        cv2.imshow('model', self.model)
        cv2.setMouseCallback("model", self.click_gather_point)
        cv2.waitKey()
        cv2.destroyAllWindows()
        model_pts = self.refPT

        self.H = cv2.getPerspectiveTransform(np.float32(frame_pts), np.float32(model_pts))

    def click_gather_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(len(self.refPT))
            point = (x, y)
            self.refPT.append(point)

    def new_frame(self, frame):
        M = self.get_homography_between_frames(frame)
        tmp_H = np.dot(M, self.H)
        lines = self.find_lines(frame)

        while not self.are_lines_vertical_in_model(lines, tmp_H):
            M = self.get_homography_between_frames(frame)
            tmp_H = np.dot(M, self.H)
            lines = self.find_lines(frame)

        self.H = tmp_H

        self.last_frame = frame

    def get_homography_between_frames(self, frame):
        # Initiate SIFT detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(self.last_frame, None)
        kp2, des2 = orb.detectAndCompute(frame, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:6]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:6]]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10000)
        return M

    def find_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(frame.shape, np.uint8)
        edges = cv2.Canny(gray, 300, 600, apertureSize=3)

        return cv2.HoughLines(edges, 1, np.pi / 180, 200, 0, 0)[:, 0, :]

    def are_lines_vertical_in_model(self, lines, H):
        return True

    def line_mask(self, frame):

        # grass [45, 44, 89] [55, 86, 174]
        # white [0, 0, 150] [165, 40, 255]
        min_hsv = np.asarray([45, 44, 89])
        max_hsv = np.asarray([55, 86, 174])
        dst = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        _grass_mask = cv2.inRange(hsv, min_hsv, max_hsv)
        grass_mask = np.zeros(frame.shape, np.uint8)
        for i in range(frame.shape[2]):
            grass_mask[:, :, i] = _grass_mask

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(frame.shape, np.uint8)
        edges = cv2.Canny(gray, 300, 600, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi/180, 200, 0, 0)
        print(lines.shape)
        for line in lines[:, 0, :]:
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 4)

        #mask = cv2.bitwise_or(grass_mask, mask)
        #return cv2.bitwise_and(mask, frame)
        return mask



