import cv2
import numpy as np
import matplotlib.pyplot as plt


refPT = list()


class ModelTransformer:

    def __init__(self, model, first_frame):
        self.refPT = list()
        self.last_frame = first_frame
        self.rows, self.cols, channels = first_frame.shape
        self.model = cv2.resize(model.image, (self.cols, self.rows))

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
        self.refPT = list()

        print(frame_pts)
        print(model_pts)

        self.H = cv2.getPerspectiveTransform(np.float32(frame_pts), np.float32(model_pts))
        print(self.H)

        frame_transformed = cv2.warpPerspective(first_frame, self.H, (self.cols, self.rows))

        cv2.imshow('model', cv2.addWeighted(self.model, 1, frame_transformed, 1, 0))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def click_gather_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(len(self.refPT))
            point = (x, y)
            self.refPT.append(point)

    def new_frame(self, frame):
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

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

        #M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.H = np.dot(M, self.H)

        frame_transformed = cv2.warpPerspective(frame, self.H, (self.cols, self.rows))

        cv2.imshow('model', cv2.addWeighted(self.model, 1, frame_transformed, 1, 0))
        cv2.waitKey()
        cv2.destroyAllWindows()

        self.last_frame = frame


