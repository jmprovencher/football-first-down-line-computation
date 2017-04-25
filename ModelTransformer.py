import cv2
import numpy as np
from random import expovariate

refPT = list()


class ModelTransformer:
    def __init__(self, model, first_frame):
        self.H = np.identity(3, np.float32)

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
            print(len(self.refPT), (x, y))
            point = (x, y)
            self.refPT.append(point)

    def new_frame(self, frame):
        warped_frame = cv2.warpPerspective(frame, self.H, (self.cols, self.rows))
        warped_mask = cv2.warpPerspective(np.ones(frame.shape, np.uint8)*255, self.H, (self.cols, self.rows))
        #M = self.get_homography_between_frames(warped_frame,
        #                                       cv2.bitwise_and(self.model, warped_mask),
        #                                       warped_mask[:, :, 0])
        M = self.get_homography_between_frames(self.last_frame, frame)
        tmp_H = np.dot(M, self.H)
        lines = self.find_lines(frame)

        counter = 0
        percent_good_lines = self.are_lines_vertical_or_horizontal_in_model(lines, tmp_H)
        best_transform = [percent_good_lines, tmp_H]
        """
        while percent_good_lines < 0.9:
            if percent_good_lines > best_transform[0]:
                best_transform[0] = percent_good_lines
                best_transform[1] = tmp_H

                # M = self.get_homography_between_frames(warped_frame,
                #                                       cv2.bitwise_and(self.model, warped_mask),
                #                                       warped_mask[:, :, 0])
                M = self.get_homography_between_frames(self.last_frame, frame)
            if M is None or M.shape != (3, 3):
                continue

            tmp_H = np.dot(M, self.H)
            lines = self.find_lines(frame)
            percent_good_lines = self.are_lines_vertical_or_horizontal_in_model(lines, tmp_H)
            if counter >= 20:
                break
            counter += 1
        """

        self.H = best_transform[1]

        self.last_frame = frame

    def get_homography_between_frames(self, img1, img2, mask=None):
        cv2.ocl.setUseOpenCL(False)

        # Initiate SIFT detector
        kernel_size = 9
        orb = cv2.ORB_create(1000, 1.2, 8, kernel_size, 0, 4, cv2.ORB_HARRIS_SCORE, kernel_size, 20)

        mask1 = mask
        mask2 = mask

        if mask is None:
            mask1 = 255 - self.get_terrain_mask(img1)
            mask2 = 255 - self.get_terrain_mask(img2)

            mask1 = cv2.bitwise_or(mask1, self.line_mask(img1)[:, :, 0])
            mask2 = cv2.bitwise_or(mask2, self.line_mask(img2)[:, :, 0])

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, mask1)
        kp2, des2 = orb.detectAndCompute(img2, mask2)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        matches_int = np.asarray([(m.queryIdx, m.trainIdx) for m in matches], np.int)

        random_draw = list()
        for i in range(20):
            value = int(expovariate(1/3))
            while value in random_draw:
                value = int(expovariate(1/3))
            random_draw.append(value)
        random_draw = np.asarray(random_draw, np.int)

        src_pts = np.float32([kp1[m[0]].pt for m in matches_int[:20]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[1]].pt for m in matches_int[:20]]).reshape(-1, 1, 2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,  matchColor=(0, 255, 0), flags=0)
        cv2.imshow('matches', img3)
        cv2.waitKey()

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10000)
        #M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return M

    def get_terrain_mask(self, frame):
        # grass [45, 44, 89] [55, 86, 174]
        # white [0, 0, 150] [165, 40, 255]
        min_hsv = np.asarray([45, 44, 89])
        max_hsv = np.asarray([55, 86, 174])
        dst = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, min_hsv, max_hsv)

        mask = cv2.erode(mask, np.ones((3, 3)))
        mask = cv2.dilate(mask, np.ones((31, 31)))
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

        return mask

    def find_lines(self, frame):
        list_of_line_points = list()

        mask = self.get_terrain_mask(frame)

        #cv2.imshow('grass', mask)
        #cv2.waitKey()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blured_frame = cv2.blur(gray, (7, 7))
        edges = cv2.Canny(blured_frame, 20, 50, apertureSize=3)
        #edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                       cv2.THRESH_BINARY_INV, 3, 2)
        edges = cv2.blur(edges, (3, 3))
        edges = cv2.bitwise_and(mask, edges)

        #cv2.imshow('frame', blured_frame)
        #cv2.imshow('edges', edges)
        #cv2.waitKey()

        """
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80, 30, 10)[:, 0, :]
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            list_of_line_points.append(((x1, y1, 1), (x2, y2, 1)))
        """

        #"""
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 400, 50, 125)[:, 0, :]
        for x1, y1, x2, y2 in lines:
            list_of_line_points.append(((x1, y1, 1), (x2, y2, 1)))
        #"""
        list_of_line_points = np.float32(list_of_line_points)

        return list_of_line_points

    def are_lines_vertical_or_horizontal_in_model(self, list_of_line_points, H):
        count_good_lines = 0

        for p1, p2 in list_of_line_points:
            p1_prime = np.dot(H, p1.T)
            p2_prime = np.dot(H, p2.T)

            p1_prime /= p1_prime[-1]
            p2_prime /= p2_prime[-1]

            if p2_prime[0] - 3 <= p1_prime[0] <= p2_prime[0] + 3 or \
                p2_prime[1] - 3 <= p1_prime[1] <= p2_prime[1] + 3:
                count_good_lines += 1

        return count_good_lines / len(list_of_line_points)

    def line_mask(self, frame):
        pts = self.find_lines(frame)

        mask = np.zeros(frame.shape, np.uint8)

        clusters = []
        cluster_values = []

        new_lines = []

        for i, (pt1, pt2) in enumerate(pts):
            pente = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            if np.isnan(pente):
                pente = frame.shape[0]
            elif np.isinf(pente):
                pente = 0

            y_intercept = pt1[1] - pente * pt1[0]

            x_intercept = (mask.shape[0]/2-y_intercept) / pente
            if np.isnan(x_intercept) or np.isinf(x_intercept):
                x_intercept = y_intercept

            x1 = 0
            y1 = pente*x1 + y_intercept
            x2 = frame.shape[1]
            y2 = pente*x2 + y_intercept

            pt1 = np.asarray([x1, y1], np.int)
            pt2 = np.asarray([x2, y2], np.int)
            rectangle = (0, 0, mask.shape[1], mask.shape[0])
            _, pt1, pt2 = cv2.clipLine(rectangle, (pt1[0], pt1[1]), (pt2[0], pt2[1]))

            """
            pt1_prime = np.dot(self.H, np.asarray(pt1 + (1,), np.float32))
            pt1_prime /= pt1_prime[-1]
            pt2_prime = np.dot(self.H, np.asarray(pt2 + (1,), np.float32))
            pt2_prime /= pt2_prime[-1]

            pt1_prime = pt1_prime.astype(np.int)
            pt2_prime = pt2_prime.astype(np.int)

            pente = (pt2_prime[1] - pt1_prime[1]) / (pt2_prime[0] - pt1_prime[0])
            if np.isnan(pente):
                pente = frame.shape[0]
            elif np.isinf(pente):
                pente = 0

            y_intercept = pt1_prime[1] - pente * pt1_prime[0]

            x_intercept = (-y_intercept) / pente
            if np.isnan(x_intercept) or np.isinf(x_intercept):
                x_intercept = y_intercept

            x1 = 0
            y1 = pente * x1 + y_intercept
            x2 = frame.shape[1]
            y2 = pente * x2 + y_intercept

            new_lines.append([[x1, y1, 1], [x2, y2, 1]])
            """
            new_lines.append([pt1, pt2])

            print(pente, x_intercept, y_intercept)

            found = False
            for index, values in enumerate(cluster_values):
                if np.abs(pente) <= 0.1:
                    if y_intercept - 10 <= values['y_intercept'] <= y_intercept + 10:
                        clusters[index].append(i)
                        found = True
                        break
                else:
                    if x_intercept - 20 <= values['x_intercept'] <= x_intercept + 20:
                        clusters[index].append(i)
                        found = True
                        break

            if not found:
                cluster_values.append({'pente': pente, 'x_intercept': x_intercept, 'y_intercept': y_intercept})
                clusters.append([i])

        print(clusters)

        new_lines = np.asarray(new_lines, np.int)

        global_lines = []
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (200, 200, 0), (200, 0, 200), (0, 200, 200), (200, 200, 200), (255, 255, 255)]
        for index, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue

            """
            for pt in cluster:
                pt1, pt2 = new_lines[pt]
                rectangle = (0, 0, mask.shape[1], mask.shape[0])
                clipped = cv2.clipLine(rectangle, (pt1[0], pt1[1]), (pt2[0], pt2[1]))
                _, pt1, pt2 = clipped
                cv2.line(mask, (pt1[0], pt1[1]), (pt2[0], pt2[1]), colors[index % len(colors)], 3)
            """

            #"""
            pt1, pt2 = (new_lines[cluster].sum(axis=0) / len(cluster)).astype(int)
            rectangle = (0, 0, mask.shape[1], mask.shape[0])
            clipped = cv2.clipLine(rectangle, (pt1[0], pt1[1]), (pt2[0], pt2[1]))
            _, pt1, pt2 = clipped
            global_lines.append((np.asarray(pt1), np.asarray(pt2)))
            cv2.line(mask, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255, 255, 255), 3)
            #"""

        for i in range(len(global_lines)):
            for j in range(len(global_lines)):
                if i == j:
                    continue

                o1, p1 = global_lines[i]
                o2, p2 = global_lines[j]

                x = o2 - o1
                d1 = p1 - o1
                d2 = p2 - o2

                cross = (d1[0] * d2[1]) - (d1[1] * d2[0])

                if np.abs(cross) > 1e-8:
                    t1 = ((x[0] * d2[1]) - (x[1] * d2[0])) / cross
                    r = (o1 + d1 * t1).astype(int)

                    cv2.circle(mask, (r[0], r[1]), 10, (255, 255, 255), cv2.FILLED)

                else:
                    print("co-cross", cross)

        return mask
