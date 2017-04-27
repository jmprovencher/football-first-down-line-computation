import cv2
import numpy as np
from random import expovariate

refPT = list()


class ModelTransformer:
    def __init__(self, model, first_frame, with_points=False):
        self.H = np.identity(3, np.float32)

        self.last_frame = first_frame
        self.rows, self.cols, channels = first_frame.shape
        self.model = cv2.resize(model.image, (self.cols, self.rows))

        frame_pts = list()
        model_pts = list()
        if with_points:
            self.refPT = list()
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

        else:
            frame_pts.append((85, 125))
            frame_pts.append((353, 107))
            frame_pts.append((167, 258))
            frame_pts.append((681, 197))

            model_pts.append((503, 49))
            model_pts.append((605, 49))
            model_pts.append((505, 289))
            model_pts.append((606, 289))

        self.H = cv2.getPerspectiveTransform(np.float32(frame_pts), np.float32(model_pts))
        self.last_H = self.H
        self.last_good_H = self.H
        self.idx = 0

    def click_gather_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(len(self.refPT), (x, y))
            point = (x, y)
            self.refPT.append(point)

    def new_frame(self, frame):
        self.idx += 1
        warped_frame = cv2.warpPerspective(frame, self.H, (self.cols, self.rows))
        warped_mask = cv2.warpPerspective(np.ones(frame.shape, np.uint8)*255, self.H, (self.cols, self.rows))

        #lines = self.find_global_lines(frame)
        counter = 0
        max_percent_good_lines = 200000
        best_transform = [max_percent_good_lines, self.last_good_H]
        while 1:#best_transform[0] >= max_percent_good_lines:
            #M = self.get_homography_between_frames(frame, self.last_frame)
            M = self.get_homography_between_lines(frame, self.last_frame)
            if M is None or M.shape != (3, 3):
                break
            #M = np.linalg.inv(M)
            tmp_H = np.dot(self.last_H, M)
            #tmp_H = np.dot(self.H, M)
            percent_good_lines = np.power(tmp_H - self.H, 2).sum() / 9
            #percent_good_lines = self.are_lines_vertical_or_horizontal_in_model(lines, tmp_H)

            # if percent_good_lines > best_transform[0]:
            if percent_good_lines < best_transform[0]:
                best_transform[0] = percent_good_lines
                best_transform[1] = tmp_H

            if counter >= 0:
                break
            counter += 1

        print(best_transform[0])
        if best_transform[0] < max_percent_good_lines:
            self.last_good_H = best_transform[1]

        self.H = best_transform[1]
        #self.last_frame = frame

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

            #line_mask1, _ = self.line_mask(img1)
            #line_mask2, _ = self.line_mask(img2)

            #mask1 = cv2.bitwise_or(mask1, line_mask1[:, :, 0])
            #mask2 = cv2.bitwise_or(mask2, line_mask2[:, :, 0])

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, None)#mask1)
        kp2, des2 = orb.detectAndCompute(img2, None)#mask2)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        matches_int = np.asarray([(m.queryIdx, m.trainIdx) for m in matches], np.int)

        random_draw = list()
        for i in range(8):
            value = int(expovariate(1/4))
            while value in random_draw:
                value = int(expovariate(1/4))
            random_draw.append(value)
        random_draw = np.asarray(random_draw, np.int)

        src_pts = np.float32([kp1[m[0]].pt for m in matches_int[random_draw]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[1]].pt for m in matches_int[random_draw]]).reshape(-1, 1, 2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:4], None,  matchColor=(0, 255, 0), flags=0)
        cv2.imshow('matches', img3)
        #cv2.waitKey()

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 40)
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blured_frame = cv2.blur(gray, (7, 7))
        edges = cv2.Canny(blured_frame, 20, 50, apertureSize=3)
        edges = cv2.blur(edges, (3, 3))
        edges = cv2.bitwise_and(mask, edges)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 400, 50, 125)[:, 0, :]
        for x1, y1, x2, y2 in lines:
            list_of_line_points.append(((x1, y1, 1), (x2, y2, 1)))

        list_of_line_points = np.float32(list_of_line_points)

        return list_of_line_points

    def are_lines_vertical_or_horizontal_in_model(self, list_of_line_points, H):
        count_good_lines = 0

        for p1, p2 in list_of_line_points:
            p1_prime = np.dot(H, (p1[0], p1[1], 1))
            p2_prime = np.dot(H, (p2[0], p2[1], 1))

            p1_prime /= p1_prime[-1]
            p2_prime /= p2_prime[-1]

            x_pos = [69, 171, 197, 223, 248, 274, 298, 325, 349, 375, 401, 426, 478, 504, 555, 579,
                     606, 630, 656, 683, 707, 733, 834]
            y_pos = [49, 432]

            if p2_prime[0] - 10 <= p1_prime[0] <= p2_prime[0] + 10:
                mid_x = p1_prime[0] + p2_prime[0]
                mid_x /= 2
                for x in x_pos:
                    if x - 10 <= mid_x <= x + 10:
                        count_good_lines += 1
                        break

            if p2_prime[1] - 10 <= p1_prime[1] <= p2_prime[1] + 10:
                mid_y = p1_prime[1] + p2_prime[1]
                mid_y /= 2
                for y in y_pos:
                    if y - 10 <= mid_y <= y + 10:
                        count_good_lines += 1
                        break

        return count_good_lines / len(list_of_line_points)

    def find_global_lines(self, frame):
        pts = self.find_lines(frame)
        clusters = []
        cluster_values = []

        new_lines = []

        for i, (pt1, pt2) in enumerate(pts):

            if pt1[0] > pt2[0]:
                pt_tmp = pt1
                pt2 = pt1
                pt1 = pt2

                pts[i] = (pt1, pt2)

            pente = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            if np.isnan(pente):
                pente = frame.shape[0]
            elif np.isinf(pente):
                pente = 0

            y_intercept = pt1[1] - pente * pt1[0]

            x_intercept = (frame.shape[0] / 2 - y_intercept) / pente
            if np.isnan(x_intercept) or np.isinf(x_intercept):
                x_intercept = y_intercept

            x1 = 0
            y1 = pente * x1 + y_intercept
            x2 = frame.shape[1]
            y2 = pente * x2 + y_intercept

            pt1 = np.asarray([x1, y1], np.int)
            pt2 = np.asarray([x2, y2], np.int)
            rectangle = (0, 0, frame.shape[1], frame.shape[0])
            _, pt1, pt2 = cv2.clipLine(rectangle, (pt1[0], pt1[1]), (pt2[0], pt2[1]))

            new_lines.append([pt1, pt2])

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

        new_lines = np.asarray(new_lines, np.int)

        global_lines = []
        for index, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue

            pt1, pt2 = (new_lines[cluster].sum(axis=0) / len(cluster)).astype(int)
            rectangle = (0, 0, frame.shape[1], frame.shape[0])
            clipped = cv2.clipLine(rectangle, (pt1[0], pt1[1]), (pt2[0], pt2[1]))
            _, pt1, pt2 = clipped
            global_lines.append([np.asarray(pt1), np.asarray(pt2)])

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

                    max_1 = np.max([o1, o2], axis=0)
                    min_1 = np.min([o1, o2], axis=0)
                    max_2 = np.max([p1, p2], axis=0)
                    min_2 = np.min([p1, p2], axis=0)

                    max = np.max([max_1, min_1, max_2, min_2], axis=0)
                    min = np.min([max_1, min_1, max_2, min_2], axis=0)

                    if min[0] <= r[0] <= max[0] and min[1] <= r[1] <= max[1]:
                        pente_1 = (p1[1] - o1[1]) / (p1[0] - o1[1])
                        pente_2 = (p2[1] - o2[1]) / (p2[0] - o2[1])
                        if np.abs(pente_1) >= 0.1:
                            if pente_1 >= 0:
                                global_lines[i][0] = r
                            else:
                                global_lines[i][1] = r
                        if np.abs(pente_2) >= 0.1:
                            if pente_2 >=0:
                                global_lines[j][0] = r
                            else:
                                global_lines[j][1] = r

        return np.asarray(global_lines)

    def line_mask(self, mask, global_lines):
        for line in global_lines:
            (o1, p1), (o2, p2) = line
            cv2.line(mask, (o1, p1), (o2, p2), (255, 255, 255), 3)

        return mask

    def match_lines(self, lines_1, lines_2):
        matches_dst = np.zeros((len(lines_1), len(lines_2)), np.float32)

        for i, line_i in enumerate(lines_1):
            for j, line_j in enumerate(lines_2):
                matches_dst[i, j] = np.power(line_i - line_j, 2).sum() / 4

        matches = list()
        for i in range(len(lines_1)):
            best_match = None
            for j in range(len(lines_2)):
                if best_match is None:
                    best_match = [j, matches_dst[i, j]]
                elif best_match[1] > matches_dst[i, j]:
                    best_match[0] = j
                    best_match[1] = matches_dst[i, j]

            matches.append([[i, best_match[0]], best_match[1]])

        return matches

    def get_homography_between_lines(self, img1, img2):
        mask = np.zeros(img1.shape, np.uint8)
        global_lines_1 = self.find_global_lines(img1)
        global_lines_2 = self.find_global_lines(img2)

        matches = self.match_lines(global_lines_1, global_lines_2)

        matches = sorted(matches, key=lambda x: x[1])

        print(matches)

        src_pts = list()
        dst_pts = list()

        colors1 = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
        colors2 = ((128, 0, 0), (0, 128, 0), (0, 0, 128))
        for i in range(min(3, len(matches) - 1)):
            index_1, index_2 = matches[i][0]

            pt11, pt12 = global_lines_1[index_1]
            pt21, pt22 = global_lines_2[index_2]

            src_pts.append(pt11)
            src_pts.append(pt12)
            dst_pts.append(pt21)
            dst_pts.append(pt22)

            cv2.line(mask, (pt11[0], pt11[1]), (pt12[0], pt12[1]), colors1[i], 2, cv2.LINE_AA)
            cv2.line(mask, (pt21[0], pt21[1]), (pt22[0], pt22[1]), colors2[i], 2, cv2.LINE_AA)

        cv2.imshow('matching lines', mask)
        cv2.waitKey(1)

        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)

        return M




