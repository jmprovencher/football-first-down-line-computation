import cv2


class LinePicker:
    def __init__(self, frame):
        self._frame = frame
        self.refPt = list()
        self.first_down_point = list()
        self.scrimmage_point = list()
        cv2.imshow('Click on first down then on scrimmage', self._frame)
        cv2.setMouseCallback("Click on first down then on scrimmage", self.click_gather_point)
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.first_down_point = self.refPt[0]
        self.scrimmage_point = self.refPt[1]

    def click_gather_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(len(self.refPt), (x, y))
            point = (x, y)
            if len(self.refPt) == 0:
                self.refPt.append(point)
            elif len(self.refPt) == 1:
                self.refPt.append(point)
