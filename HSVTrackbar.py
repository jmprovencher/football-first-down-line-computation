import cv2
import numpy as np
from VideoPlayer import VideoPlayer


class HSVPicker():
    def __init__(self, frame):
        self._image = frame
        cv2.namedWindow('image')


    def nothing(self, x):
        pass

    def getHSVMask(self):
        mask=None
        # easy assigments
        hh = 'Hue High'
        hl = 'Hue Low'
        sh = 'Saturation High'
        sl = 'Saturation Low'
        vh = 'Value High'
        vl = 'Value Low'

        cv2.createTrackbar(hl, 'image', 43, 179, self.nothing)
        cv2.createTrackbar(hh, 'image', 57, 179, self.nothing)
        cv2.createTrackbar(sl, 'image', 46, 255, self.nothing)
        cv2.createTrackbar(sh, 'image', 86, 255, self.nothing)
        cv2.createTrackbar(vl, 'image', 81, 255, self.nothing)
        cv2.createTrackbar(vh, 'image', 178, 255, self.nothing)

        while (1):
            # image=cv2.GaussianBlur(self._image(5,5),0)

            hsv = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            hul = cv2.getTrackbarPos(hl, 'image')
            huh = cv2.getTrackbarPos(hh, 'image')
            sal = cv2.getTrackbarPos(sl, 'image')
            sah = cv2.getTrackbarPos(sh, 'image')
            val = cv2.getTrackbarPos(vl, 'image')
            vah = cv2.getTrackbarPos(vh, 'image')
            HSVLOW = np.array([hul, sal, val])
            HSVHIGH = np.array([huh, sah, vah])

            # apply the range on a mask
            mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
            res = cv2.bitwise_and(self._image, self._image, mask=mask)
            cv2.imshow('image', res)
        return mask
