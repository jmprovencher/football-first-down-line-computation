import cv2
import numpy as np
from VideoPlayer import VideoPlayer


# optional argument
def nothing(x):
    pass


vp = VideoPlayer('resources/video/field1/WideWide - Clip 001.mp4')
frames = vp.extract_frames()
cv2.namedWindow('image')
image = frames[40]



# easy assigments
hh = 'Hue High'
hl = 'Hue Low'
sh = 'Saturation High'
sl = 'Saturation Low'
vh = 'Value High'
vl = 'Value Low'


cv2.createTrackbar(hl, 'image', 0, 179, nothing)
cv2.createTrackbar(hh, 'image', 0, 179, nothing)
cv2.createTrackbar(sl, 'image', 0, 255, nothing)
cv2.createTrackbar(sh, 'image', 0, 255, nothing)
cv2.createTrackbar(vl, 'image', 0, 255, nothing)
cv2.createTrackbar(vh, 'image', 0, 255, nothing)

while(1):
    image=cv2.GaussianBlur(image,(5,5),0)

    hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('image', res)


