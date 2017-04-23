
import cv2
import numpy as np
from VideoPlayer import VideoPlayer

colors = []


def on_mouse_click (event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONUP:
        colors.append(frame[y,x].tolist())


def main():
    vp = VideoPlayer('resources/video/field1/WideWide - Clip 001.mp4')
    frames = vp.extract_frames()

    #for frame in frames:
    hsv = cv2.cvtColor(frames[13], cv2.COLOR_BGR2HSV)
    if colors:
        cv2.putText(hsv, str(colors[-1]), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.imshow('frame', hsv)
    cv2.setMouseCallback('frame', on_mouse_click, hsv)

    cv2.waitKey()

    cv2.destroyAllWindows()

    # avgb = int(sum(c[0] for c in colors) / len(colors))
    # avgg = int(sum(c[0] for c in colors) / len(colors))
    # avgr = int(sum(c[0] for c in colors) / len(colors))
    # print avgb, avgg, avgr

    minb = min(c[0] for c in colors)
    ming = min(c[1] for c in colors)
    minr = min(c[2] for c in colors)
    maxb = max(c[0] for c in colors)
    maxg = max(c[1] for c in colors)
    maxr = max(c[2] for c in colors)
    print(minr, ming, minb, maxr, maxg, maxb)

    lb = [minb,ming,minr]
    ub = [maxb,maxg,maxr]
    print(lb, ub)

if __name__ == "__main__":
    main()