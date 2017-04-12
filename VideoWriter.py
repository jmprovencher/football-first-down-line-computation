import cv2


class VideoWriter:
    def __init__(self, name, frames):
        self._name = name + '.mp4'
        self._cap = cv2.VideoCapture(0)
        self._fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self._out = VideoWriter(self._name, self._fourcc, 30.0, (640, 480))
