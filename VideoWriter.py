import cv2


class VideoWriter:
    def __init__(self, name, frames):
        self._name = name + '.mp4'
        self.frames = frames
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._out = cv2.VideoWriter(self._name, self._fourcc, 30.0, (self.frames[0].shape[1], self.frames[0].shape[0]),
                                    True)
        print('Saving to video file...')
        for frame in frames:
            self._out.write(frame)
        self._out.release()
        print(self._name, ' processed')
