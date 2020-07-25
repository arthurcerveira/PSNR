import cv2
import numpy as np


class VideoCaptureYUV(object):
    def __init__(self, filename, resolution):
        self.width, self.height = resolution
        self.frame_len = self.width * self.height * 3 // 2
        self.file = filename
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        with open(self.file, 'rb') as f:
            raw = f.read(self.frame_len)

        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape(self.shape)

        return yuv

    def read_frame(self):
        yuv = self.read_raw()
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420, 3)

        return frame
