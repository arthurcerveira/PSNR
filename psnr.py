import cv2
import numpy as np


class VideoCaptureYUV(object):
    def __init__(self, filename, resolution):
        self.width, self.height = resolution
        self.frame_len = self.width * self.height * 3 // 2
        self.file = open(filename, 'rb')
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        raw = self.file.read(self.frame_len)

        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape(self.shape)

        return yuv

    def read_frame(self):
        yuv = self.read_raw()
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420, 3)

        return frame

    def close(self):
        self.file.close()


def calculate_psnr(original, encoded, resolution, frames, max_value=255):
    original_video = VideoCaptureYUV(original, resolution)
    encoded_video = VideoCaptureYUV(encoded, resolution)
    psnr_array = list()

    for frame in range(frames):
        original_frame = original_video.read_frame()
        encoded_frame = encoded_video.read_frame()

        # Convert frames to float
        original_frame = np.array(original_frame, dtype=np.float32)
        encoded_frame = np.array(encoded_frame, dtype=np.float32)

        # Calculate mean squared error
        mse = np.mean((original_frame - encoded_frame) ** 2)

        # PSNR in dB
        psnr = 20 * np.log10(max_value / (np.sqrt(mse)))

        psnr_array.append(psnr)

    # Close YUV streams
    original_video.close()
    encoded_video.close()

    average_psnr = np.average(psnr_array)

    return average_psnr
