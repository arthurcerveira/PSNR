import numpy as np


class VideoCaptureYUV(object):
    def __init__(self, filename, resolution):
        self.width, self.height = resolution
        self.file = open(filename, 'rb')

    def read_frame(self, bitdepth):
        uv_width = self.width // 2
        uv_height = self.height // 2

        Y = self.read_channel(self.height, self.width, bitdepth)
        U = self.read_channel(uv_height, uv_width, bitdepth)
        V = self.read_channel(uv_height, uv_width, bitdepth)

        return Y, U, V

    def read_channel(self, height, width, bitdepth):
        channel = np.zeros((height, width), np.uint8)

        for m in range(height):
            for n in range(width):
                if bitdepth == 8:
                    pel8bit = int.from_bytes(self.file.read(
                        1), byteorder='little', signed=False)
                    channel[m, n] = np.uint8(pel8bit)
                elif bitdepth == 10:
                    pel10bit_v = int.from_bytes(self.file.read(
                        2), byteorder='little', signed=False)
                    channel[m, n] = np.uint8((pel10bit_v + 2) / 4)

        return channel

    def close(self):
        self.file.close()


def psnr_channel(original, encoded, max_value):
    # Convert frames to double
    original = np.array(original, dtype=np.double)
    encoded = np.array(encoded, dtype=np.double)

    # Calculate mean squared error
    mse = np.mean((original - encoded) ** 2)

    # PSNR in dB
    psnr = 10 * np.log10((max_value * max_value) / (mse))
    return psnr, mse


def calculate_psnr(original, encoded, resolution, frames, max_value=255):
    original_video = VideoCaptureYUV(original, resolution)
    encoded_video = VideoCaptureYUV(encoded, resolution)

    psnr_y_array = list()
    psnr_u_array = list()
    psnr_v_array = list()
    mse_array = list()

    for frame in range(frames):
        original_y, original_u, original_v = original_video.read_frame(bitdepth=8)
        encoded_y, encoded_u, encoded_v = encoded_video.read_frame(bitdepth=8)

        psnr_y, mse_y = psnr_channel(original_y, encoded_y, max_value)
        psnr_y_array.append(psnr_y)

        psnr_u, mse_u = psnr_channel(original_u, encoded_u, max_value)
        psnr_u_array.append(psnr_u)

        psnr_v, mse_v = psnr_channel(original_v, encoded_v, max_value)
        psnr_v_array.append(psnr_v)

        mse = (4 * mse_y + mse_u + mse_v) / 6  # Weighted MSE
        mse_array.append(mse)

    # Close YUV streams
    original_video.close()
    encoded_video.close()

    # Average PSNR between all frames
    psnr_y = np.average(psnr_y_array)
    psnr_u = np.average(psnr_u_array)
    psnr_v = np.average(psnr_v_array)

    # Calculate YUV-PSNR based on average MSE
    mse_yuv = np.average(mse_array)
    psnr_yuv = 10 * np.log10((max_value * max_value) / (mse_yuv))

    return psnr_y, psnr_u, psnr_v, psnr_yuv


if __name__ == "__main__":
    original_video = "BQSquare_416x240_60.yuv"
    encoded_video = "BQSquare-Encoded-LD-8bits.yuv"
    resolution = (416, 240)

    psnr_y, pnsr_u, psnr_v, psnr_yuv = calculate_psnr(
        original_video, encoded_video, resolution, 5)

    print(f'Y-PSNR: {psnr_y:.4f}dB')
    print(f'U-PSNR: {pnsr_u:.4f}dB')
    print(f'V-PSNR: {psnr_v:.4f}dB')
    print(f'YUV-PSNR: {psnr_yuv:.4f}dB')
