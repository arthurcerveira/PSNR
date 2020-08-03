import numpy as np


MAX_VALUE = 1020


class VideoCaptureYUV(object):
    def __init__(self, filename, resolution, bitdepth):
        self.file = open(filename, 'rb')
        self.width, self.height = resolution
        self.uv_width = self.width // 2
        self.uv_height = self.height // 2
        self.bitdepth = bitdepth

    def read_frame(self):
        Y = self.read_channel(self.height, self.width)
        U = self.read_channel(self.uv_height, self.uv_width)
        V = self.read_channel(self.uv_height, self.uv_width)

        return Y, U, V

    def read_channel(self, height, width):
        channel_len = height * width
        shape = (height, width)

        if self.bitdepth == 8:
            raw = self.file.read(channel_len)
            channel_8bits = np.frombuffer(raw, dtype=np.uint8)
            channel = np.array(channel_8bits, dtype=np.uint16) << 2  # Convert 8bits to 10 bits 

        elif self.bitdepth == 10:
            raw = self.file.read(2 * channel_len)  # Read 2 bytes for every pixel
            channel = np.frombuffer(raw, dtype=np.uint16)
        
        channel = channel.reshape(shape)

        return channel

    def close(self):
        self.file.close()


def psnr_channel(original, encoded):
    # Convert frames to double
    original = np.array(original, dtype=np.double)
    encoded = np.array(encoded, dtype=np.double)

    # Calculate mean squared error
    mse = np.mean((original - encoded) ** 2)

    # PSNR in dB
    psnr = 10 * np.log10((MAX_VALUE * MAX_VALUE) / mse)
    return psnr, mse


def calculate_psnr(original, encoded, resolution, frames, original_bitdepth, encoded_bitdepth):
    original_video = VideoCaptureYUV(original, resolution, original_bitdepth)
    encoded_video = VideoCaptureYUV(encoded, resolution, encoded_bitdepth)

    psnr_y_array = list()
    psnr_u_array = list()
    psnr_v_array = list()
    mse_array = list()

    for frame in range(frames):
        original_y, original_u, original_v = original_video.read_frame()
        encoded_y, encoded_u, encoded_v = encoded_video.read_frame()

        psnr_y, mse_y = psnr_channel(original_y, encoded_y)
        psnr_y_array.append(psnr_y)

        psnr_u, mse_u = psnr_channel(original_u, encoded_u)
        psnr_u_array.append(psnr_u)

        psnr_v, mse_v = psnr_channel(original_v, encoded_v)
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
    psnr_yuv = 10 * np.log10((MAX_VALUE * MAX_VALUE) / mse_yuv)

    return psnr_y, psnr_u, psnr_v, psnr_yuv


if __name__ == "__main__":
    original_video = "RaceHorses_416x240_30.yuv"
    encoded_video = "RaceHorses-Decoded-LD-10bits.yuv"
    resolution = (416, 240)
    frames = 5
    original_bitdepth = 8
    encoded_bitdepth = 10

    psnr_y, pnsr_u, psnr_v, psnr_yuv = calculate_psnr(
        original_video, encoded_video, resolution, 
        frames, original_bitdepth, encoded_bitdepth
    )

    print(f'Y-PSNR: {psnr_y:.4f}dB')
    print(f'U-PSNR: {pnsr_u:.4f}dB')
    print(f'V-PSNR: {psnr_v:.4f}dB')
    print(f'YUV-PSNR: {psnr_yuv:.4f}dB')
