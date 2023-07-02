import numpy as np
from PIL import Image
import os


def compute_mean_std(image_folder):
    pixel_nums = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)

    for filename in os.listdir(image_folder):
        img = Image.open(os.path.join(image_folder, filename))
        img = img.convert("RGB")
        img = np.array(img) / 255.0
        pixel_nums += img.shape[0] * img.shape[1]
        channel_sum += np.sum(img, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(img), axis=(0, 1))

    rgb_mean = channel_sum / pixel_nums
    rgb_std = np.sqrt(channel_sum_squared / pixel_nums - np.square(rgb_mean))

    return rgb_mean, rgb_std


image_folder = "./iclevr"
mean, std = compute_mean_std(image_folder)
print("Mean:", mean)
print("Std:", std)
