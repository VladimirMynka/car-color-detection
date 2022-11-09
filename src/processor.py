import cv2
import numpy as np
from PIL import Image

from constants import KERNELS


def get_all(image):
    return np.size(image) // 3

def conv(img, kernel):
    return cv2.filter2D(np.array(img), -1, kernel)

class Processor:
    def __init__(
        self, 
        bounds_kernels=KERNELS, 
        filter1=80, 
        when_black=100, 
        when_white=700, 
        when_grey_std=40
    ):
        self.bounds_kernels = bounds_kernels
        self.filter1 = filter1
        self.when_black = when_black
        self.when_white = when_white
        self.when_grey_std = when_grey_std

    def get_bounds(self, image):
        kerneled_imges = [conv(image, np.array(kernel)) for kernel in self.bounds_kernels]
        result = kerneled_imges[0]
        for img_k in kerneled_imges[1:]:
            result += img_k
        result = result // len(KERNELS)
        return Image.fromarray(result)
        
    def process_image(self, image):
        img = np.array(image)
        step1 = self.get_bounds(image)

        step10 = np.array(step1)
        step10[np.sum(step10, axis=2) > self.filter1] = 1
        step10[step10 != 1] = 0

        step20 = img * step10

        step30 = conv(step20, np.array([1 * 3] * 3) / 9)
        mask1 = np.sum(step30, axis=2) < self.when_black
        mask2 = np.sum(step30, axis=2) > self.when_white
        mask3 = np.std(step30, axis=2) < self.when_grey_std
        step30[mask1] = 0
        step30[mask2] = 0
        step30[mask3] = 0
        step30[~(mask1 | mask2 | mask3)] = 1

        step40 = img * step30
        return Image.fromarray(step40)

    def process_images(self, images):
        return [self.process_image(image) for image in images]

    def load(dictio):
        return Processor(**dictio)

    def __dict__(self):
        return {
            'bounds_kernels': self.bounds_kernels,
            'filter1': self.filter1,
            'when_black': self.when_black,
            'when_white': self.when_white,
            'when_grey_std': self.when_grey_std
        }