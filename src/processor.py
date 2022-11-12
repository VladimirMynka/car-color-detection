import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from constants import KERNELS


def get_all(image):
    return np.size(image) // 3

def conv(img, kernel):
    return cv2.filter2D(np.array(img), -1, kernel)

def log_images(images, titles):
    assert len(images) == len(titles)
    fig = plt.figure(figsize=(25*len(images), 25))
    for i in range(len(images)):
        ax = fig.add_subplot(len(images), 1, i + 1)
        plt.imshow(images[i])
        ax.set_title(titles[i])
    plt.show()

class Processor:
    def __init__(
        self, 
        bounds_kernels=KERNELS, 
        filter1=80, 
        when_black=100, 
        when_white=700, 
        when_grey_std=40,
    ):
        self.bounds_kernels = bounds_kernels
        self.filter1 = filter1
        self.when_black = when_black
        self.when_white = when_white
        self.when_grey_std = when_grey_std

    def get_bounds(self, image: Image) -> Image:
        kerneled_imges = [conv(image, np.array(kernel)) for kernel in self.bounds_kernels]
        result = kerneled_imges[0]
        for img_k in kerneled_imges[1:]:
            result += img_k
        result = result // len(KERNELS)
        return Image.fromarray(result)
        
    def process_image(self, image: Image, logging=False) -> Image:
        img = np.array(image)
        step1 = self.get_bounds(image)

        step10 = np.array(step1)
        step10[np.sum(step10, axis=2) > self.filter1] = 255
        step10[step10 != 255] = 0
        step10 //= 255

        step20 = img * step10

        step30 = conv(step20, np.array([1 * 3] * 3) / 9)
        mask3 = np.std(step30, axis=2) < self.when_grey_std
        step30[mask3] = 0

        if logging:
            log_images([img, step10 * 255, step20, step30], ['input', 'step10 = get_bounds(input)', 'step20 = input * step10', 'result = drop_grey(step20)'])
        return Image.fromarray(step30)

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