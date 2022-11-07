import numpy as np
import cv2
from PIL import Image


KERNELS=[
    [
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ],
    [
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3]
    ]
    # [
    #     [1, 1, 1],
    #     [1, -9, 1],
    #     [1, 1, 1]
    # ],
    # [
    #     [1, 1, 1],
    #     [1, 0, 0],
    #     [1, 0, -5]
    # ],
    # [
    #     [1, 1, 1],
    #     [0, 0, 1],
    #     [-5, 0, 1]
    # ],
    # [
    #     [1, 0, -5],
    #     [1, 0, 0],
    #     [1, 1, 1]
    # ]
]


def conv(img, kernel):
    return cv2.filter2D(np.array(img), -1, kernel)

def get_bounds(img):
    kerneled_imges = [conv(img, np.array(kernel)) for kernel in KERNELS]
    result = kerneled_imges[0]
    for img_k in kerneled_imges[1:]:
        result += img_k
    result = result // len(KERNELS)
    return Image.fromarray(result)



def get_only_not_black(image):
    return (np.sum(image, axis=2) != 0).sum()

def get_avg_colors(images, len_searcher=get_only_not_black):
    r,g,b=0,0,0
    for image in images:
        arr = np.array(image)
        r1 = arr[:, :, 0].sum()
        g1 = arr[:, :, 1].sum()
        b1 = arr[:, :, 2].sum()
        l = len_searcher(arr)
        if l == 0:
            l = 1
        r += r1 / l; g += g1 / l; b += b1 / l
    return np.array([r,g,b])/len(images)

def process_image(image):
    img = np.array(image)
    step1 = get_bounds(image)

    step10 = np.array(step1)
    step10[np.sum(step10, axis=2) > 80] = 1
    step10[step10 != 1] = 0

    step20 = img * step10

    # step30 = cv2.fastNlMeansDenoisingColored(step20, None, 10, 10, 7, 25)
    step30 = conv(step20, np.array([1 * 3] * 3) / 9)
    # step30 = step20
    mask1 = np.sum(step30, axis=2) < 100
    mask2 = np.sum(step30, axis=2) > 700
    mask3 = np.std(step30, axis=2) < 40
    step30[mask1] = 0
    step30[mask2] = 0
    step30[mask3] = 0
    step30[~(mask1 | mask2 | mask3)] = 1

    step40 = img * step30
    return Image.fromarray(step40)

def process_images(images):
    return [process_image(image) for image in images]


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_std_colors(images, mean, len_searcher=get_only_not_black):
    rgb = np.array([0, 0, 0]).astype(float)
    for image in images:
        arr = np.array(image)
        r1 = arr[:, :, 0].sum()
        g1 = arr[:, :, 1].sum()
        b1 = arr[:, :, 2].sum()
        l = len_searcher(arr)
        if l == 0:
            l = 1
        cur_mean = np.array([r1, g1, b1]).astype(float)
        cur_mean /= l
        rgb += (cur_mean - mean) ** 2
    return np.sqrt(rgb / len(images)) 