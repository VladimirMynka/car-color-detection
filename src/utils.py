import numpy as np

from constants import KERNELS


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def euclide(x, y):
    return 1 / np.sum((x - y) ** 2)


def image_to_colors(ndarr):
    return ndarr.transpose([2, 0, 1]).reshape(3, -1)


def get_only_not_black(image):
    return (np.sum(image, axis=2) != 0).sum()


def get_avg_colors(images, len_searcher=get_only_not_black):
    rgb = np.array([0, 0, 0]).astype(float)
    for image in images:
        rgb += get_avg_colors_one(image, len_searcher)
    return rgb / len(images)


def get_std_colors(images, mean, len_searcher=get_only_not_black):
    rgb = np.array([0, 0, 0]).astype(float)
    for image in images:
        cur_mean = get_avg_colors_one(image, len_searcher)
        rgb += (cur_mean - mean) ** 2
    return np.sqrt(rgb / len(images)) 


def get_avg_colors_one(image, len_searcher=get_only_not_black):
    ndarr = np.array(image)
    rgb = image_to_colors(ndarr).sum(axis=1)
    l = len_searcher(ndarr)
    if l == 0:
        l = 1
    return rgb / l

def dict_to_df(vocab):
    data = {'path': [], 'color': []}
    for key in vocab:
        data['path'] += vocab[key]
        data['color'] += [key] * len(vocab[key])
    return data

def df_to_dict(df):
    keys = df['color'].unique()
    return {key: list(df[df['color'] == key]['path']) for key in keys}