import numpy as np
import matplotlib.pyplot as plt

from constants import KERNELS


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def euclide(x, y):
    return 1 / np.sum((x - y) ** 2)


def image_to_colors(ndarr):
    return ndarr.transpose([2, 0, 1]).reshape(3, -1)


def get_only_not_black(image):
    return (np.sum(image, axis=2) != 0).sum()


def get_all(image):
    return np.size(image) // 3


def get_avg_colors(images, len_searcher=get_only_not_black):
    return get_avgs_colors(images, len_searcher).mean(axis=0)


def get_std_colors(images, mean, len_searcher=get_only_not_black):
    avgs = get_avgs_colors(images, len_searcher)
    avgs = (avgs - mean) ** 2
    return np.sqrt(avgs.mean(axis=0))


def get_avgs_colors(images, len_searcher=get_only_not_black):
    rgb = []
    for image in images:
        rgb.append(get_avg_colors_one(image, len_searcher))
    return np.array(rgb)


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


def plot_trained(d):
    plt.figure(figsize=(20, 10))
    plt.bar(d.keys(),np.array(list(d.values()))[:,0], color='red', alpha=1)
    plt.bar(d.keys(),np.array(list(d.values()))[:,1], color='green', alpha=0.8)
    plt.bar(d.keys(),np.array(list(d.values()))[:,2], color='blue', alpha=0.4)