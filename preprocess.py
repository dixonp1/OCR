import numpy as np


def binaralize(img):
    hist, bins = np.histogram(img.ravel(), np.arange(256))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist = hist.astype(float)

    wf = np.cumsum(hist)
    wb = np.cumsum(hist[::-1])[::-1]

    mf = np.cumsum(hist * bin_centers) / wf
    mb = (np.cumsum((hist * bin_centers)[::-1]) / (wb[::-1] + 1e-16))[::-1]

    variance = wf[:-1] * wb[1:] * (mf[:-1] - mb[1:]) ** 2

    idx = np.argmax(variance)
    threshold = bins[:-1][idx]
    bimg = np.zeros_like(img)
    bimg[img >= threshold] = 255
    return bimg


def segment(img):
    img_ones = (img == 0).astype(int)
    horizontal_hist = np.sum(img_ones, axis=1)
    vertical_hist = np.sum(img_ones, axis=0)

    return horizontal_hist, vertical_hist


