import numpy as np


def binaralize(img):
    hist_data = np.zeros(256)
    threshold = 0

    img_reshaped = img.reshape(-1)
    img_len = len(img_reshaped)
    for i in range(img_len):
        h = int(0xFF & img_reshaped[i])
        hist_data[h] = hist_data[h] + 1

    sum_t = 0
    for i in range(256):
        sum_t = sum_t + i * hist_data[i]

    sum_b = 0
    w_b = 0
    w_f = 0
    max_var = 0

    for t in range(256):
        w_b = w_b + hist_data[t]
        if w_b == 0:
            continue

        w_f = img_len - w_b
        if w_f == 0:
            break

        sum_b = sum_b + t * hist_data[t]

        m_b = sum_b / w_b
        m_f = (sum_t - sum_b) / w_f

        variance = (w_b * w_f) * (m_b - m_f) * (m_b - m_f)
        if variance > max_var:
            max_var = variance
            threshold = t

    return np.where((0xFF & img) >= threshold, 0, 255)


def segment(img):
    w, h = img.shape
    img_ones = (img == 0).astype(int)
    horizontal_hist = np.sum(img_ones, axis=1)
    vertical_hist = np.sum(img_ones, axis=0)
