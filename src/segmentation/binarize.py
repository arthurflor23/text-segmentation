import numpy as np
import cv2 as cv

### su ###

nfns = [
    lambda x: np.roll(x, -1, axis=0),
    lambda x: np.roll(np.roll(x, 1, axis=1), -1, axis=0),
    lambda x: np.roll(x, 1, axis=1),
    lambda x: np.roll(np.roll(x, 1, axis=1), 1, axis=0),
    lambda x: np.roll(x, 1, axis=0),
    lambda x: np.roll(np.roll(x, -1, axis=1), 1, axis=0),
    lambda x: np.roll(x, -1, axis=1),
    lambda x: np.roll(np.roll(x, -1, axis=1), -1, axis=0)
]

def su_plus(img):
    _, otsu_ocimg = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    su_ocimg = su(img)

    contours, _ = cv.findContours(otsu_ocimg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        approx = cv.convexHull(contour)
        cv.drawContours(otsu_ocimg, [approx], -1, (0, 255, 0), 1)

    cv.floodFill(otsu_ocimg, None, (16, 16), 0)
    im_floodfill = su_ocimg.copy()
    cv.floodFill(im_floodfill, None, (16, 16), 0)

    R = (otsu_ocimg + (su_ocimg & cv.bitwise_not(im_floodfill)))
    
    mask = np.ones((3, 3), np.uint8)
    R = cv.erode(R, None, iterations=1)
    R = cv.dilate(R, None, iterations=1)

    R = cv.medianBlur(R, 3)
    _, R = cv.threshold(R, 0, 255, cv.THRESH_BINARY)

    return R


def su(img):
    gfn = nfns
    N_MIN = 4

    I = img.astype(np.float64)
    cimg = localminmax(I, gfn)

    _, ocimg = cv.threshold(rescale(cimg).astype(img.dtype), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    E = ocimg.astype(np.float64)
    E_mean = np.zeros(I.shape, dtype=np.float64)

    N_e = numnb(ocimg, gfn)
    nbmask = N_e > 0

    for fn in gfn:
        E_mean += fn(I) * fn(E)

    E_mean[nbmask] /= N_e[nbmask]
    E_var = np.zeros(I.shape, dtype=np.float64)

    for fn in gfn:
        tmp = (fn(I) - E_mean) * fn(E)
        E_var += tmp * tmp

    E_var[nbmask] /= N_e[nbmask]
    E_std = np.sqrt(E_var) * 0.5

    R = np.ones(I.shape) * 255
    R[(I <= E_mean+E_std) & (N_e >= N_MIN)] = 0

    return R.astype(np.uint8)


def localminmax(img, fns):
    mi = img.astype(np.float64)
    ma = img.astype(np.float64)
    for i in range(len(fns)):
        rolled = fns[i](img)
        np.minimum(mi, rolled, out=mi)
        np.maximum(ma, rolled, out=ma)
    result = (ma-mi) / (mi+ma+1e-16)
    return result


def numnb(bi, fns):
    nb = bi.astype(np.float64)
    i = np.zeros(bi.shape, nb.dtype)
    i[bi == bi.max()] = 1
    i[bi == bi.min()] = 0
    for fn in fns:
        nb += fn(i)
    return nb


def rescale(r, maxvalue=255):
    mi = r.min()
    return maxvalue*(r-mi) / (r.max()-mi)


### sauvola ###


def sauvola(img, window, dr, k):
    rows, cols = img.shape
    impad = padding(img, window)

    mean, sqmean = integralMean(impad, rows, cols, window)
    n = window[0] * window[1]
    variance = (sqmean - (mean**2) / n) / n
    std = variance ** 0.5

    threshold = mean * (1 + k * (std / dr - 1))
    check_border = (mean >= 100)
    threshold = threshold * check_border

    output = np.array(255 * (img >= threshold), 'uint8')

    return output


def padding(img, window):
    pad = int(np.floor(window[0] / 2))
    img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT)

    return img


def integralMean(img, rows, cols, window):
    m, n = window
    sum, sqsum = cv.integral2(img)
    isum = sum[m:rows + m, n:cols + n] + \
        sum[0:rows, 0:cols] - \
        sum[m:rows + m, 0:cols] - \
        sum[0:rows, n:cols + n]

    isqsum = sqsum[m:rows + m, n:cols + n] + \
        sqsum[0:rows, 0:cols] - \
        sqsum[m:rows + m, 0:cols] - \
        sqsum[0:rows, n:cols + n]

    mean = isum / (m * n)
    sqmean = isqsum / (m * n)

    return mean, sqmean


def ind2sub(ind, shape):
    row = ind / shape[1]
    col = ind % shape[1]

    return row, col


### otsu ###


def otsu(img):
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    return binary
