import cv2
import numpy as np

CLASS_NAMES_GEOMETRIC = {
    1: 'rectangle',
    2: 'triangle',
    3: 'ellipse'
}


def random_triangle(image, mask, x, y, color, radius_range=(3, 28)):
    a, b, c, d, e = np.random.randint(*radius_range, size=5)
    triangle_cnt = np.array([[x, y - a], [x + b, y + c], [x - d, y + e]])
    cv2.drawContours(image, [triangle_cnt], 0, color, -1)
    cv2.drawContours(mask, [triangle_cnt], 0, 1, -1)
    return image, mask


def random_ellipse(image, mask, x, y, color, radius_range=(3, 28)):
    rh, rw = np.random.randint(*radius_range, size=2)
    angle = np.random.randint(0, 360)
    cv2.ellipse(image, (x, y), axes=(rh, rw), angle=angle, startAngle=0, endAngle=360, color=color, thickness=-1)
    cv2.ellipse(mask, (x, y), axes=(rh, rw), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
    return image, mask


def random_rectangle(image, mask, x, y, color, radius_range=(3, 28)):
    rh, rw = np.random.randint(*radius_range, size=2)
    angle = np.random.randint(0, 360)
    tmp = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(tmp, (x, y), axes=(rh, rw), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
    c = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = c[0] if len(c) == 2 else c[1]
    box = np.int0(cv2.boxPoints(cv2.minAreaRect(c[0])))
    cv2.drawContours(image, [box], 0, color, -1)
    cv2.drawContours(mask, [box], 0, 1, -1)
    return image, mask


def random_circle(image, mask, x, y, color, radius_range=(3, 28)):
    rad = np.random.randint(*radius_range)
    cv2.circle(image, center=(x, y), radius=rad, color=color, thickness=-1)
    cv2.circle(mask, center=(x, y), radius=rad, color=1, thickness=-1)
    return image, mask


def random_geometric_objects(height=256, width=256, radius_range=(3, 28), intensity_range=(0, 180), margin=13):
    img = np.zeros((height, width, 3), dtype='uint8') + 255
    mrad = np.max(radius_range)
    xa, xb = margin + mrad, img.shape[1] - mrad - margin
    ya, yb = margin + mrad, img.shape[0] - mrad - margin
    step = int(mrad * 1.5)
    xy = np.mgrid[xa:xb:step, ya:yb:step].reshape((2, -1))
    xy_rad = np.random.randint(*radius_range, xy.shape[1:])
    masks = []
    labels = []
    classes = []
    for idx, (x, y) in enumerate(xy.T):
        rad = xy_rad[idx]
        x = np.clip(int(x), 0, img.shape[1]) + np.random.randint(0, int(rad * .5))
        y = np.clip(int(y), 0, img.shape[0]) + np.random.randint(0, int(rad * .5))
        color = np.random.randint(*intensity_range, 3)
        variant = np.random.choice([1, 2, 3])
        mask = np.zeros((height, width), dtype='uint8')
        classes.append(variant)
        if variant == 1:
            img, mask = random_rectangle(img, mask, x, y, color.tolist(), radius_range=radius_range)
        elif variant == 2:
            img, mask = random_triangle(img, mask, x, y, color.tolist(), radius_range=radius_range)
        else:
            img, mask = random_ellipse(img, mask, x, y, color.tolist(), radius_range=radius_range)
        masks.append(mask)
        label = np.copy(mask).astype('int')
        label[label > .5] += idx
        labels.append(label)
    return img, np.array(masks), np.stack(labels, -1), np.array(classes)
