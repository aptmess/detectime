import cv2
import logging
import numpy as np
import albumentations

log = logging.getLogger(__name__)


def image_crop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return x1, x2, y1, y2


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0),
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def result_crop(img,
                face,
                crop_coefficient=1.5):
    x, y, w, h = face
    max_size = max(w, h)
    x_c = x + w / 2
    y_c = y + h / 2

    x4 = max_size * crop_coefficient + x_c
    y4 = max_size * crop_coefficient + y_c
    x3 = - max_size * crop_coefficient + x_c
    y3 = - max_size * crop_coefficient + y_c

    x3, x4, y3, y4 = image_crop(img, [int(x3), int(y3), int(x4), int(y4)])
    return x3, y3, x4, y4


def get_max_bbox(face_boxes):
    face_frame_sizes = [x[2] * x[3] for x in face_boxes]
    max_frame_index = np.argmax(face_frame_sizes)
    return face_boxes[max_frame_index]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, annotation):
        for t in self.transforms:
            img, annotation = t(img, annotation)
        return img, annotation


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, annotation):
        img = cv2.resize(image, (self.size, self.size))
        return img, annotation


class PreparedAug(object):
    def __init__(self):
        augmentation = [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=10, p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=20,
                                              sat_shift_limit=40,
                                              val_shift_limit=50,
                                              p=0.5),
            albumentations.GaussianBlur(p=0.4),
            albumentations.ToGray(p=0.3)
        ]
        self.augmentation = albumentations.Compose(augmentation)

    def __call__(self, image, annotation):
        image = self.augmentation(image=image)['image']
        return image, annotation


class DefaultAugmentations(object):
    def __init__(self, config):
        self.augment = Compose([
            Resize(size=config.dataset.input_size),
            PreparedAug()
        ])

    def __call__(self, image, annotation):
        return self.augment(image, annotation)


class ValidationAugmentations(object):
    def __init__(self, config):
        self.augment = Compose([
            Resize(size=config.dataset.input_size),
        ])

    def __call__(self, image, annotation):
        return self.augment(image, annotation)


def get_train_aug(config):
    if config.dataset.augmentations == 'default':
        train_augmentation = DefaultAugmentations(config)
    else:
        raise Exception("Unknown type of augmentation: {}".format(
            config.dataset.augmentations)
        )
    return train_augmentation


def get_val_aug(config):
    if config.dataset.augmentations_valid == 'default':
        val_augmentation = ValidationAugmentations(config)
    else:
        raise Exception("Unknown type of augmentation: {}".format(
            config.dataset.augmentations)
        )
    return val_augmentation
