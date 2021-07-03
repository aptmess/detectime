import json
import logging
import cv2
import torch
from collections import defaultdict
from torchvision import transforms as tfs
from detectime.augmentations import (
    get_train_aug,
    get_val_aug
)
from detectime.utils import read_image

log = logging.getLogger(__name__)



class GestureDataset(object):
    def __init__(self,
                 config,
                 img_path,
                 folder_annotation,
                 is_train):
        if is_train:
            self.annotations_main = folder_annotation
            self.transforms = get_train_aug(config)
        else:
            self.annotations_main = folder_annotation
            self.transforms = get_val_aug(config)
        self.img_path = img_path
        self.preprocess = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        self.data = defaultdict(list)
        self.read_annotations()
        self.dataset_length = len(self.data)

    def read_annotations(self):
        with open(self.annotations_main) as f:
            data_json = json.load(f)

        self.data = data_json

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        cv2.setNumThreads(6)
        sample = self.data[idx]
        image = read_image(
            str(self.img_path / sample['frame_path'])
        )
        x1, y1, x2, y2 = sample['bbox']
        cropped_image = image[y1:y2, x1:x2]
        crop, _ = self.transforms(cropped_image, None)
        crop = self.preprocess(crop)
        return crop, sample['label'] - 1


def get_data_loaders(config,
                     img_path,
                     train_annotation,
                     test_annotation):

    log.info("Preparing train reader...")
    train_dataset = GestureDataset(
        config=config,
        img_path=img_path,
        folder_annotation=train_annotation,
        is_train=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True
    )
    log.info("Done.")

    log.info("Preparing valid reader...")
    val_dataset = GestureDataset(
        config=config,
        img_path=img_path,
        folder_annotation=test_annotation,
        is_train=False
    )
    valid_loader = torch.utils.data.DataLoader(
         val_dataset,
         batch_size=config.dataset.batch_size,
         shuffle=False,
         num_workers=config.dataset.num_workers,
         drop_last=False,
         pin_memory=True
    )
    log.info("Done.")
    return train_loader, valid_loader
