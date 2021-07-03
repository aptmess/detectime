import os
import cv2
import json
import torch
import logging
import pandas as pd
import numpy as np
from collections import namedtuple, OrderedDict
from torchvision import models

log = logging.getLogger(__name__)


def read_image(image_path):
    img = cv2.imread(image_path,
                     cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def load_resnet(path,
                model_type,
                num_classes,
                device='cuda'):
    if model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(512, num_classes)
        model.load_state_dict(
            torch.load(path, map_location='cpu')["state_dict"]
        )
    else:
        raise Exception("Unknown model type: {}".format(model_type))
    model.to(device)
    model.eval()
    return model


def save_results(scores,
                 frame_paths,
                 save_path):
    result_df = pd.DataFrame({
        'no_gesture': scores[:, 0],
        'stop': scores[:, 1],
        'victory': scores[:, 2],
        'mute': scores[:, 3],
        'ok': scores[:, 4],
        'like': scores[:, 5],
        'dislike': scores[:, 6],
        'frame_path': frame_paths
    })

    result_df.to_csv(save_path, index=False)


def train_valid_split(
        train_hands_path,
        output_save_dir,
        val_size=0.15):
    annotations = pd.read_json(str(train_hands_path))
    log.info(len(annotations))
    all_videos = list(annotations['video_name'].unique())
    log.info(len(all_videos))
    number_val_samples = int(len(all_videos) * val_size)
    val_videos = np.random.choice(all_videos,
                                  size=number_val_samples,
                                  replace=False)
    val_data = annotations[
        annotations['video_name'].isin(val_videos)
    ]
    train_data = annotations[
        ~(annotations['video_name'].isin(val_videos))
    ]
    log.info(f'train data length={len(train_data)}')
    log.info(f'validation data length={len(val_data)}')
    log.info(f'Savedir train and valid data: {output_save_dir}')
    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir)
    train_data.to_json(
        str(output_save_dir / 'train.json'),
        orient='records',
        indent=4
    )
    val_data.to_json(
        str(output_save_dir / 'valid.json'),
        orient='records',
        indent=4
    )

    return train_data, val_data


def save_checkpoint(model, optimizer, scheduler, epoch, outdir):
    """Saves checkpoint to disk"""
    filename = "model_{:04d}.pth".format(epoch)
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict([
        ('state_dict', weights),
        ('optimizer', optimizer.state_dict()),
        ('scheduler', scheduler.state_dict()),
        ('epoch', epoch),
    ])

    torch.save(state, filename)





