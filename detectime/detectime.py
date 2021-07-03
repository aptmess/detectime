# IMPORT
import cv2
import torch
import logging
import numpy as np
import pandas as pd
import face_detection as fd
import matplotlib.pyplot as plt

# FROM
from tqdm import tqdm
from torchvision import transforms as tfs

# IMPORTS MODULES
from detectime.maskrcnn import (
    load_model_custom,
    InferenceConfig
)

from detectime.augmentations import ValidationAugmentations

from detectime.utils import (
    load_resnet,
    save_results,
)
from detectime.detection import (
    detect_hand,
    modified_hand_detection
)

log = logging.getLogger(__name__)

# BASE_PARAMS

CLASS_NAME2LABEL_DICT = {
    0: 'no_gesture',
    1: 'stop',
    2: 'victory',
    3: 'mute',
    4: 'ok',
    5: 'like',
    6: 'dislike'
}

WHO_DETECT = {
    0: 'HAND',
    1: 'FACE',
    2: 'NONE'
}


def detectron(config):
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    model_detector_faces = fd.build_detector(
        config.detection.detector_type,
        confidence_threshold=.5,
        nms_iou_threshold=.3,
        device=device,
        max_resolution=640
    )

    model_detector_hands = load_model_custom(
        config=InferenceConfig(),
        load_model_path=config.weights.detection_model_path
    )

    model = load_resnet(
        path=config.weights.classification_model_path,
        model_type=config.model.model_type,
        num_classes=config.dataset.num_of_classes,
        device=device_name
    )

    softmax_func = torch.nn.Softmax(dim=1)
    validation_augmentation = ValidationAugmentations(config=config)
    preprocess = tfs.Compose(
        [
            tfs.ToTensor(),
            tfs.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    test_df = pd.read_csv(config.path.input_path)
    len_ = len(test_df)
    scores = np.zeros((len_, 7), dtype=np.float32)

    for idx, image_path in tqdm(enumerate(test_df.frame_path.values),
                                total=len_):
        img = cv2.imread(image_path,
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if config.detection.detection_type == 'modified':
            final_image, finder = modified_hand_detection(
                img=img,
                model_detector_faces=model_detector_faces,
                model_detector_hands=model_detector_hands
            )
        else:
            final_image, finder = detect_hand(
                img=img,
                model_detector_faces=model_detector_faces,
                model_detector_hands=model_detector_hands
            )
        log.info(f'DETECTOR: {WHO_DETECT[finder]}')
        if config.utils.show_image_after_detection:
            plt.imshow(final_image)
            plt.show()
        if finder == 2:
            scores[idx, 0] = 1
        else:
            crop, _ = validation_augmentation(
                image=final_image,
                annotation=None
            )
            crop = preprocess(crop).unsqueeze(0)
            crop = crop.to(device_name)
            out = model(crop)
            out = softmax_func(out).squeeze().detach().cpu().numpy()
            scores[idx] = np.r_[0, out]
        if config.utils.show_gesture_prediction_result:
            value = list(scores[idx])
            max_prob = max(value)
            predicted_label = value.index(max_prob)
            plt.imshow(img)
            plt.title(f"""
            predicted gesture: {CLASS_NAME2LABEL_DICT[predicted_label]}, 
            probability: {max_prob:.3}""")
            plt.show()

        if idx % 1000 == 0 and idx > 0:
            save_results(
                scores=scores,
                frame_paths=test_df.frame_path.values,
                save_path=config.path.out_path
            )

    save_results(
        scores=scores,
        frame_paths=test_df.frame_path.values,
        save_path=config.path.out_path
    )
