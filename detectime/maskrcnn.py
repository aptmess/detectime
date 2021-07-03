import logging
from mrcnn.config import Config
from mrcnn import model as model_lib

log = logging.getLogger(__name__)


def load_model_custom(config, load_model_path):
    model = model_lib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir='./model'
    )
    model.load_weights(
        load_model_path,
        by_name=True
    )

    return model


class HandConfig(Config):
    NAME = "hand"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.95


class InferenceConfig(HandConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
