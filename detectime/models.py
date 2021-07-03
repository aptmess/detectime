import logging
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)


def load_model(config,
               device='cuda'):
    if config.model.model_type == 'resnet34':
        log.info("ResNet34")
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,
                             config.dataset.num_of_classes)
    elif config.model.model_type == 'resnext101_32x8d':
        log.info("ResNext101_32x8d")
        model = models.resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,
                             config.dataset.num_of_classes)
    elif config.model.model_type == 'resnet152':
        log.info("ResNet152")
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,
                             config.dataset.num_of_classes)
    else:
        raise Exception('model type is not supported:',
                        config.model.model_type)
    model.to(device)
    return model
