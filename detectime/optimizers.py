import torch
import logging

log = logging.getLogger(__name__)


def get_optimizer(config, net):
    lr = config.train.learning_rate
    log.info(lr)

    log.info(f"Opt: {config.train.optimizer}")

    if config.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           net.parameters()),
                                    lr=lr,
                                    momentum=config.train.momentum)
    elif config.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            net.parameters()),
                                     lr=lr)
    else:
        raise Exception("Unknown type of optimizer: {}".format(
            config.train.optimizer)
        )
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.train.n_epoch)
    else:
        raise Exception("Unknown type of lr schedule: {}".format(
            config.train.lr_schedule)
        )
    return scheduler
