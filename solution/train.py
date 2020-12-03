import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('box*', batch_size=16, transform=transform)

    det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    size_loss = torch.nn.MSELoss(reduction='none')
    depth_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    is_puck_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()

        for img, gt_depth, gt_det, gt_size, gt_is_puck in train_data:
            img, gt_depth, gt_det, gt_size, gt_is_puck = img.to(device), gt_depth.to(device), gt_det.to(device), gt_size.to(device), gt_is_puck.float().to(device)

            size_w, _ = gt_det.max(dim=1, keepdim=True)

            det, size, depth, is_puck = model(img)
            # Continuous version of focal loss
            p_det = torch.sigmoid(det * (1-2*gt_det))
            det_loss_val = (det_loss(det, gt_det)*p_det).mean() / p_det.mean()
            size_loss_val = (size_w * size_loss(size, gt_size)).mean() / size_w.mean()
            depth_loss_val = depth_loss(depth, gt_depth) * 0.5
            is_puck_loss_val = is_puck_loss(is_puck, gt_is_puck) * 0.1
            loss_val = det_loss_val + size_loss_val * args.size_weight + depth_loss_val + is_puck_loss_val

            if train_logger is not None and global_step % 25 == 0:
                log(train_logger, img, gt_det, det, gt_depth, depth, global_step)

            if train_logger is not None:
                train_logger.add_scalar('det_loss', det_loss_val, global_step)
                train_logger.add_scalar('size_loss', size_loss_val, global_step)
                train_logger.add_scalar('depth_loss', depth_loss_val, global_step)
                train_logger.add_scalar('is_puck_loss', is_puck_loss_val, global_step)
                train_logger.add_scalar('loss', loss_val, global_step)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if valid_logger is None or train_logger is None:
            print('epoch %-3d' %
                  (epoch))
        save_model(model)

def log(logger, imgs, gt_det, det, gt_depth, depth, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('det_label', gt_det[:16,0:3,:,:], global_step)
    logger.add_images('det_pred', torch.sigmoid(det[:16,0:3,:,:]), global_step)
    logger.add_images('depth_label', gt_depth[:16], global_step)
    logger.add_images('depth_pred', torch.sigmoid(depth[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=120)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.4, 0.4, 0.4, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
                        #default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)

    args = parser.parse_args()
    train(args)
