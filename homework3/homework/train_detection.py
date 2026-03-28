import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets import load_road_data
from .metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    transform_pipeline: str = "default",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_road_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_road_data("drive_data/val", transform_pipeline="default", shuffle=False)

    # create loss functions and optimizer
    segmentation_loss_func = nn.CrossEntropyLoss()
    depth_loss_func = nn.L1Loss()  # Mean Absolute Error for depth
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # create metrics
    train_metric = DetectionMetric()
    val_metric = DetectionMetric()

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        # reset metrics at beginning of epoch
        train_metric.reset()
        val_metric.reset()

        model.train()

        for batch in train_data:
            img = batch["image"].to(device)
            track_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)

            # forward pass
            logits, raw_depth = model(img)
            
            # compute losses
            seg_loss = segmentation_loss_func(logits, track_labels)
            depth_loss = depth_loss_func(raw_depth, depth_labels)
            
            # combine losses (you can adjust the weighting)
            total_loss = seg_loss + depth_loss

            # backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # update metrics
            with torch.inference_mode():
                preds, depth_preds = model.predict(img)
                train_metric.add(preds, track_labels, depth_preds, depth_labels)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                img = batch["image"].to(device)
                track_labels = batch["track"].to(device)
                depth_labels = batch["depth"].to(device)

                # forward pass
                logits, raw_depth = model(img)
                
                # update metrics
                preds, depth_preds = model.predict(img)
                val_metric.add(preds, track_labels, depth_preds, depth_labels)

        # compute metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()

        # extract individual metrics
        train_iou = train_metrics["iou"]
        train_depth_error = train_metrics["abs_depth_error"]
        train_tp_depth_error = train_metrics["tp_depth_error"]
        
        val_iou = val_metrics["iou"]
        val_depth_error = val_metrics["abs_depth_error"]
        val_tp_depth_error = val_metrics["tp_depth_error"]

        # log to tensorboard
        logger.add_scalar("train/iou", train_iou, epoch)
        logger.add_scalar("train/depth_error", train_depth_error, epoch)
        logger.add_scalar("train/tp_depth_error", train_tp_depth_error, epoch)
        
        logger.add_scalar("val/iou", val_iou, epoch)
        logger.add_scalar("val/depth_error", val_depth_error, epoch)
        logger.add_scalar("val/tp_depth_error", val_tp_depth_error, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_iou={train_iou:.4f} "
                f"val_iou={val_iou:.4f} "
                f"train_depth_err={train_depth_error:.4f} "
                f"val_depth_err={val_depth_error:.4f} "
                f"train_tp_depth_err={train_tp_depth_error:.4f} "
                f"val_tp_depth_err={val_tp_depth_error:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--transform_pipeline", type=str, default="default", choices=["default", "aug"])

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))