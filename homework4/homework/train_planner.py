"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
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

    # extract training-only kwargs so they are not forwarded to load_model
    optimizer_name = kwargs.pop("optimizer", "adam").lower()

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", transform_pipeline="state_only", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", transform_pipeline="state_only", shuffle=False)

    # create loss function and optimizer
    loss_func = nn.MSELoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Using optimizer: {optimizer.__class__.__name__}")
    
    # create metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        # reset metrics at beginning of epoch
        train_metric.reset()
        val_metric.reset()

        model.train()

        for batch in train_data:
            track_left, track_right, waypoints, waypoints_mask = batch["track_left"], batch["track_right"], batch["waypoints"], batch["waypoints_mask"]
            track_left, track_right, waypoints, waypoints_mask = track_left.to(device), track_right.to(device), waypoints.to(device), waypoints_mask.to(device)

            # forward pass
            pred = model(track_left, track_right)
            loss = loss_func(pred * waypoints_mask[..., None], waypoints * waypoints_mask[..., None])

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update metrics
            train_metric.add(pred, waypoints, waypoints_mask)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                track_left, track_right, waypoints, waypoints_mask = batch["track_left"], batch["track_right"], batch["waypoints"], batch["waypoints_mask"]
                track_left, track_right, waypoints, waypoints_mask = track_left.to(device), track_right.to(device), waypoints.to(device), waypoints_mask.to(device)

                # forward pass
                pred = model(track_left, track_right)
                
                # update metrics
                val_metric.add(pred, waypoints, waypoints_mask)

        # compute metrics
        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()

        epoch_train_long = train_metrics["longitudinal_error"]
        epoch_train_lat = train_metrics["lateral_error"]
        epoch_val_long = val_metrics["longitudinal_error"]
        epoch_val_lat = val_metrics["lateral_error"]

        # log to tensorboard
        logger.add_scalar("train/longitudinal_error", epoch_train_long, epoch)
        logger.add_scalar("train/lateral_error", epoch_train_lat, epoch)
        logger.add_scalar("val/longitudinal_error", epoch_val_long, epoch)
        logger.add_scalar("val/lateral_error", epoch_val_lat, epoch)
        logger.add_scalar("train/num_samples", train_metrics["num_samples"], epoch)
        logger.add_scalar("val/num_samples", val_metrics["num_samples"], epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_long={epoch_train_long:.4f} "
                f"train_lat={epoch_train_lat:.4f} "
                f"val_long={epoch_val_long:.4f} "
                f"val_lat={epoch_val_lat:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
