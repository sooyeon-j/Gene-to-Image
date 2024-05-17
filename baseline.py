import argparse
from contextlib import nullcontext
import os
import copy
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torch import optim
import torch.nn as nn

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from types import SimpleNamespace
from fastprogress import progress_bar, master_bar
from utils_bl import *
from torchvision import models
import logging
import wandb
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = False

config = SimpleNamespace(
    run_name="baseline",
    dataset="AttnGAN",
    epochs=200,
    seed=42,
    batch_size=16,
    width=256,
    height=256,
    num_classes=3,
    device="cuda",
    slice_size=1,
    use_wandb=True,
    do_validation=True,
    fp16=True,
    log_every_epoch=10,
    num_workers=10,
    lr=0.00001,  # 0.001
    test_balanced=True,
    use_sampled=True,
)

# print(config)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(0.5),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = torch.nn.Linear(32 * 64 * 64, 64, bias=True)
        self.fc2 = torch.nn.Linear(32 * 64 * 64, 3, bias=True)

        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        out = self.layer1(x)  # (bs, 32, 128, 128)
        out = self.layer2(out)  # (bs, 32, 64, 64)
        # out = self.layer3(out) # (bs, 64, 32, 32)
        out = self.flatten(out)
        # out = self.fc1(out)
        # out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        # out = self.sigmoid(out)
        return out


class Baseline:
    def __init__(self, num_classes=3, device="cuda"):

        self.model = CNN()
        # self.model.classifier._modules['6'] = nn.Linear(4096, num_classes)
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device)
        self.device = device

    def prepare(self, args):
        mk_folders(args.run_name, args.num_classes)
        (
            self.train_loader,
            self.gen_loader,
            self.val_loader,
            self.test_loader,
            self.concat_loader,
        ) = get_mri(args)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=0.05
        )
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
        #  steps_per_epoch=len(self.train_dataset), epochs=args.epochs)
        self.ce = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # self.scheduler.step()

    def save_model(self, run_name, use_wandb=False, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model, os.path.join("models_bl", run_name, f"model.pt"))
        torch.save(self.optimizer, os.path.join("models_bl", run_name, f"optim.pt"))
        if use_wandb:
            at = wandb.Artifact(
                "model",
                type="model",
                description="Model weights for classification",
                metadata={"epoch": epoch},
            )
            at.add_dir(os.path.join("models_bl", run_name))
            wandb.log_artifact(at)

    def one_epoch(self, train=True, use_wandb=False):
        avg_loss = []
        avg_acc = []
        avg_prec = []
        avg_rec = []
        avg_f1 = []

        if train:
            self.model.train()
            pbar = progress_bar(self.train_loader, leave=False)
        else:
            self.model.eval()
            pbar = progress_bar(self.test_loader, leave=False)

        for i, (images, labels) in enumerate(pbar):
            with torch.cuda.amp.autocast() and (
                torch.inference_mode() if not train else torch.enable_grad()
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                ### concat generated model
                gen_image, gen_labels = next(iter(self.gen_loader))
                gen_image, gen_labels = gen_image.to(self.device), gen_labels.to(
                    self.device
                )
                if config.use_sampled:
                    images = torch.cat((images, gen_image), 0)
                    labels = torch.cat((labels, gen_labels), 0)

                output = self.model(images)
                predicted = torch.argmax(output, 1)
                loss = self.ce(output, labels)
                acc = accuracy_score(labels.detach().cpu(), predicted.detach().cpu())
                prec = precision_score(
                    labels.detach().cpu(),
                    predicted.detach().cpu(),
                    average="macro",
                    zero_division=np.nan,
                )
                rec = recall_score(
                    labels.detach().cpu(),
                    predicted.detach().cpu(),
                    average="macro",
                    zero_division=np.nan,
                )
                f1 = f1_score(
                    labels.detach().cpu(),
                    predicted.detach().cpu(),
                    average="macro",
                    zero_division=np.nan,
                )

                avg_loss.append(loss.item())
                avg_prec.append(prec)
                avg_rec.append(rec)
                avg_f1.append(f1)
                avg_acc.append(acc)

            if train:
                self.train_step(loss)
                if use_wandb:
                    wandb.log({"train_ce_one_epoch": loss.item()})
                    # "learning_rate": self.scheduler.get_last_lr()[0]})
            else:
                if use_wandb:
                    wandb.log({"val_ce_one_epoch": loss.item()})
            pbar.comment = f"TRAIN: {train}, LOSS={loss.item():2.3f}, ACC={acc:2.3f}"
        return (
            np.mean(avg_loss),
            np.mean(avg_acc),
            np.mean(avg_prec),
            np.mean(avg_rec),
            np.mean(avg_f1),
        )

    def fit(self, args):
        avg_loss = []
        avg_acc = []
        best_val_acc = 0

        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            (
                train_avg_loss,
                train_avg_acc,
                train_avg_prec,
                train_avg_rec,
                train_avg_f1,
            ) = self.one_epoch(train=True, use_wandb=args.use_wandb)

            if args.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_avg_loss,
                        "train_acc": train_avg_acc,
                        "train_rec": train_avg_rec,
                        "train_prec": train_avg_prec,
                        "train_f1": train_avg_f1,
                    }
                )
                # "learning_rate": self.scheduler.get_last_lr()[0]})

            if args.do_validation:
                (
                    val_avg_loss,
                    val_avg_acc,
                    val_avg_prec,
                    val_avg_rec,
                    val_avg_f1,
                ) = self.one_epoch(train=False, use_wandb=args.use_wandb)
                if args.use_wandb:
                    wandb.log(
                        {
                            "val_loss": val_avg_loss,
                            "val_acc": val_avg_acc,
                            "val_rec": val_avg_rec,
                            "val_prec": val_avg_prec,
                            "val_f1": val_avg_f1,
                        }
                    )
                    print("val_loss", val_avg_loss, "val_acc", val_avg_acc)

                avg_loss.append(val_avg_loss)
                avg_acc.append(val_avg_acc)

                if val_avg_acc >= best_val_acc:
                    best_val_acc = val_avg_acc
                    (
                        test_avg_loss,
                        test_avg_acc,
                        test_avg_prec,
                        test_avg_rec,
                        test_avg_f1,
                    ) = self.one_epoch(train=False, use_wandb=args.use_wandb)
                    if args.use_wandb:
                        wandb.log(
                            {
                                "test_loss": test_avg_loss,
                                "test_acc": test_avg_acc,
                                "test_rec": test_avg_rec,
                                "test_prec": test_avg_prec,
                                "test_f1": test_avg_f1,
                            }
                        )
                        print("test_loss", test_avg_loss, "test_acc", test_avg_acc)

        self.save_model(run_name=args.run_name, use_wandb=args.use_wandb, epoch=epoch)
        print(f"average loss: {np.mean(avg_loss):.3f}")
        print(f"average acc: {np.mean(avg_acc):.3f}")


def parse_args(config):
    parser = argparse.ArgumentParser(description="Process hyper-parameters")
    parser.add_argument(
        "--run_name", type=str, default=config.run_name, help="name of the run"
    )
    parser.add_argument(
        "--dataset", type=str, default=config.dataset, help="name of the dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.epochs, help="number of epochs"
    )
    parser.add_argument("--seed", type=int, default=config.seed, help="random seed")
    parser.add_argument(
        "--batch_size", type=int, default=config.batch_size, help="batch size"
    )
    parser.add_argument("--width", type=int, default=config.width, help="image size")
    parser.add_argument("--height", type=int, default=config.height, help="image size")
    parser.add_argument(
        "--num_classes", type=int, default=config.num_classes, help="number of classes"
    )
    parser.add_argument("--device", type=str, default=config.device, help="device")
    parser.add_argument(
        "--use_wandb", type=bool, default=config.use_wandb, help="use wandb"
    )
    parser.add_argument(
        "--test_balanced", type=bool, default=config.test_balanced, help="test balance"
    )

    parser.add_argument(
        "--use_sampled", type=bool, default=config.use_sampled, help="use sample"
    )

    parser.add_argument("--lr", type=float, default=config.lr, help="learning rate")
    parser.add_argument(
        "--slice_size", type=int, default=config.slice_size, help="slice size"
    )
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == "__main__":
    
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    baseline = Baseline()
    with (
        wandb.init(project="baseline", group="CNN", config=config)
        if config.use_wandb
        else nullcontext()
    ):
        baseline.prepare(config)
        baseline.fit(config)
