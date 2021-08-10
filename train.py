import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from absl import app, flags, logging
from tqdm import tqdm

from model import Network, init_weights

from config import Config

from dataset import VRICDataset

from utils import prepare_device

flags.DEFINE_string("config", "config/config.json", "Config file for project")

FLAGS = flags.FLAGS


def main(argv):
    del argv

    config = Config(FLAGS.config)

    data_loader_config = config._get_data_loader()

    train_data_config = data_loader_config["train"]
    val_data_config = data_loader_config["val"]

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                (data_loader_config["img_width"], data_loader_config["img_height"])
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                (data_loader_config["img_width"], data_loader_config["img_height"])
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_ds = VRICDataset(transforms=train_transforms, split="train")
    val_ds = VRICDataset(transforms=val_transforms, split="val")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=train_data_config["batch_size"],
        shuffle=train_data_config["shuffle"],
    )

    # device
    device, device_ids = prepare_device(config.config_data["n_gpu"])

    # model
    num_classes = max(train_ds.num_classes, val_ds.num_classes)
    net = Network(num_classes)
    net.apply(init_weights)
    net.to(device)

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.config_data["optimizer"]["lr"],
        weight_decay=config.config_data["optimizer"]["weight_decay"],
    )

    for epoch in range(config.config_data["epochs"]):
        # train one epoch
        net.train()
        train_loss = 0.0
        for data, target in tqdm(train_dataloader):
            images, labels = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(
            10 * "-" + "Epoch {}: Train Loss {:.6f}".format(
                epoch, train_loss / len(train_dataloader)
            ) + 10 * "-"
        )

if __name__ == "__main__":
    app.run(main)
