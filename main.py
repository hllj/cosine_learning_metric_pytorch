import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from absl import app, flags, logging

from model import Feature, Network, init_weights

from config import Config

from dataset import VRICDataset

flags.DEFINE_string("config", "config/config.json", "Config file for project")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    # net = Network(num_classes=1500)
    # net.apply(init_weights)
    # print(10 * "-" + "Network" + 10 * "-")
    # print(str(net))
    # print(10 * "-" + "Testing model" + 10 * "-")
    # x = torch.randn(64, 3, 128, 64)
    # print(net(x).shape)

    print(10 * "-" + "Testing dataset" + 10 * "-")

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
    val_dataloader = DataLoader(
        val_ds,
        batch_size=val_data_config["batch_size"],
        shuffle=val_data_config["shuffle"],
    )
    logging.get_absl_handler().use_absl_log_file("absl_logging", "./")
    logging.info("test")
    import IPython

    IPython.embed()
    exit(1)


if __name__ == "__main__":
    app.run(main)
