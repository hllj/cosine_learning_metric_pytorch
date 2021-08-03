import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from absl import app, flags, logging

from model import Feature, Network, init_weights

from dataset import VRICDataset

flags.DEFINE_integer("batchsize", 8, "Batch size")

FLAGS = flags.FLAGS


def main(argv):
    del argv
    net = Network(num_classes=1500)
    net.apply(init_weights)
    print(10 * "-" + "Network" + 10 * "-")
    print(str(net))
    print(10 * "-" + "Testing model" + 10 * "-")
    x = torch.randn(64, 3, 128, 64)
    print(net(x).shape)

    print(10 * "-" + "Testing dataset" + 10 * "-")
    train_transforms = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((128, 64)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((128, 64)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_ds = VRICDataset(transforms=train_transforms, split="train")
    val_ds = VRICDataset(transforms=val_transforms, split="val")

    train_dataloader = DataLoader(train_ds, batch_size=FLAGS.batchsize, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=FLAGS.batchsize, shuffle=True)
    import IPython

    IPython.embed()
    exit(1)


if __name__ == "__main__":
    app.run(main)
