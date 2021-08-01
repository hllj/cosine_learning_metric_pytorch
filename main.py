import torch
from absl import app, flags, logging

from model import Feature, Network, init_weights

flags.DEFINE_string("name", "Hop", "Your name.")

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


if __name__ == "__main__":
    app.run(main)
