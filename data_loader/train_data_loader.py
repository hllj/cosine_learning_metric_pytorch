from torchvision import datasets, transforms
from base import BaseDataLoader

class TrainDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
