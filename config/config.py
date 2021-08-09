import json


class Config:
    def __init__(self, config_path):
        config_file = open(config_path)

        self.config_data = json.load(config_file)

    def _get_data_loader(self):
        return self.config_data["data_loader"]

    def _get_ngpu(self):
        return self.config_data["n_gpu"]
