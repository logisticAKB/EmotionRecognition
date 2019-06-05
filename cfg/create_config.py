from os.path import exists
from configparser import ConfigParser


class Config:
    def __init__(self):
        self.path = "cfg\\train.cfg"

    def load_cfg(self):
        if not exists(self.path):
            self.create_cfg()

        cfg = ConfigParser()
        cfg.read(self.path)

        loaded_cfg = {"num_features": int(cfg.get("Settings", "num_features")),
                      "num_labels": int(cfg.get("Settings", "num_labels")),
                      "batch_size": int(cfg.get("Settings", "batch_size")),
                      "epochs": int(cfg.get("Settings", "epochs")),
                      "width": int(cfg.get("Settings", "width")),
                      "height": int(cfg.get("Settings", "height"))}

        return loaded_cfg

    def create_cfg(self):
        cfg = ConfigParser()
        cfg.add_section("Settings")
        cfg.set("Settings", "num_features", "64")
        cfg.set("Settings", "num_labels", "7")
        cfg.set("Settings", "batch_size", "64")
        cfg.set("Settings", "epochs", "100")
        cfg.set("Settings", "weights_save_state", "-1")
        cfg.set("Settings", "width", "48")
        cfg.set("Settings", "height", "48")

        with open(self.path, "w") as cfg_file:
            cfg.write(cfg_file)
