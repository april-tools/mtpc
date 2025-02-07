import os
import re
import hydra
import torch
import OmegaConf

from nanogpt.utils import get_local_device


class Checkpoint(object):
    """In the checkpoint folder we keep:
    1. a yaml config with the model spec
    2. a model@steps.pt file with the state dict of the model,
    optimizer, scheduler, etc."""

    def __init__(self, folder, config, global_step=0):
        super().__init__()
        self.folder = folder
        self.config = config
        self.global_step = global_step
        self.configpath = os.path.join(self.folder, "config.yaml")
        self.expname = self.config.expname
        assert global_step >= 0
        if self.global_step == 0:
            self.modelpath = None
        else:
            self.modelpath = os.path.join(self.folder, "model@%d.pt" % global_step)

    def __repr__(self):
        return '%s@%s' % (self.expname, self.global_step)

    def _load_state(self, device=None):
        if device is None:
            device = get_local_device()
        # load state dict from the saved file
        # and load state dict for the items passed in
        state = torch.load(self.modelpath, weights_only=True, map_location=device)
        return state

    def save(self, global_step=0, model=None, optimizer=None, scheduler=None, **kwargs):
        assert global_step >= 0
        # We haven't begun training, just serialise the config file
        if global_step == 0:
            with open(self.configpath, "w") as f:
                OmegaConf.save(self.config, f)
        else:
            assert model is not None
            model_state_dict = model.state_dict()
            optimizer_state_dict = None if optimizer is None else optimizer.state_dict()
            scheduler_state_dict = None if scheduler is None else scheduler.state_dict()

        state = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
        }
        torch.save(save, self.modelpath)

    def restore(self, model, optimizer=None, scheduler=None, device=None):
        # NOTE: modifies inplace
        # Load state_dict from checkpoint and apply to the objects
        # If global_step == 0, there is no object
        assert self.global_step != 0
        state = self._load_state(device=device)

        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if schedular is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])

    @property
    def model(self):
        model = hydra.utils.instantiate(cfg.model).model
        # If we have begun training, you are getting the saved model
        if global_step > 0:
            state = self._load_state(device=device)
            model.load_state_dict(state["model_state_dict"])
        # Otherwise, you get a randomly initialised one
        return model

    @classmethod
    def load(cls, filepath):
        folder = os.path.dirname(filepath)
        # If we pass a .pt file, load a specific checkpoint
        if filepath.endswith(".pt"):
            modelname = os.path.basename(filepath)
            match = re.match(r"model@(?P<global_step>\d+).pt", modelname)
            if match is None:
                raise ValueError("Could not extract global_step from modelname")
            global_step = int(match.group("global_step"))

            configname = "%s.yaml" % folder
            config = OmegaConf.load(configname)
        # If we pass the config path, just load the config.
        elif filepath.endswith(".yaml"):
            config = OmegaConf.load(filepath)
            global_step = 0
        else:
            raise ValueError("Invalid checkpoint/config file: %s" % filepath)
        return cls(folder=folder, config=config, global_step=global_step)
