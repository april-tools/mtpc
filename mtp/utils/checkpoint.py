import os
import re
import hydra
import torch

from omegaconf import OmegaConf

from mtp.utils.distributed import get_local_device


def maskcwd(func):
    # https://github.com/omry/omegaconf/blob/117f7de07285e4d1324b9229eaf873de15279457/omegaconf/omegaconf.py#L184
    # Omegaconf uses abspath everywhere, which when combined with hydra's
    # assumption about cwd can be a pain.
    # Since we want relative paths to work from cli from MTP_ROOT
    # change dir and restore, use abs paths if calling checkpoint from train.py
    def wrapper(*args, **kwargs):
        cwd = os.getcwd()
        os.chdir(os.environ['MTP_ROOT'])
        result = func(*args, **kwargs)
        os.chdir(cwd)
        return result
    return wrapper


class Checkpoint(object):
    """In the checkpoint folder we keep:
    1. a yaml config with the model spec
    2. a model@steps.pt file with the state dict of the model,
    optimizer, scheduler, etc."""

    def __init__(self, folder, config, global_step=0):
        super().__init__()
        self.folder = folder
        self.config = config
        assert global_step >= 0
        self.global_step = global_step
        self.configpath = os.path.join(self.folder, "config.yaml")
        self.expname = self.config.expname

    def __repr__(self):
        return "%s@%s" % (self.expname, self.global_step)

    def _load_state(self, device=None):
        if device is None:
            device = get_local_device()
        # load state dict from the saved file
        # and load state dict for the items passed in
        state = torch.load(
            self.modelpath,
            weights_only=True,
            map_location=device
        )
        return state

    def _load_model_state_dict(self, model, saved_model_state_dict):
        # We want to deal with the situation of missing keys
        # E.g. if we are loading a LM from a checkpoint we do not need to save
        # the params in the state dict, because we can load the checkpoint
        ok_if_missing = {k for k, v in model.state_dict().items()
                         if v is None}
        # Drop keys that are None - None means ok if missing
        model_state = {k: v for k, v in saved_model_state_dict.items()
                       if k not in ok_if_missing}
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if len(unexpected_keys) > 1:
            raise ValueError('Found unexpected keys when loading: %s' % unexpected_keys)
        if len(missing_keys) > 1:
            if not set(missing_keys).issubset(set(ok_if_missing)):
                unexpected = set(ok_if_missing).difference(set(missing_keys))
                raise ValueError('Found unexpected missing keys when loading: %s' % unexpected)

    @maskcwd
    def save(self, global_step=0, model=None, optimizer=None, scheduler=None):
        assert global_step >= 0
        # Advance global_step
        self.global_step = global_step
        if global_step == 0 and model is None:
            # We haven't begun training, just serialise the config file
            with open(self.configpath, "w") as f:
                OmegaConf.save(self.config, f)
        else:
            assert model is not None
            model_state_dict = model.state_dict()
            optimizer_state_dict = (
                None if optimizer is None else optimizer.state_dict()
            )
            scheduler_state_dict = (
                None if scheduler is None else scheduler.state_dict()
            )

            state = {
                "global_step": global_step,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "scheduler_state_dict": scheduler_state_dict,
            }
            torch.save(state, self.modelpath)

    def restore(self, model, optimizer=None, scheduler=None, device=None):
        # NOTE: modifies inplace
        # Load state_dict from checkpoint and apply to the objects
        state = self._load_state(device=device)

        # Deal with cases where we only serialize subset of params
        self._load_model_state_dict(model, state['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])

    @property
    def modelpath(self):
        modelpath = os.path.join(
            self.folder, "model@%d.pt" % self.global_step
        )
        return modelpath

    @property
    def model(self):
        model = hydra.utils.instantiate(self.config.model).model

        device = get_local_device()
        model = model.to(device)
        # If we have begun training, you are getting the saved model
        if self.global_step >= 0:
            state = self._load_state(device=device)
            # Deal with cases where we only serialize subset of params
            self._load_model_state_dict(model, state['model_state_dict'])
        # Otherwise, you get a randomly initialised one
        return model

    @classmethod
    @maskcwd
    def load(cls, filepath, overrides = None):
        # Omegaconf is a pain to use without an absolute path
        # so just bite the bullet and use the same for pytorch
        folder = os.path.abspath(os.path.dirname(filepath))
        # If we pass a .pt file, load a specific checkpoint
        if filepath.endswith(".pt"):
            modelname = os.path.basename(filepath)
            match = re.match(r"model@(?P<global_step>\d+).pt", modelname)
            if match is None:
                raise ValueError(
                    "Could not extract global_step from modelname"
                )
            global_step = int(match.group("global_step"))

            configname = os.path.join(folder, "config.yaml")
            config = OmegaConf.load(configname)
        # If we pass the config path, just load the config.
        elif filepath.endswith(".yaml"):
            config = OmegaConf.load(filepath)
            global_step = 0
        else:
            raise ValueError("Invalid checkpoint/config file: %s" % filepath)
        if overrides is not None:
            for key, value in overrides.items():
                OmegaConf.update(config, key, value)
        return cls(folder=folder, config=config, global_step=global_step)
