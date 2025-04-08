from .hf_dataloader import HFDistributedDataLoader
from .sharegpt import ShareGPTDataLoader


HFDistributedDataLoader.register('Aeala/ShareGPT_Vicuna_unfiltered', ShareGPTDataLoader)
