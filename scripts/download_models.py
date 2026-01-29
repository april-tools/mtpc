import os
from huggingface_hub import get_collection, snapshot_download

for slug in ["agrv/evabyte-mtpc-lora-continued", "agrv/evabyte-mtpc-no-lora", "agrv/llama-mtpc-lora-continued", "agrv/llama-mtpc-no-lora"]:
    collection = get_collection(slug)
    model, _, subset = slug.split('-', 2)
    _, model = model.split('/')
    for item in collection.items:
        if item.item_type == "model":
            foldername = item.item_id.replace("agrv/", "")
            snapshot_download(
                repo_id=item.item_id,
                allow_patterns=["model@0.pt", "model@900.pt", "config.yaml"],
                local_dir=f"{os.environ['MTP_ROOT']}/outputs/models/{model}/{subset}/{foldername}"
            )
