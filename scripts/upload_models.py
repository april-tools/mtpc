import os

from huggingface_hub import HfApi
from pathlib import Path
from collections import defaultdict

# hf_token = os.environ["HF_TOKEN"]
root_folder = os.environ["MTP_ROOT"]

api = HfApi()
username = api.whoami()['name']

# api.create_repo(repo_id=f"{username}/test", repo_type='model', private=True, exist_ok=True)


models_dir = Path(root_folder, "outputs", "models")
model_ids = defaultdict(list)
for arch_folder in models_dir.iterdir():
    print(f"Processing {arch_folder}...")
    for group_folder in arch_folder.iterdir():
        print(f"Processing {group_folder}...")
        for model_folder in group_folder.iterdir():
            print(f"   Processing model {model_folder.name} of {group_folder.name}...")
            if model_folder.is_dir():
                try:
                    model_folder_name = model_folder.name.replace(f"{arch_folder.name}-", "")
                    repo_id = f"{username}/{arch_folder.name}-{model_folder_name}"
                    model_ids[group_folder.name].append(repo_id)

                    # Create private repo
                    api.create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        private=False,
                        exist_ok=False,
                    )

                    api.upload_folder(
                        folder_path=str(model_folder),
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    print(f"✓ Uploaded {arch_folder.name}-{model_folder_name}")
                except Exception as e:
                    print(f"✗ Failed {model_folder_name}: {e}")

for arch_folder in models_dir.iterdir():
    arch_cased = arch_folder.name.capitalize()
    for group_folder in arch_folder.iterdir():
        try:
            # Try to create a collection
            collection = api.create_collection(
                title=f"{arch_cased}-MTPC-{group_folder.name}",
                namespace=username,  # or org name
                private=True,  # optional
            )
        except Exception:
            # If collection already exists, try to get it
            try:
                collection_slug = f"{arch_cased}-MTPC-{group_folder.name}".lower().replace(" ", "-")
                collection = api.get_collection(collection_slug=collection_slug, namespace=username)
                print(f"✓ Using existing collection {group_folder.name}")
            except Exception as e2:
                print(f"✗ Failed creating/fetching collection {group_folder.name}: {e2}")
                continue

        # Add models to the collection
        for model_id in model_ids[group_folder.name]:
            try:
                api.add_collection_item(
                    collection_slug=collection.slug,
                    item_id=model_id,
                    item_type="model",
                )
            except Exception as e:
                print(f"✗ Failed adding model {model_id} to collection {group_folder.name}: {e}")
