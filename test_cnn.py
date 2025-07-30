import os
from huggingface_hub import hf_hub_download, snapshot_download
from cnn import Model
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict


# Configuration
repo_id = "jonathansuru/cnn"
model_filename = "model.safetensors"

model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
)
print(f"Model downloaded to: {model_path}")

model = Model()
state_dict = safe_load(model_path)
load_state_dict(model, state_dict)