from huggingface_hub import hf_hub_download
import os

REPO_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
FILENAME = "pytorch_model.bin"

PATH = "./server/models/wespeaker-resnet-34/1/"
file = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=PATH,
    force_download=True,
    local_dir_use_symlinks=False,
)

os.rename(file, f"{PATH}/wespeaker-voxceleb-resnet34-LM.bin")
