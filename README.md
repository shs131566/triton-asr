# Triton ASR
Triton backend for ASR(Automatic Speech Recognition). The goal of Triton ASR is to let you serve ASR models written in Python by Triton Inference Server. The ASR models includes tasks such as recognition, speaker separation, and sentence segmentation.

# Quick start
1. download [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) for embedding model.

```bash
python model_download.py
```
2. Run docker compose
```bash
docker compose --profile inference up --build -d
```

3. Test infernence server
In server/inference_test.py change your sample auido file path at line 9
``` python
url = "localhost:8000"
audio_path = "SAMPLE_AUDIO_FILE" # YOUR AUDIO FILE
model_name = "whisper-large"
```
``` bash
python server/inference_test.py
```

