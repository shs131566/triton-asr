# Triton ASR
Triton backend for ASR(Automatic Speech Recognition). The goal of Triton ASR is to let you serve ASR models written in Python by Triton Inference Server. The ASR models includes tasks such as recognition, speaker diarization, and sentence segmentation.

## Recognition: Whisper
- [large-v2](https://github.com/m-bain/whisperX)
- [large-v3](https://github.com/openai/whisper)

## Speaker diarization
- [embedding](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/tree/main)

- clustering: AgglomerativeClustering
# Quick start
1. `pip install -r requirements.txt`

2. download [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) for embedding model.

```bash
python model_download.py
```
3. Run docker compose
```bash
docker compose --profile inference up --build -d
```

4. Test infernence server
In server/inference_test.py change your sample auido file path at line 9
``` python
url = "localhost:8000"
audio_path = "SAMPLE_AUDIO_FILE" # YOUR AUDIO FILE
model_name = "whisper-large"
```
``` bash
python server/inference_test.py
```

