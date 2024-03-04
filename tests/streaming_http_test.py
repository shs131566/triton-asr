import json

import librosa
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

url = "localhost:8123"
audio_path = "/home/shs1566/triton-asr/1.wav"
model_name = "streaming"

triton_client = InferenceServerClient(url=url, network_timeout=3600)

audio, sr = librosa.load(audio_path, sr=16000)
audio = audio.reshape(1, -1)

audio_input = InferInput(name="audio", shape=audio.shape, datatype="FP32")
sr_input = InferInput(name="sample_rate", shape=[1], datatype="INT32")

audio_input.set_data_from_numpy(audio)
sr_input.set_data_from_numpy(np.array([sr], dtype=np.int32))


result = triton_client.infer(
    model_name=model_name,
    inputs=[audio_input, sr_input],
    timeout=360000,
)

transcripts = result.as_numpy("transcription")
print(json.loads(transcripts[0]))