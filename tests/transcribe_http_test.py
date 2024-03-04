import json

import librosa
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

url = "localhost:8123"
audio_path = "/home/shs1566/triton-asr/1.wav"
model_name = "whisper-large"

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

model_name = "wespeaker-resnet-34"

transcript_input = InferInput("transcription", [1], "BYTES")
transcript_input.set_data_from_numpy(transcripts)

result = triton_client.infer(
    model_name=model_name,
    inputs=[audio_input, sr_input, transcript_input],
    timeout=36000,
)

embeddings = result.as_numpy("embeddings") 

transcripts = result.as_numpy("transcription")
transcripts = json.loads(transcripts[0])

for transcript in transcripts["segments"]:
    print(f"화자{transcript['speaker']}: {transcript['text']}")
