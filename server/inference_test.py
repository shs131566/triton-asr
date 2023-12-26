import json

import librosa
import numpy as np
import tritonhttpclient
from tritonclient.http import InferenceServerClient

url = "localhost:8000"
audio_path = "SAMPLE_AUDIO_FILE"
model_name = "whisper-large"

triton_client = InferenceServerClient(url=url, network_timeout=3600)

audio, sr = librosa.load(audio_path, sr=16000)
audio = audio.reshape(1, -1)

audio_input = tritonhttpclient.InferInput(
    name="audio", shape=audio.shape, datatype="FP32"
)
sr_input = tritonhttpclient.InferInput(name="sample_rate", shape=[1], datatype="INT32")

audio_input.set_data_from_numpy(audio)
sr_input.set_data_from_numpy(np.array([sr], dtype=np.int32))

result = triton_client.infer(
    model_name=model_name,
    inputs=[audio_input, sr_input],
    timeout=360000,
)

transcripts = result.as_numpy("transcription")

print(transcripts.shape)
model_name = "wespeaker-resnet-34"

transcript_input = tritonhttpclient.InferInput("transcription", [1], "BYTES")
transcript_input.set_data_from_numpy(transcripts)

result = triton_client.infer(
    model_name=model_name,
    inputs=[audio_input, sr_input, transcript_input],
    timeout=36000,
)

embeddings = result.as_numpy("embeddings")
print(embeddings.shape)

transcripts = result.as_numpy("transcription")
transcripts = json.loads(transcripts[0])

for transcript in transcripts["segments"]:
    print(f"참석자{transcript['speaker']}: {transcript['text']}")
