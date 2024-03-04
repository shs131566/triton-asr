import librosa
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput
import json

audio_path = "/home/shs1566/triton-asr/sample.wav"
grpc_server_url = "localhost:8124"
model_name = "whisper-large"


triton_client = InferenceServerClient(
    url=grpc_server_url,
)
audio, sr = librosa.load(audio_path, sr=16000)
audio = audio.reshape(1, -1)

audio_input = InferInput(name="audio", shape=audio.shape, datatype="FP32")
sr_input = InferInput(name="sample_rate", shape=[1], datatype="INT32")

audio_input.set_data_from_numpy(audio)
sr_input.set_data_from_numpy(np.array([sr], dtype=np.int32))

result = triton_client.infer(
    model_name=model_name,
    inputs=[audio_input, sr_input],
    timeout=36000,
)

transcripts = result.as_numpy("transcription")
# Get the output arrays from the results

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