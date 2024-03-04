import json
from typing import List
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
import whisper
from loguru import logger


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        self.device_id = args["model_instance_device_id"]

        logger.info(f"Using device {self.device}:{self.device_id}")
        self.model = whisper.load_model(
            "large-v3",
            device=f"{self.device}:{self.device_id}",
        )
        self.options = whisper.DecodingOptions(language="ko", without_timestamps=True)
        
    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                inp = pb_utils.get_input_tensor_by_name(request, "audio")
                input_audio = inp.as_numpy()

                logger.info(f"input_audio.shape: {input_audio.shape}")
                audio_tensor = torch.tensor(whisper.pad_or_trim(input_audio.flatten())).to(f"{self.device}:{self.device_id}")
                mel = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)
                outputs = self.model.decode(mel, self.options)
                if outputs.compression_ratio >= 2.0:
                    return None
                
                logger.info(f"type(outputs): {type(outputs.text)}")
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "transcription",
                            np.array(
                                [json.dumps(outputs.text)],
                                dtype=np.string_,
                            ),
                        ),
                    ]
                )
                logger.info(f"inference_response: {inference_response}")
                responses.append(inference_response)
            except Exception as e:
                logger.error(f"Failed to execute request: {e}")
        return responses
