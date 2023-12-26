import json
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
import whisper_timestamped as whisperx
from loguru import logger


class TritonPythonModel:
    def initialize(self, args):
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]

        logger.info(f"Using device {device}:{device_id}")
        self.model = whisperx.load_model(
            "large-v2",
            device="cuda",
            device_index=int(device_id),
            compute_type="float16",
            language="ko",
        )

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                inp = pb_utils.get_input_tensor_by_name(request, "audio")
                input_audio = inp.as_numpy()
                logger.info(f"input_audio_shape: {input_audio.shape}")
                outputs = self.model.transcribe(input_audio)
                logger.critical("end")
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "transcription",
                            np.fromstring(
                                json.dumps(outputs, ensure_ascii=False),
                                dtype=np.uint8,
                            ),
                        )
                    ]
                )

                responses.append(inference_response)
            except Exception as e:
                logger.error(f"Failed to execute request: {e}")
        return responses
