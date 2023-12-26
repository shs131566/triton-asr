import json
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
import whisper_timestamped as whisper
from loguru import logger


class TritonPythonModel:
    def initialize(self, args):
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]

        logger.info(f"Using device {device}:{device_id}")
        self.model = whisper.load_model(
            "large-v3",
            device=f"{device}:{device_id}",
        )

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                inp = pb_utils.get_input_tensor_by_name(request, "audio")
                input_audio = inp.as_numpy()

                logger.info(f"input_audio.shape: {input_audio.shape}")
                outputs = whisper.transcribe(
                    self.model,
                    input_audio.reshape(-1),
                    language="ko",
                    vad="auditok",
                    min_word_duration=0.1,
                    beam_size=5,
                    remove_punctuation_from_words=True,
                    compute_word_confidence=False,
                    remove_empty_words=True,
                )

                logger.info(f"type(outputs): {type(outputs)}")
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "transcription",
                            np.array(
                                [json.dumps(outputs)],
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
