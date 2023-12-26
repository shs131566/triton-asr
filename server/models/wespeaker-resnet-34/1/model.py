import json
from typing import List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from cluster import AgglomerativeClustering
from loguru import logger
from wespeaker_resnet import WeSpeakerResNet34


class TritonPythonModel:
    def initialize(self, args):
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]

        logger.info(f"Using device {device}:{device_id}")
        self.model = WeSpeakerResNet34.load_from_checkpoint(
            "/models/wespeaker-resnet-34/1/wespeaker-voxceleb-resnet34-LM.bin",
            map_location=f"{device}:{device_id}"
            if args["model_instance_kind"] == "GPU"
            else "cpu",
            strict=False,
        )
        self.model.eval()
        self.model.to(device)

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                inp = pb_utils.get_input_tensor_by_name(request, "audio")
                input_audio = inp.as_numpy()
                input_audio = input_audio.reshape(-1)

                sr = pb_utils.get_input_tensor_by_name(
                    request, "sample_rate"
                ).as_numpy()[0]

                transcripts = pb_utils.get_input_tensor_by_name(
                    request, "transcription"
                ).as_numpy()[0]

                embeddings = []
                transcripts = json.loads(transcripts)

                for transcript in transcripts["segments"]:
                    start, end = transcript["start"], transcript["end"]
                    audio_segment = input_audio[int(start * sr) : int(end * sr)]
                    audio_segment = torch.Tensor(audio_segment).reshape(1, 1, -1)
                    audio_segment = audio_segment.to(self.model.device)
                    embedding = self.model(audio_segment)
                    embeddings.append(embedding.detach().cpu().numpy())

                logger.info(f"Embeddings: {embeddings}")
                if len(embeddings) == 1:
                    transcripts["segments"][0]["speaker"] = 0
                else:
                    cluster_model = AgglomerativeClustering()
                    clusters = cluster_model.cluster(np.vstack(embeddings))

                    for i, transcript in enumerate(transcripts["segments"]):
                        transcript["speaker"] = int(clusters[i])

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "transcription",
                            np.array(
                                [json.dumps(transcripts)],
                                dtype=np.string_,
                            ),
                        ),
                        pb_utils.Tensor(
                            "embeddings",
                            np.vstack(embeddings),
                        ),
                    ]
                )
                responses.append(inference_response)
            except Exception as e:
                logger.error(f"Failed to execute request: {e}")
        return responses
