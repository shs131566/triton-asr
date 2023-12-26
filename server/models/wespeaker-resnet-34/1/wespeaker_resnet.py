from functools import partial

import pytorch_lightning as pl
import torch
import torchaudio.compliance.kaldi as kaldi
from base_resnet import BasicBlock, ResNet


def ResNet34(feat_dim, embed_dim, pooling_func="TSTP", two_emb_layer=True):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
    )


class WeSpeakerResNet34(pl.LightningModule):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        frame_length: int = 25,
        frame_shift: int = 10,
        num_mel_bins: int = 80,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            "sample_rate",
            "num_channels",
            "num_mel_bins",
            "frame_length",
            "frame_shift",
            "dither",
            "window_type",
            "use_energy",
        )
        self.fbank = partial(
            kaldi.fbank,
            num_mel_bins=self.hparams.num_mel_bins,
            frame_length=self.hparams.frame_length,
            frame_shift=self.hparams.frame_shift,
            dither=self.hparams.dither,
            sample_frequency=self.hparams.sample_rate,
            window_type=self.hparams.window_type,
            use_energy=self.hparams.use_energy,
        )
        self.resnet = ResNet34(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )

    def compute_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = waveforms * (1 << 15)
        features = torch.vmap(self.fbank)(waveforms)

        return features - torch.mean(features, dim=1, keepdim=True)

    def forward(self, waveforms: torch.Tensor, weights: torch.Tensor = None):
        fbank = self.compute_fbank(waveforms)
        return self.resnet(fbank, weights=weights)[1]
