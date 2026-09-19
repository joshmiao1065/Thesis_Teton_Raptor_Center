"""ConvNeXT-Base-BirdSet-XCL: model loading and batched preprocessing.

The preprocessing reproduces the reference pipeline used for the baseline (32 kHz, n_fft 1024,
hop 320, power 2, 128 mel bins, dB with top_db 80 relative to each window's own maximum,
normalize with mean -4.268 / std 4.569), but works on a batch of windows at once.
"""

import torch
import torchaudio

MODEL_ID = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
SAMPLE_RATE = 32_000
WINDOW_S = 5
TARGETS = {  # eBird-style codes used by the model's label set
    "flaowl": "flammulated owl",
    "grgowl": "great gray owl",
    "norgos": "northern goshawk",
    "brdowl": "barred owl",
    "borowl": "boreal owl",
}


class Preprocessor(torch.nn.Module):
    """(B, samples) float waveform at 32 kHz -> (B, 1, 128, frames) normalized mel-dB."""

    def __init__(self, amin=1e-10, top_db=80.0, mean=-4.268, std=4.569):
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=320, power=2.0)
        self.mel = torchaudio.transforms.MelScale(n_mels=128, n_stft=513, sample_rate=SAMPLE_RATE)
        self.amin, self.top_db, self.mean, self.std = amin, top_db, mean, std

    @torch.no_grad()
    def forward(self, wave):
        mel = self.mel(self.spec(wave))
        db = 10.0 * torch.log10(torch.clamp(mel, min=self.amin))  # ref = 1.0
        floor = db.amax(dim=(-2, -1), keepdim=True) - self.top_db  # per window
        db = torch.maximum(db, floor)
        return ((db - self.mean) / self.std).unsqueeze(1)


def load_model(device="cuda"):
    from transformers import ConvNextForImageClassification

    model = ConvNextForImageClassification.from_pretrained(MODEL_ID, ignore_mismatched_sizes=True)
    return model.to(device).eval()


def target_ids(model):
    return [model.config.label2id[code] for code in TARGETS]
