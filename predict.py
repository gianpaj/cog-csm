from cog import BasePredictor, Input, Path, File
import os
import time
import subprocess
import torchaudio
from generator import load_csm_1b

MODEL_CACHE = "csm-1b"
LLAMA_CACHE = "Llama-3.2-1B"
CSM_URL = "https://weights.replicate.delivery/default/sesame/csm-1b/model.tar"
LLAMA_URL = (
    "https://weights.replicate.delivery/default/meta-llama/Llama-3.2-1B/model.tar"
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(CSM_URL, MODEL_CACHE)
        # Download the llama weights if they don't exist
        if not os.path.exists(LLAMA_CACHE):
            download_weights(LLAMA_URL, LLAMA_CACHE)
        # Load the model
        self.generator = load_csm_1b(device="cuda")

    def predict(
        self,
        text: str = Input(
            description="Text to convert to speech", default="Hello from Sesame."
        ),
        speaker: int = Input(
            description="Speaker ID (0 or 1)", default=0, choices=[0, 1]
        ),
        max_audio_length_ms: int = Input(
            description="Maximum audio length in milliseconds",
            default=10000,
            ge=1000,
            le=30000,
        ),
        context_text: str = Input(
            description="Trascript of the audio file to clone the voice from",
            default="",
        ),
        context_audio: File = Input(
            description="Audio file to clone the voice from",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
        )

        output_path = Path("/tmp/output.wav")
        torchaudio.save(
            output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate
        )

        return output_path
