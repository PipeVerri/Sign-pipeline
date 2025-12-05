# Using a transcription model, create spanish subtitles for the videos in raw, yt subtitles aren't good enough
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
from dotenv import load_dotenv
from pathlib import Path
import os

root_dir = Path(__file__).parent.parent.resolve()
load_dotenv(root_dir / ".env")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device=device)
model_a, metadata = whisperx.load_align_model(language_code="es", device=device)
diarize_model = DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)

files = ["../data/CN Sordos/cn.mp4"]
for file in files:
    audio = whisperx.load_audio(file)
    result = model.transcribe(audio)
    print("transcribed, ", end="")
    # Forced alignment to improve timestamps
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print("aligned", end="")
    # Diarization for multiple speakers
    diarized_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarized_segments, aligned)
    print("done")

    print(result)