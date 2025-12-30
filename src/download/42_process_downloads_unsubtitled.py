from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

root_dir = Path(__file__).parent.parent.parent.resolve()

def delete_file(f):
    try:
        os.remove(f)
    except OSError:
        pass

def process_file(PATH, f):
    """Process a single file with VAD detection"""
    try:
        # Load model per thread to avoid threading issues
        model = load_silero_vad()
        wav = read_audio(PATH / "audio" / f)
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=False)
        if len(speech_timestamps) == 0:
            name = f.split(".")[0]
            os.rename(PATH / "video" / f"{name}.mp4", PATH / "unlabeled/video" / f"{name}.mp4")
            os.rename(PATH / "audio" / f"{name}.mp3", PATH / "unlabeled/audio" / f"{name}.mp3")
            delete_file(PATH / f"{name}.vtt")
            return f"Moved {name} to unlabeled"
        return f"Processed {f} - speech detected"
    except Exception as e:
        return f"Error processing {f}: {str(e)}"


def process_folder(PATH, max_workers=24):
    """Process all files in a folder in parallel using threads"""
    os.makedirs(PATH / "unlabeled/video", exist_ok=True)
    os.makedirs(PATH / "unlabeled/audio", exist_ok=True)
    files = os.listdir(PATH / "audio")

    # Process files in parallel using threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, PATH, f): f for f in files}

        for future in as_completed(futures):
            filename = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error with {filename}: {str(e)}")


if __name__ == "__main__":
    folders = ("0-AsociacionCivil", "1-videolibros_private", "2-videolibros_public")

    # Process folders sequentially (one folder at a time)
    # but files within each folder in parallel using threads
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        process_folder(root_dir / "data" / "raw" / folder)
        print(f"Completed folder: {folder}")