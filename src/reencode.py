#!/usr/bin/env python3

import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import sys


# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def get_codec_info(file_path):
    """Get video and audio codec information using ffprobe."""
    try:
        # Get video codec
        video_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'json',
            str(file_path)
        ]
        video_result = subprocess.run(video_cmd, capture_output=True, text=True)
        video_data = json.loads(video_result.stdout)
        video_codec = video_data.get('streams', [{}])[0].get('codec_name', 'unknown')

        # Get audio codec
        audio_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'json',
            str(file_path)
        ]
        audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
        audio_data = json.loads(audio_result.stdout)
        audio_codec = audio_data.get('streams', [{}])[0].get('codec_name', 'unknown')

        return video_codec, audio_codec
    except Exception as e:
        return 'unknown', 'unknown'


def should_skip_file(file_path):
    """Check if file is already properly encoded."""
    video_codec, audio_codec = get_codec_info(file_path)

    # Skip if already h264/aac and is mp4
    if (video_codec == 'h264' and
            audio_codec == 'aac' and
            file_path.suffix.lower() == '.mp4'):
        return True, video_codec, audio_codec

    return False, video_codec, audio_codec


def reencode_file(file_path, backup_dir, worker_id):
    """Re-encode a single video file."""
    filename = file_path.name
    basename_no_ext = file_path.stem

    # Paths
    temp_file = file_path.parent / f"{basename_no_ext}_temp.mp4"
    backup_file = backup_dir / filename
    final_file = file_path.parent / f"{basename_no_ext}.mp4"

    result = {
        'file': filename,
        'status': 'processing',
        'worker_id': worker_id
    }

    print(f"{Colors.BLUE}[Worker {worker_id}]{Colors.NC} Processing: {filename}")

    # Check if already properly encoded
    should_skip, video_codec, audio_codec = should_skip_file(file_path)

    if should_skip:
        print(f"{Colors.GREEN}[Worker {worker_id}] ⏭{Colors.NC}  Already encoded: {filename}")
        result['status'] = 'skipped'
        result['video_codec'] = video_codec
        result['audio_codec'] = audio_codec
        return result

    print(f"{Colors.YELLOW}[Worker {worker_id}]{Colors.NC} Current: {video_codec}/{audio_codec} -> h264/aac")

    # FFmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(file_path),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-ar', '16000',
        '-ac', '1',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-vsync', 'cfr',
        '-y',
        '-loglevel', 'error',
        '-stats',
        str(temp_file)
    ]

    try:
        # Run FFmpeg
        process = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if process.returncode != 0:
            print(f"{Colors.RED}[Worker {worker_id}] ❌{Colors.NC} FFmpeg failed: {filename}")
            print(f"   Error: {process.stderr[:200]}")
            result['status'] = 'failed'
            result['error'] = process.stderr[:200]
            temp_file.unlink(missing_ok=True)
            return result

        # Verify output file
        if not temp_file.exists() or temp_file.stat().st_size < 1000:
            print(f"{Colors.RED}[Worker {worker_id}] ❌{Colors.NC} Output invalid: {filename}")
            result['status'] = 'failed'
            result['error'] = 'Output file too small or missing'
            temp_file.unlink(missing_ok=True)
            return result

        # Backup original
        file_path.rename(backup_file)

        # Move temp to final location
        temp_file.rename(final_file)

        print(f"{Colors.GREEN}[Worker {worker_id}] ✓{Colors.NC} Success: {basename_no_ext}.mp4")
        result['status'] = 'success'
        result['output_file'] = final_file.name

        return result

    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}[Worker {worker_id}] ❌{Colors.NC} Timeout: {filename}")
        result['status'] = 'failed'
        result['error'] = 'Timeout (>1 hour)'
        temp_file.unlink(missing_ok=True)
        return result

    except Exception as e:
        print(f"{Colors.RED}[Worker {worker_id}] ❌{Colors.NC} Exception: {filename}")
        print(f"   Error: {str(e)}")
        result['status'] = 'failed'
        result['error'] = str(e)
        temp_file.unlink(missing_ok=True)
        return result


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    else:
        input_dir = Path('.')

    # Number of concurrent workers (default: CPU count - 1)
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, subprocess.os.cpu_count() // (3 / 2))

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Concurrent Video Re-encoding Script")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Directory: {input_dir.resolve()}")
    print(f"Workers: {max_workers}")
    print()

    # Setup directories
    backup_dir = input_dir / '.originals_backup'
    backup_dir.mkdir(exist_ok=True)

    failed_log = input_dir / 'failed_reencoding.txt'
    failed_log.unlink(missing_ok=True)

    # Find all video files
    video_extensions = ['*.mp4', '*.mkv', '*.webm', '*.avi', '*.m4v', '*.flv', '*.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(ext))

    if not video_files:
        print(f"{Colors.YELLOW}No video files found!{Colors.NC}")
        return

    print(f"Found {len(video_files)} video files")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Counters
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }

    start_time = datetime.now()

    # Process files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(reencode_file, file, backup_dir, i % max_workers + 1): file
            for i, file in enumerate(video_files)
        }

        # Process completed tasks
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                status = result['status']

                if status == 'success':
                    results['success'].append(result)
                elif status == 'skipped':
                    results['skipped'].append(result)
                else:
                    results['failed'].append(result)
                    # Log failed files
                    with open(failed_log, 'a', encoding='utf-8') as f:
                        f.write(f"{result['file']}\n")
                        if 'error' in result:
                            f.write(f"  Error: {result['error']}\n")

            except Exception as e:
                print(f"{Colors.RED}❌{Colors.NC} Unexpected error processing {file.name}: {e}")
                results['failed'].append({'file': file.name, 'error': str(e)})

    end_time = datetime.now()
    duration = end_time - start_time

    # Print summary
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Re-encoding Complete!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Total files:    {len(video_files)}")
    print(f"{Colors.GREEN}Success:        {len(results['success'])}{Colors.NC}")
    print(f"{Colors.YELLOW}Skipped:        {len(results['skipped'])}{Colors.NC}")
    print(f"{Colors.RED}Failed:         {len(results['failed'])}{Colors.NC}")
    print(f"\nTime elapsed:   {duration}")

    if results['success']:
        print(f"\nOriginal files backed up in: {backup_dir}")
        print(f"To delete backups: rm -rf {backup_dir}")

    if results['failed']:
        print(f"\n{Colors.RED}⚠{Colors.NC}  Failed files logged in: {failed_log}")


if __name__ == '__main__':
    main()