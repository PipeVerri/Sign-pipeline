import subprocess
import numpy as np
import av


class GPUVideoReader:
    """Video reader using NVDEC (ffmpeg -hwaccel cuda) with PyAV fallback.

    Probes video properties with PyAV; decodes via ffmpeg subprocess for GPU
    or PyAV for CPU.
    """

    def __init__(self, video_path, use_gpu=True):
        self.video_path = str(video_path)
        self.current_frame = 0
        self._probe_properties()
        self.use_gpu_requested = use_gpu
        if use_gpu:
            self._init_gpu_decoder()
        else:
            self._init_pav_decoder()

    def _probe_properties(self):
        container = av.open(self.video_path)
        stream = container.streams.video[0]
        self.fps = float(stream.average_rate)
        self.total_frames = stream.frames or -1
        self.width = stream.width
        self.height = stream.height
        container.close()

    def _init_gpu_decoder(self, start_frame=0):
        start_time = start_frame / self.fps
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-ss", f"{start_time:.3f}",
            "-i", self.video_path,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-",
            "-hwaccel", "cuda",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8
            )
            self._ff_frame_bytes = self.width * self.height * 3
            self.use_gpu = True
            self.current_frame = start_frame
        except Exception:
            self._init_pav_decoder()

    def _init_pav_decoder(self, start_frame=0):
        self.use_gpu = False
        self.container = av.open(self.video_path)
        stream = self.container.streams.video[0]
        stream.thread_type = "AUTO"
        
        if start_frame > 0:
            target_ts = int(start_frame * av.time_base / self.fps)
            self.container.seek(target_ts, stream=stream)
        
        self._frame_iter = self.container.decode(stream)
        self.current_frame = start_frame

    def seek(self, frame_index):
        """Seek to a specific frame index."""
        if frame_index == self.current_frame:
            return
            
        if self.use_gpu:
            self.release()
            self._init_gpu_decoder(start_frame=frame_index)
        else:
            self.release()
            self._init_pav_decoder(start_frame=frame_index)

    def read(self):
        if self.use_gpu:
            raw = self.proc.stdout.read(self._ff_frame_bytes)
            if not raw or len(raw) < self._ff_frame_bytes:
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            self.current_frame += 1
            return True, frame
        else:
            try:
                frame = next(self._frame_iter)
                self.current_frame += 1
                return True, frame.to_ndarray(format="rgb24")
            except StopIteration:
                return False, None

    def get_fps(self):
        return self.fps

    def release(self):
        if getattr(self, "proc", None):
            try:
                self.proc.kill()
                self.proc.stdout.close()
                self.proc.stderr.close()
            except Exception:
                pass
            self.proc = None
        if getattr(self, "container", None):
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None


def read_video_for_clips(path, clips, sample_rate=6, use_gpu=True, seek_threshold=30):
    """Open a video and yield (clip, frame_rgb, timestamp_s) for every sampled
    frame across all clips in a single sequential pass.
    
    If multiple clips overlap at the same frame, it yields (clip, frame, ts)
    once for each clip to maintain compatibility with existing loops.
    """
    reader = GPUVideoReader(path, use_gpu=use_gpu)
    fps_original = reader.get_fps()
    skip_rate = max(1, int(round(fps_original / sample_rate)))

    # Each clip is (start_frame, end_frame, clip_object)
    clip_ranges = sorted([
        (int(c.start * fps_original), int(c.end * fps_original), c)
        for c in clips
    ], key=lambda x: x[0])

    try:
        current_frame = 0
        while clip_ranges:
            # Find the next frame that belongs to any clip
            next_start = clip_ranges[0][0]
            
            # If we are far from the next start, seek
            if next_start - current_frame > seek_threshold:
                reader.seek(next_start)
                current_frame = next_start

            # Read frame
            ret, frame = reader.read()
            if not ret:
                break
            
            ts = current_frame / fps_original
            
            # Check which clips this frame belongs to
            # Since clips are sorted by start, and we might have overlapping clips,
            # we check all clips that could contain this frame.
            active_clips_indices = []
            for i, (start, end, clip) in enumerate(clip_ranges):
                if current_frame >= start and current_frame < end:
                    if current_frame % skip_rate == 0:
                        yield clip, frame, ts
                elif current_frame >= end:
                    active_clips_indices.append(i)
                elif current_frame < start:
                    # Since sorted by start, no more clips will match this frame
                    break
            
            # Remove clips that have ended
            for i in reversed(active_clips_indices):
                clip_ranges.pop(i)
                
            current_frame += 1
            
    finally:
        reader.release()
