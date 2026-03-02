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

    def _init_gpu_decoder(self):
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-hwaccel", "cuda",
            "-i", self.video_path,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8
            )
            self._ff_frame_bytes = self.width * self.height * 3
            self.use_gpu = True
        except Exception:
            self._init_pav_decoder()

    def _init_pav_decoder(self):
        self.use_gpu = False
        self.container = av.open(self.video_path)
        stream = self.container.streams.video[0]
        stream.thread_type = "AUTO"
        self._frame_iter = self.container.decode(stream)

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
        if getattr(self, "container", None):
            try:
                self.container.close()
            except Exception:
                pass


def read_video_for_clips(path, clips, sample_rate=6, use_gpu=True):
    """Open a video and yield (clip, frame_rgb, timestamp_s) for every sampled
    frame across all clips in a single sequential pass.

    A fresh reader is created per call so each call starts from frame 0.
    Clips are processed in temporal order; frames between clips are discarded.
    """
    reader = GPUVideoReader(path, use_gpu=use_gpu)
    fps_original = reader.get_fps()
    skip_rate = max(1, int(round(fps_original / sample_rate)))

    sorted_clips = sorted(clips, key=lambda c: c.start)
    clip_frame_ranges = [
        (int(c.start * fps_original), int(c.end * fps_original), c)
        for c in sorted_clips
    ]

    current_clip_idx = 0
    frame_count = 0

    try:
        while current_clip_idx < len(clip_frame_ranges):
            ret, frame = reader.read()
            if not ret:
                break

            start_frame, end_frame, clip = clip_frame_ranges[current_clip_idx]

            if frame_count < start_frame:
                frame_count += 1
                continue

            if frame_count >= end_frame:
                current_clip_idx += 1
                frame_count += 1
                continue

            if frame_count % skip_rate == 0:
                yield clip, frame, frame_count / fps_original

            frame_count += 1
    finally:
        reader.release()
