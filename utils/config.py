from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DownloadConfig:
    format: str = "(bestvideo[vcodec^=avc][height<=1080]/bestvideo[vcodec^=avc])+bestaudio[ext=m4a]/best"
    sub_langs: list[str] = field(default_factory=lambda: ["es.*"])
    cookies_from_browser: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> DownloadConfig:
        return cls(
            format=d.get("format", cls.__dataclass_fields__["format"].default),
            sub_langs=d.get("sub_langs", ["es.*"]),
            cookies_from_browser=d.get("cookies_from_browser"),
        )


@dataclass
class VideoAudioSeparationConfig:
    delete_original: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> VideoAudioSeparationConfig:
        return cls(delete_original=d.get("delete_original", False))


@dataclass
class WhisperConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    language: str = "es"

    @classmethod
    def from_dict(cls, d: dict) -> WhisperConfig:
        return cls(
            model=d.get("model", "large-v3-turbo"),
            device=d.get("device", "cuda"),
            language=d.get("language", "es"),
        )


@dataclass
class BoundingBoxesConfig:
    model_path: str = "models/yolo11m.pt"
    fps: int = 6
    batch_size: int = 32
    batch_queue: int = 32

    @classmethod
    def from_dict(cls, d: dict) -> BoundingBoxesConfig:
        return cls(
            model_path=d.get("model_path", "models/yolo11m.pt"),
            fps=int(d.get("fps", 6)),
            batch_size=d.get("batch_size", 32),
            batch_queue=d.get("batch_queue", 32),
        )


@dataclass
class LandmarksConfig:
    fps: int = 6
    num_workers: int = 1
    max_clip_frame_separation: float = 1.0
    min_clip_duration_frames: int = 36   # fps * min_clip_seconds (6 * 6 = 36)
    moving_threshold: float = 0.25
    write_buffer_size: int = 160
    mode: str = "balanced"      # rtmlib mode: balanced, performance, lightweight
    backend: str = "onnxruntime"
    device: str = "cuda"

    @classmethod
    def from_dict(cls, d: dict) -> LandmarksConfig:
        return cls(
            fps=d.get("fps", 6),
            num_workers=d.get("num_workers", 1),
            max_clip_frame_separation=d.get("max_clip_frame_separation", 1.0),
            min_clip_duration_frames=d.get("min_clip_duration_frames", 36),
            moving_threshold=d.get("moving_threshold", 0.25),
            write_buffer_size=d.get("write_buffer_size", 160),
            mode=d.get("mode", "balanced"),
            backend=d.get("backend", "onnxruntime"),
            device=d.get("device", "cuda"),
        )


@dataclass
class Options:
    download: DownloadConfig = field(default_factory=DownloadConfig)
    video_audio_separation: VideoAudioSeparationConfig = field(default_factory=VideoAudioSeparationConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    bounding_boxes: BoundingBoxesConfig = field(default_factory=BoundingBoxesConfig)
    landmarks: LandmarksConfig = field(default_factory=LandmarksConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Options:
        return cls(
            download=DownloadConfig.from_dict(d.get("download", {})),
            video_audio_separation=VideoAudioSeparationConfig.from_dict(d.get("video_audio_separation", {})),
            whisper=WhisperConfig.from_dict(d.get("whisper", {})),
            bounding_boxes=BoundingBoxesConfig.from_dict(d.get("bounding_boxes", {})),
            landmarks=LandmarksConfig.from_dict(d.get("landmarks", {})),
        )


@dataclass
class Source:
    name: str
    url: Optional[str] = None
    path: Optional[str] = None
    subs: bool = False
    auto_subs: bool = False
    generate_subs: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Source:
        return cls(
            name=d["name"],
            url=d.get("url"),
            path=d.get("path"),
            subs=d.get("subs", False),
            auto_subs=d.get("auto_subs", False),
            generate_subs=d.get("generate_subs", False),
        )


@dataclass
class PipelineConfig:
    sources: list[Source]
    options: Options

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        return cls(
            sources=[Source.from_dict(s) for s in d.get("sources", [])],
            options=Options.from_dict(d.get("options", {})),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))
