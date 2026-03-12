from dataclasses import dataclass, field
from sortedcontainers import SortedDict

from .config import MAX_CLIP_FRAME_SEPARATION


class PersonResults:
    def __init__(self, id):
        self.id = id
        self.clips = []

    @dataclass
    class Clip:
        start: float
        end: float
        boxes: SortedDict = field(default_factory=SortedDict)
        max_box_size: dict = field(default_factory=lambda: {"x": 0, "y": 0})

    def add_bounding_box_frame(self, timestamp, bounding_box):
        x_size = bounding_box[2] - bounding_box[0]
        y_size = bounding_box[3] - bounding_box[1]
        if len(self.clips) > 0 and self.clips[-1].end + MAX_CLIP_FRAME_SEPARATION > timestamp:
            self.clips[-1].boxes[timestamp] = bounding_box
            self.clips[-1].end = timestamp
            self.clips[-1].max_box_size["x"] = max(self.clips[-1].max_box_size["x"], x_size)
            self.clips[-1].max_box_size["y"] = max(self.clips[-1].max_box_size["y"], y_size)
        else:
            self.clips.append(PersonResults.Clip(
                start=timestamp,
                end=timestamp,
                boxes=SortedDict({timestamp: bounding_box}),
                max_box_size={"x": x_size, "y": y_size},
            ))
