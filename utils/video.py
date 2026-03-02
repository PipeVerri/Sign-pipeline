import av

def frames_for_segment(path, start_s=0, end_s=None, sample_rate=6):
    """
    Yield (ndarray_bgr, timestamp_s) for frames in [start_s, end_s).
    Uses container.seek to jump to a keyframe near start_s and decodes forward.
    """
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # FFmpeg's timebase for seek is AV_TIME_BASE (microseconds)
    av_time_base = 1_000_000
    # seek to nearest keyframe <= start_s
    container.seek(int(start_s * av_time_base), any_frame=False, backward=True, stream=stream)

    # how often to yield frames relative to decoded frames
    # better: use timestamps to decide sampling, but this is simple
    skip_every = max(1, int(round(stream.average_rate / float(sample_rate))))

    decoded_count = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            # frame.pts * frame.time_base yields timestamp in seconds
            ts = float(frame.pts * frame.time_base)
            if ts < start_s:
                continue
            if end_s is not None and ts >= end_s:
                container.close()
                return

            # sample every skip_every decoded frames (you can base on time instead)
            if decoded_count % skip_every == 0:
                img = frame.to_ndarray(format="bgr24")  # numpy array like OpenCV
                yield img, ts
            decoded_count += 1

    container.close()
