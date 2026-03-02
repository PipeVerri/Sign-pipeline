from sortedcontainers import SortedDict

class SegmentTree:
    """
    Represents a set of integers using disjoint, merged intervals [start, end] inclusive.
    All operations are O(log n).
    """
    def __init__(self):
        self._ivs = SortedDict()

    def add_point(self, x: int):
        ivs = self._ivs
        if not ivs:
            ivs[x] = x
            return

        idx = ivs.bisect_right(x)

        new_start = x
        new_end = x

        # merge with left neighbor
        if idx > 0:
            left_start, left_end = ivs.peekitem(idx - 1)
            if left_end >= x - 1:   # touching or overlapping
                new_start = left_start
                new_end = max(new_end, left_end)
                del ivs[left_start]
                idx -= 1

        # merge with right neighbors
        while idx < len(ivs):
            right_start, right_end = ivs.peekitem(idx)
            if right_start > new_end + 1:
                break
            new_end = max(new_end, right_end)
            del ivs[right_start]

        ivs[new_start] = new_end

    def __contains__(self, x: int) -> bool:
        return self.get_interval(x) is not None

    def get_interval(self, x: int):
        """
        Return the (start, end) interval containing x, or None.
        Runs in O(log n).
        """
        ivs = self._ivs
        if not ivs:
            return None

        idx = ivs.bisect_right(x) - 1
        if idx < 0:
            return None

        start, end = ivs.peekitem(idx)
        if start <= x <= end:
            return (start, end)

        return None

    def intervals(self):
        return list(self._ivs.items())
