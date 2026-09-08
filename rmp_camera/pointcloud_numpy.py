"""Validated strided PointCloud2 numeric views without per-point Python objects."""
import numpy as np

_TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def read_numeric_cloud(message, names=('x', 'y', 'z')):
    width, height = int(message.width), int(message.height)
    step, row = int(message.point_step), int(message.row_step)
    if min(width, height, step, row) < 0 or row < width * step:
        raise ValueError('invalid PointCloud2 dimensions/strides')
    fields = {field.name: field for field in message.fields}
    formats, offsets = [], []
    for name in names:
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in _TYPES:
            raise ValueError('missing or invalid scalar point field: ' + name)
        dtype = np.dtype(('>' if message.is_bigendian else '<') + _TYPES[field.datatype])
        if field.offset < 0 or field.offset + dtype.itemsize > step:
            raise ValueError('point field outside point_step: ' + name)
        formats.append(dtype)
        offsets.append(field.offset)
    if not width or not height:
        return np.empty((0, len(names)), dtype=np.float64)
    if step <= 0 or len(message.data) < (height - 1) * row + width * step:
        raise ValueError('truncated PointCloud2 payload')
    dtype = np.dtype(dict(names=list(names), formats=formats, offsets=offsets, itemsize=step))
    records = np.ndarray((height, width), dtype=dtype, buffer=message.data, strides=(row, step))
    return np.stack([records[name] for name in names], axis=-1).reshape(-1, len(names)).astype(
        np.float64, copy=False)


class LatestCloudBuffer:
    """Small bounded queue: newest usable frame, not newest regardless of sync."""
    def __init__(self, capacity=5, max_age_s=.25):
        self.capacity = capacity
        self.max_age_s = max_age_s
        self.frames = []
        self.last_stamp = None
        self.last_clock = None
        self.expired = 0
        self.superseded = 0
        self.out_of_order = 0
        self.expired_now = 0

    def reset(self):
        self.frames.clear()
        self.last_stamp = None

    def add(self, message, received, clock_ns):
        rewound = self.last_clock is not None and clock_ns < self.last_clock
        if rewound:
            self.reset()
        self.last_clock = clock_ns
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if self.last_stamp is not None and stamp <= self.last_stamp:
            self.out_of_order += 1
            return False, rewound
        self.last_stamp = stamp
        self.frames.append((message, received, stamp))
        if len(self.frames) > self.capacity:
            self.superseded += len(self.frames) - self.capacity
            self.frames = self.frames[-self.capacity:]
        return True, rewound

    def take(self, clock_ns, wall_time, ready):
        self.expired_now = 0
        kept = []
        for item in self.frames:
            age = (clock_ns - item[2]) / 1e9
            if age > self.max_age_s or wall_time - item[1] > self.max_age_s or age < -.05:
                self.expired_now += 1
            else:
                kept.append(item)
        self.expired += self.expired_now
        self.frames = kept
        for index in range(len(kept) - 1, -1, -1):
            if ready(kept[index][0]):
                self.superseded += index
                self.frames = kept[index + 1:]
                return kept[index]
        return None
