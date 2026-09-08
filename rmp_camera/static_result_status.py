"""Generation-stamped validity protocol for static sphere results.

The cloud is authoritative only when paired with a successful result status.
Failures and spatially incomplete observations are never empty confirmations.
"""

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import Header


STATUS_NAME = "static_sphere_generation"
VALID_STATES = ("valid", "empty", "unknown")


def make_result_status(stamp, frame_id, state, reason=""):
    if state not in VALID_STATES:
        raise ValueError("invalid static result state")
    return DiagnosticArray(header=Header(stamp=stamp, frame_id=frame_id), status=[
        DiagnosticStatus(name=STATUS_NAME,
            level=DiagnosticStatus.STALE if state == "unknown" else DiagnosticStatus.OK,
            message=state, values=[KeyValue(key="reason", value=reason)])])


def read_result_status(message):
    if len(message.status) != 1 or message.status[0].name != STATUS_NAME:
        raise ValueError("unrecognized static result status")
    item = message.status[0]
    if item.message not in VALID_STATES:
        raise ValueError("unrecognized static result state")
    expected = DiagnosticStatus.STALE if item.message == "unknown" else DiagnosticStatus.OK
    if item.level != expected:
        raise ValueError("inconsistent static result status")
    return item.message


class StaticResultGate:
    """Bounded, exact-generation join; wall time handles a stopped bag clock."""

    def __init__(self, timeout_s=1.0, capacity=8):
        self.timeout_s = timeout_s
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.pending = {}
        self.last_completed = None

    def add(self, stamp, kind, payload, now):
        self.pending = {k: v for k, v in self.pending.items()
                        if now - v[0] <= self.timeout_s}
        if self.last_completed is not None and stamp <= self.last_completed:
            return None
        if kind == "status" and payload == "unknown":
            self.last_completed = stamp
            self.pending = {k: v for k, v in self.pending.items() if k > stamp}
            return None
        entry = self.pending.setdefault(stamp, [now, {}])
        entry[1][kind] = payload
        while len(self.pending) > self.capacity:
            del self.pending[next(iter(self.pending))]
        parts = entry[1]
        if "cloud" not in parts or "status" not in parts:
            return None
        self.last_completed = stamp
        self.pending = {k: v for k, v in self.pending.items() if k > stamp}
        return parts["cloud"], parts["status"]
