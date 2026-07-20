import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Value is not JSON serializable: {type(value).__name__}')


class JSONLMetricsWriter:
    def __init__(self, path, run_id=None, enabled=True, sync=True):
        self.path = Path(path)
        self.run_id = run_id
        self.enabled = enabled
        self.sync = sync
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event):
        if not self.enabled:
            return False
        payload = _json_safe({
            **event,
            'schema_version': SCHEMA_VERSION,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'run_id': self.run_id,
        })
        encoded = (json.dumps(
            payload,
            allow_nan=False,
            separators=(',', ':'),
            sort_keys=True,
        ) + '\n').encode('utf-8')
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError(f'incomplete metrics write: {written}/{len(encoded)} bytes')
            if self.sync:
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True


def read_complete_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    events = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
    return events
