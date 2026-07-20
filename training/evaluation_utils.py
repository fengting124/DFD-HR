import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch


def normalize_checkpoint_state_dict(checkpoint):
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError('Checkpoint must be a state dict or contain a state_dict mapping.')

    normalized = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix('module.')
        if new_key.startswith('base_model.'):
            new_key = 'backbone.' + new_key.removeprefix('base_model.')
        if new_key.startswith('classifier.'):
            new_key = 'head.' + new_key.removeprefix('classifier.')
        if new_key in normalized:
            raise ValueError(f'Checkpoint key normalization collision: {new_key}')
        normalized[new_key] = value
    return normalized


def load_checkpoint_strict(model, checkpoint_path, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return {'tensor_count': len(state_dict)}


def select_fixed_subset(dataset, max_samples):
    sample_count = len(dataset.image_list)
    if max_samples is None or max_samples >= sample_count:
        return sample_count
    if max_samples < 1:
        raise ValueError('max_samples must be positive.')

    groups = defaultdict(list)
    for index, (path, label) in enumerate(zip(dataset.image_list, dataset.label_list)):
        groups[int(label)].append((str(path), index))
    for entries in groups.values():
        entries.sort()

    selected = []
    offsets = {label: 0 for label in groups}
    labels = sorted(groups)
    while len(selected) < max_samples:
        made_progress = False
        for label in labels:
            offset = offsets[label]
            if offset < len(groups[label]):
                selected.append(groups[label][offset][1])
                offsets[label] += 1
                made_progress = True
                if len(selected) == max_samples:
                    break
        if not made_progress:
            break

    dataset.image_list = [dataset.image_list[index] for index in selected]
    dataset.label_list = [dataset.label_list[index] for index in selected]
    dataset.data_dict = {
        key: [values[index] for index in selected]
        for key, values in dataset.data_dict.items()
    }
    return len(selected)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_metrics(metrics):
    return {
        dataset_name: {
            metric_name: _json_value(value)
            for metric_name, value in dataset_metrics.items()
            if metric_name not in {'pred', 'label'}
        }
        for dataset_name, dataset_metrics in metrics.items()
    }


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_write_json(payload, output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix='.evaluation-', suffix='.tmp', dir=output_dir)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            json.dump(_json_value(payload), file, indent=2, sort_keys=True)
            file.write('\n')
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
