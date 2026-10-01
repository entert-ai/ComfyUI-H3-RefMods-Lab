"""Versioned multimodal RefMods and native H3 conditioning payloads."""

import io
import hashlib
import base64
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import save_file
import torch

FORMAT = "h3-refmods-lab"
VERSION = 2


def kind(entry):
    return entry.get("kind", "image")


def tensor_fields(entry):
    fields = ["jpeg"]
    if kind(entry) != "audio":
        fields.append("latent")
    if kind(entry) in ("video", "video_audio"):
        fields.extend(("video_jpeg", "video_offsets"))
    if kind(entry) in ("audio", "video_audio"):
        fields.append("audio_latent")
    return fields


def video_presentation(frames):
    encoded = [jpeg_tensor(frame[None]) for frame in frames]
    offsets = [0]
    for frame in encoded:
        offsets.append(offsets[-1] + frame.numel())
    return torch.cat(encoded), torch.tensor(offsets, dtype=torch.int64)


def source_video(entry):
    data, offsets = entry["video_jpeg"], entry["video_offsets"].tolist()
    return torch.cat([source_image({"jpeg": data[a:b]}) for a, b in zip(offsets, offsets[1:])])


def text_items(entries):
    items = []
    for entry in entries:
        modality = kind(entry)
        if modality == "image":
            items.append({"type": "image", "data": source_image(entry)})
        else:
            if modality in ("audio", "video_audio"):
                # Native H3 Qwen uses only the audio label, not the waveform.
                items.append({"type": "audio"})
            if modality in ("video", "video_audio"):
                items.append({"type": "video", "data": source_video(entry), "timestamps": entry["timestamps"]})
    return items


def reference_size(width, height, max_edge):
    scale = min(1.0, max_edge / max(width, height))
    return (max(32, round(width * scale / 32) * 32),
            max(32, round(height * scale / 32) * 32))


def validate(pack, allow_empty=False):
    if not isinstance(pack.get("entries"), list) or (not pack["entries"] and not allow_empty):
        raise ValueError("A RefMod must contain at least one reference source.")
    for entry in pack["entries"]:
        modality = kind(entry)
        if modality not in ("image", "video", "audio", "video_audio"):
            raise ValueError("Unknown reference kind.")
        if modality != "audio":
            z = entry["latent"]
            temporal_ok = z.ndim == 5 and (z.shape[2] == 1 if modality == "image" else z.shape[2] >= 2 and z.shape[2] % 5 == 2)
            if (not temporal_ok or tuple(z.shape[:2]) != (1, 24)
                    or min(z.shape[-2:]) < 2 or z.shape[-2] % 2 or z.shape[-1] % 2):
                raise ValueError("Expected H3 visual latent [1,24,T,H/16,W/16], with aligned spatial and temporal dimensions.")
            if not z.is_floating_point() or not torch.isfinite(z).all():
                raise ValueError("Reference latents must be finite floating-point tensors.")
        if modality in ("audio", "video_audio"):
            audio = entry["audio_latent"]
            if audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2) or audio.shape[-1] < 1:
                raise ValueError("Expected H3 stereo audio latent [1,32,2,T40].")
            if not audio.is_floating_point() or not torch.isfinite(audio).all():
                raise ValueError("Audio latents must be finite floating-point tensors.")
        strength = entry.get("strength", 1.0)
        if not math.isfinite(strength) or not 0 <= strength <= 2:
            raise ValueError("Reference strength must be between 0 and 2.")
        pixels = entry["jpeg"]
        if pixels.dtype != torch.uint8 or pixels.ndim != 1:
            raise ValueError("Stored source image must be a JPEG byte tensor.")
        with Image.open(io.BytesIO(pixels.cpu().numpy().tobytes())) as image:
            if image.format != "JPEG" or (modality != "audio" and image.size != (z.shape[-1] * 16, z.shape[-2] * 16)):
                raise ValueError("Stored image dimensions do not match the reference latent.")
        if modality in ("video", "video_audio"):
            data, offsets = entry["video_jpeg"], entry["video_offsets"]
            if data.dtype != torch.uint8 or data.ndim != 1 or offsets.dtype != torch.int64 or offsets.ndim != 1:
                raise ValueError("Video presentation must contain JPEG bytes and integer offsets.")
            points = offsets.tolist()
            expected_frames = 5 + ((z.shape[2] - 2) // 5) * 17
            expected_samples = len(range(0, expected_frames, 12))
            if len(points) != expected_samples + 1 or points[0] != 0 or points[-1] != data.numel() or any(a >= b for a, b in zip(points, points[1:])):
                raise ValueError("Invalid video presentation offsets or frame count.")
            if entry.get("timestamps") != [i / 2 for i in range(expected_samples)]:
                raise ValueError("Video presentation must use 2 fps timestamps starting at zero.")
            for a, b in zip(points, points[1:]):
                with Image.open(io.BytesIO(data[a:b].cpu().numpy().tobytes())) as image:
                    if image.format != "JPEG" or image.size != (z.shape[-1] * 16, z.shape[-2] * 16):
                        raise ValueError("Video presentation dimensions do not match its latent.")


def jpeg_tensor(image):
    array = (image[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=95, subsampling=0)
    return torch.from_numpy(np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy())


def source_image(entry):
    with Image.open(io.BytesIO(entry["jpeg"].cpu().numpy().tobytes())) as image:
        array = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(array).float().div_(255).unsqueeze(0)


def active_entries(pack):
    return [e for e in pack["entries"] if e.get("strength", 1.0) > 0]


def token_count(entries):
    count = 0
    for entry in entries:
        modality = kind(entry)
        if modality not in ("image", "video", "audio", "video_audio"):
            raise ValueError("Cannot count tokens for an unknown reference kind.")
        if modality != "audio":
            z = entry["latent"]
            count += z.shape[2] * (z.shape[3] // 2) * (z.shape[4] // 2)
        if modality in ("audio", "video_audio"):
            count += entry["audio_latent"].shape[-1] * 2
    return count


def check_budget(entries, max_tokens):
    count = token_count(entries)
    if max_tokens > 0 and count > max_tokens:
        raise ValueError(f"References require {count:,} DiT tokens; budget is {max_tokens:,}. Reduce resolution or reference count, or raise the budget. Nothing was dropped.")
    return count


def combine(pack_a, pack_b, strength_a, strength_b):
    return combine_many([(pack_a, strength_a), (pack_b, strength_b)])


def combine_many(inputs):
    entries = []
    for pack, strength in inputs:
        if pack is None:
            continue
        validate(pack, allow_empty=True)
        if not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 2:
            raise ValueError("Each input strength must be between 0 and 2.")
        for entry in pack["entries"]:
            updated = dict(entry, strength=entry.get("strength", 1.0) * strength)
            entries.append(updated)
    result = {"entries": entries}
    validate(result, allow_empty=True)
    return result


def blocks(entries):
    result = []
    for entry in entries:
        modality, strength = kind(entry), entry.get("strength", 1.0)
        block = {"kind": modality}
        if modality != "audio":
            z = entry["latent"]
            block.update(latent=z * strength, latent_h=z.shape[3], latent_w=z.shape[4])
        if modality in ("video", "video_audio"):
            block.update(latent_t=entry["latent"].shape[2], ref_audio_t=0, audio_latent=None)
        if modality in ("audio", "video_audio"):
            z = entry["audio_latent"]
            block.update(audio_latent=z * strength, ref_audio_t=z.shape[-1])
        result.append(block)
    return result


def append_conditioning(conditioning, entries):
    refs = blocks(entries)
    return [[embedding, dict(meta, minimax_refs=list(meta.get("minimax_refs", [])) + refs)]
            for embedding, meta in conditioning]


def describe(pack):
    entries = active_entries(pack)
    lines = [f"{len(entries)} active sources; {token_count(entries):,} DiT reference tokens (excludes text/vision tokens)."]
    counters = {"Picture": 0, "Video": 0, "Audio": 0}
    for entry in entries:
        modality = kind(entry)
        labels = ["Picture"] if modality == "image" else (["Audio", "Video"] if modality == "video_audio" else ["Audio" if modality == "audio" else "Video"])
        tags = []
        for label in labels:
            counters[label] += 1
            tags.append(f"<{label} {counters[label]}>")
        size = ""
        if modality != "audio":
            z = entry["latent"]
            size = f"{z.shape[4] * 16}×{z.shape[3] * 16}; "
        timing = f"{entry['duration_seconds']:.3f}s; " if "duration_seconds" in entry else ""
        lines.append(f"{' + '.join(tags)} = {entry['name']} [{entry.get('source_index', 1)}]; {modality}; "
                     f"{size}{timing}strength {entry.get('strength', 1.0):g}; VAE: {entry.get('vae_label', 'unspecified')}")
        if entry.get("description"):
            lines.append(entry["description"])
    lines.append("Descriptions are metadata; write subject definitions in your prompt. Zero-strength references are omitted.")
    return "\n".join(lines)


def thumbnails(pack):
    tiles = []
    for entry in pack["entries"]:
        with Image.open(io.BytesIO(entry["jpeg"].cpu().numpy().tobytes())) as image:
            tile = ImageOps.pad(image.convert("RGB"), (256, 256), color=(32, 32, 32))
            tiles.append(torch.from_numpy(np.array(tile, copy=True)).float() / 255)
    return torch.stack(tiles) if tiles else torch.zeros(0, 256, 256, 3)


def source_ids(pack):
    counts, result = {}, []
    for entry in pack["entries"]:
        digest = hashlib.sha256()
        digest.update(json.dumps([entry["name"], entry.get("source_index", 1)], ensure_ascii=False).encode())
        digest.update(entry["jpeg"].cpu().numpy().tobytes())
        if kind(entry) != "image":
            digest.update(kind(entry).encode())
            for field in ("latent", "audio_latent", "video_jpeg", "video_offsets"):
                if field in entry:
                    value = entry[field].detach().cpu().contiguous()
                    digest.update(str((value.dtype, tuple(value.shape))).encode())
                    digest.update(value.view(torch.uint8).numpy().tobytes())
        base = digest.hexdigest()
        counts[base] = counts.get(base, 0) + 1
        result.append(f"{base}:{counts[base]}")
    return result


def selected_ids(pack, selection):
    ids = source_ids(pack)
    if selection == "all":
        return set(ids)
    try:
        chosen = json.loads(selection)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Invalid source selection. Refresh sources and choose Select all or the desired images.") from error
    if not isinstance(chosen, list) or any(not isinstance(item, str) for item in chosen):
        raise ValueError("Source selection must be 'all' or a JSON list of source IDs.")
    missing = set(chosen) - set(ids)
    if missing:
        raise ValueError("Saved selection contains sources missing from this pack. Refresh sources and select again before generating.")
    return set(chosen)


def select_sources(pack, selection):
    validate(pack, allow_empty=True)
    chosen = selected_ids(pack, selection)
    return {"entries": [entry for identifier, entry in zip(source_ids(pack), pack["entries"]) if identifier in chosen]}


def source_catalog(pack):
    result = []
    for identifier, entry in zip(source_ids(pack), pack["entries"]):
        with Image.open(io.BytesIO(entry["jpeg"].cpu().numpy().tobytes())) as image:
            image = ImageOps.pad(image.convert("RGB"), (192, 192), color=(32, 32, 32))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
        timing = f" / {entry['duration_seconds']:.2f}s" if "duration_seconds" in entry else ""
        result.append({"source_id": identifier, "label": f"{entry['name']} / {kind(entry)} / source {entry.get('source_index', 1)}{timing}",
            "tokens": token_count([entry]), "strength": entry.get("strength", 1.0),
            "thumbnail": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
    return result


def save_pack(pack, directory, name, workflow=None, prompt=None):
    validate(pack)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Filename must use 1–80 letters, numbers, underscores or hyphens, starting with a letter or number.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, manifest = {}, []
    for i, entry in enumerate(pack["entries"]):
        for field in tensor_fields(entry):
            tensors[f"ref.{i}.{field}"] = entry[field].detach().cpu().contiguous().clone()
        manifest.append({k: entry[k] for k in ("kind", "name", "description", "source_index", "vae_label", "audio_vae_label", "strength", "original_width", "original_height", "source_fps", "start_seconds", "duration_seconds", "frame_count", "timestamps") if k in entry})
    metadata = {"format": FORMAT, "version": str(VERSION), "manifest": json.dumps(manifest, ensure_ascii=False)}
    if workflow is not None:
        metadata["workflow"] = json.dumps(workflow, ensure_ascii=False)
    if prompt is not None:
        metadata["prompt"] = json.dumps(prompt, ensure_ascii=False)
    fd, temporary = tempfile.mkstemp(prefix=".refmod-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata=metadata)
        if workflow is not None:
            with open(temporary, "rb") as handle:
                header_size = int.from_bytes(handle.read(8), "little")
            if header_size + 8 > 4 * 1024 * 1024:
                raise ValueError("Embedded workflow exceeds ComfyUI's 4 MiB metadata import limit. Reduce the graph or turn off Embed workflow.")
        for counter in range(1, 100000):
            destination = directory / f"{name}_{counter:05d}.safetensors"
            try:
                os.link(temporary, destination)
                return destination
            except FileExistsError:
                continue
        raise ValueError("No unused numbered filename available.")
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_pack(path):
    entries = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT or metadata.get("version") not in ("1", "2"):
            raise ValueError("This is not an H3 RefMods Lab v1/v2 pack. Other RefMod formats are not supported yet.")
        manifest = json.loads(metadata["manifest"])
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("RefMod manifest must contain reference entries.")
        if any(not isinstance(entry, dict) or not isinstance(entry.get("name"), str) for entry in manifest):
            raise ValueError("RefMod manifest entry requires a name.")
        if metadata["version"] == "1" and any(kind(entry) != "image" for entry in manifest):
            raise ValueError("Version 1 supports images only.")
        expected = {f"ref.{i}.{field}" for i, entry in enumerate(manifest) for field in tensor_fields(entry)}
        if set(handle.keys()) != expected:
            raise ValueError("RefMod tensors do not match its manifest.")
        for i, entry in enumerate(manifest):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError("RefMod manifest entry requires a name.")
            entries.append(dict(entry, **{field: handle.get_tensor(f"ref.{i}.{field}") for field in tensor_fields(entry)}))
    pack = {"entries": entries}
    validate(pack)
    return pack
