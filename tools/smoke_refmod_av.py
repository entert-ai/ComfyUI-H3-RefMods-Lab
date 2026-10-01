"""Synthetic real-VAE video/audio comparison; no diffusion model or generation."""
import importlib.util
import json
from pathlib import Path
import sys
import os
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
comfy_root_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COMFYUI_ROOT")
if not comfy_root_arg:
    raise SystemExit("Pass the ComfyUI installation directory as the first argument, or set COMFYUI_ROOT.")
COMFY = Path(comfy_root_arg).resolve()
if not (COMFY / "main.py").is_file():
    raise SystemExit("The supplied directory is not a ComfyUI installation.")
(ROOT / "artifacts").mkdir(exist_ok=True)
try:
    with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3) as response:
        queue = json.load(response)
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise SystemExit("ComfyUI is busy; leave its GPU work undisturbed and rerun this smoke test when idle.")
except OSError:
    pass
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--lowvram"]
import torch
import nodes as comfy_nodes
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

spec = importlib.util.spec_from_file_location("h3_refmods_lab", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from h3_refmods_lab import core, nodes


class Clip:
    def tokenize(self, prompt, **kwargs):
        self.items = kwargs.get("minimax_ref_items")
        return prompt
    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1), {}]]


torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
visual_name = "minimax_h3_video_vae_fp16.safetensors"
audio_name = "minimax_h3_audio_vae_fp32.safetensors"
visual = comfy_nodes.VAELoader().load_vae(visual_name)[0]
audio_vae = comfy_nodes.VAELoader().load_vae(audio_name)[0]
# A short valid clip keeps this encoding test small. This is not a quality test.
frames = torch.linspace(0, 1, 22 * 64 * 96 * 3).reshape(22, 64, 96, 3)
signal = 0.1 * torch.sin(torch.arange(64000).float() * (2 * torch.pi * 440 / 32000))
audio = {"waveform": signal[None, None].repeat(1, 2, 1), "sample_rate": 32000}
created = nodes.CreateVideo.execute(visual, frames, 24, "synthetic_motion", "", visual_name,
    0, 22 / 24, 96, 8192, audio=audio, audio_vae=audio_vae, audio_vae_label=audio_name).result[0]
standalone = nodes.CreateAudio.execute(audio_vae, audio, "synthetic_sound", "", audio_name, 0, 2, 8192).result[0]
prepared = nodes.prepare_audio(audio, 0, 22 / 24, require_full=True)
clip = Clip()
with torch.inference_mode():
    native = MiniMaxH3ReferenceToVideo.execute(clip, "test", 96, 64, 22, "match", vae=visual,
        audio_vae=audio_vae, ref_videos={"ref_video_1": frames},
        ref_video_audios={"ref_video_audio_1": prepared}, ref_audios={"ref_audio_1": audio}).result[0][0][1]["minimax_refs"]
actual = created["entries"][0]
torch.testing.assert_close(actual["latent"], native[0]["latent"].cpu(), rtol=0, atol=0)
torch.testing.assert_close(actual["audio_latent"], native[0]["audio_latent"].cpu(), rtol=0, atol=0)
torch.testing.assert_close(standalone["entries"][0]["audio_latent"], native[1]["audio_latent"].cpu(), rtol=0, atol=0)
mixed = core.combine(created, standalone, 1, 1)
destination = ROOT / "artifacts/refmod_av_smoke"
destination.mkdir(parents=True, exist_ok=True)
path = core.save_pack(mixed, destination, "synthetic_av")
restored = core.load_pack(path)
for a, b in zip(mixed["entries"], restored["entries"]):
    for field in core.tensor_fields(a):
        torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)
presentation = core.text_items(restored["entries"])
assert [item["type"] for item in presentation] == ["audio", "video", "audio"]
assert presentation[1]["timestamps"] == clip.items[1]["timestamps"]
report = {"visual_vae": visual_name, "audio_vae": audio_name, "video_frames": 22, "video_size": [96, 64],
    "video_latent_shape": list(actual["latent"].shape), "paired_audio_shape": list(actual["audio_latent"].shape),
    "standalone_audio_shape": list(standalone["entries"][0]["audio_latent"].shape),
    "native_visual_exact_match": True, "native_paired_audio_exact_match": True,
    "native_standalone_audio_exact_match": True, "save_load_exact_match": True,
    "qwen_item_order_and_timestamps_match": True, "reference_tokens": core.token_count(mixed["entries"]),
    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": round(time.monotonic() - started, 2),
    "saved_pack": str(path)}
(destination / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

