"""Compare native and RefMod encoding using a synthetic image and the real H3 VAE."""
import importlib.util
import json
from pathlib import Path
import sys
import os
import time

ROOT = Path(__file__).resolve().parents[1]
comfy_root_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COMFYUI_ROOT")
if not comfy_root_arg:
    raise SystemExit("Pass the ComfyUI installation directory as the first argument, or set COMFYUI_ROOT.")
COMFY = Path(comfy_root_arg).resolve()
if not (COMFY / "main.py").is_file():
    raise SystemExit("The supplied directory is not a ComfyUI installation.")
(ROOT / "artifacts").mkdir(exist_ok=True)
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
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1), {}]]


torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
vae_name = "minimax_h3_video_vae_fp16.safetensors"
vae = comfy_nodes.VAELoader().load_vae(vae_name)[0]
image = torch.linspace(0, 1, 448 * 768 * 3).reshape(1, 448, 768, 3)
created = nodes.Create.execute(vae, image, "synthetic_test", "Synthetic gradient for VAE verification", vae_name, 768, 8192).result[0]
with torch.inference_mode():
    native = MiniMaxH3ReferenceToVideo.execute(Clip(), "test", 768, 448, 5, "match", vae=vae, ref_images={"ref_image_1": image}).result[0][0][1]["minimax_refs"][0]["latent"]
actual = created["entries"][0]["latent"]
torch.testing.assert_close(actual, native.cpu(), rtol=0, atol=0)
path = core.save_pack(created, ROOT / "artifacts/refmod_vae_smoke", "synthetic_test")
restored = core.load_pack(path)
torch.testing.assert_close(restored["entries"][0]["latent"], actual, rtol=0, atol=0)
report = {"vae": vae_name, "source": "synthetic gradient 768x448", "latent_shape": list(actual.shape),
    "native_latent_exact_match": True, "save_load_exact_match": True,
    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    "elapsed_seconds": round(time.monotonic() - started, 2), "saved_pack": str(path)}
(ROOT / "artifacts/refmod_vae_smoke/report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

