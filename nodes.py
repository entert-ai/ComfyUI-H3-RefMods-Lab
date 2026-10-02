from pathlib import Path
import math
import json

import torch
import comfy.utils
import comfy.audio
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio, video_latent_t
from comfy_api.latest import io
import folder_paths
from comfy.cli_args import args

from . import core

REFMOD = io.Custom("H3_REFMOD_LAB")
CATEGORY = "MiniMax H3/RefMods Lab"
FOLDER = "h3_refmods_lab"
if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, str(Path(folder_paths.models_dir) / FOLDER))
    folder_paths.folder_names_and_paths[FOLDER][1].add(".safetensors")


class Create(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabCreate", display_name="Create H3 RefMod", category=CATEGORY,
            description="Encode each input image independently. A batch is separate pictures, not a video. Uses the native H3 VAE; no training or diffusion model needed.",
            inputs=[io.Vae.Input("vae"), io.Image.Input("images"),
                io.String.Input("name", default="subject"), io.String.Input("description", default="", multiline=True),
                io.String.Input("vae_label", default="H3 visual VAE (checkpoint unspecified)", tooltip="Record the selected VAE filename for reproducibility; this label does not select a model."),
                io.Int.Input("max_edge", default=768, min=32, max=2048, step=32),
                io.Int.Input("max_tokens", default=8192, min=0, max=1048576, tooltip="DiT reference-token budget; 0 disables. Excludes Qwen vision tokens.")],
            outputs=[REFMOD.Output(), io.String.Output(display_name="details")])

    @classmethod
    def execute(cls, vae, images, name, description, vae_label, max_edge, max_tokens):
        if not isinstance(vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("Connect the full MiniMax H3 visual VAE, not an audio or tiny preview VAE.")
        if images.ndim != 4 or images.shape[-1] not in (3, 4) or images.shape[0] == 0:
            raise ValueError("Connect a nonempty IMAGE batch [N,H,W,3/4].")
        if not torch.isfinite(images).all():
            raise ValueError("Input images contain non-finite pixels.")
        h, w = images.shape[1:3]
        tw, th = core.reference_size(w, h, max_edge)
        estimated = images.shape[0] * (tw // 32) * (th // 32)
        if max_tokens > 0 and estimated > max_tokens:
            raise ValueError(f"Images require {estimated:,} DiT tokens; budget is {max_tokens:,}. Reduce max_edge or image count, or raise the budget.")
        entries = []
        progress = comfy.utils.ProgressBar(images.shape[0])
        for index, image in enumerate(images):
            resized = comfy.utils.common_upscale(image[None, ..., :3].movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
            with torch.inference_mode():
                z = vae.encode(resized).detach().cpu().contiguous()
            entries.append({"name": name, "description": description, "source_index": index + 1,
                "vae_label": vae_label, "original_width": w, "original_height": h,
                "strength": 1.0, "latent": z, "jpeg": core.jpeg_tensor(resized)})
            progress.update(1)
        pack = {"entries": entries}
        core.validate(pack)
        core.check_budget(entries, max_tokens)
        return io.NodeOutput(pack, core.describe(pack))


def prepare_audio(audio, start_seconds, duration_seconds, require_full=False):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Connect a valid AUDIO source.")
    waveform, sr = audio["waveform"], audio["sample_rate"]
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or waveform.shape[-1] == 0:
        raise ValueError("Audio must be one mono or stereo source [1,1/2,samples].")
    if not isinstance(sr, (int, float)) or not math.isfinite(sr) or sr <= 0 or not torch.isfinite(waveform).all():
        raise ValueError("Audio requires a positive sample rate and finite samples.")
    first = round(start_seconds * sr)
    count = round(duration_seconds * sr)
    if first >= waveform.shape[-1] or count < 1:
        raise ValueError("The selected audio interval is empty.")
    if require_full and first + count > waveform.shape[-1] + 1:
        raise ValueError("The soundtrack must cover the selected video interval. Trim the video or supply matching audio.")
    waveform = waveform[..., first:first + count]
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    return {"waveform": waveform, "sample_rate": sr}


def encode_audio(audio_vae, audio):
    if not isinstance(audio_vae.first_stage_model, MiniMaxH3AudioVAE):
        raise ValueError("Connect the MiniMax H3 audio VAE, not the visual VAE or another model's audio VAE.")
    with torch.inference_mode():
        latent, _ = _encode_ref_audio(audio_vae, audio)
    return latent.detach().cpu().contiguous()


def clip_controls(start_seconds, duration_seconds):
    if not math.isfinite(start_seconds) or start_seconds < 0 or not math.isfinite(duration_seconds) or not 0 < duration_seconds <= 15:
        raise ValueError("Choose a nonnegative start and a duration greater than zero and at most 15 seconds.")


class CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabCreateVideo", display_name="Create H3 Video RefMod", category=CATEGORY,
            description="One IMAGE batch is one video. Samples at 24 fps, trims down to 5+17k frames, stores exact H3 latents and 2 fps JPEG frames for Qwen. Optional audio is the synchronized soundtrack of this same clip.",
            inputs=[io.Vae.Input("vae"), io.Image.Input("frames"),
                io.Float.Input("source_fps", default=24, min=0.01, max=240, force_input=True),
                io.String.Input("name", default="subject_video"), io.String.Input("description", default="", multiline=True),
                io.String.Input("vae_label", default="H3 visual VAE (checkpoint unspecified)"),
                io.Float.Input("start_seconds", default=0, min=0, max=86400, step=0.01),
                io.Float.Input("duration_seconds", default=3, min=0.21, max=15, step=0.01),
                io.Int.Input("max_edge", default=512, min=32, max=2048, step=32),
                io.Int.Input("max_tokens", default=8192, min=0, max=1048576),
                io.Audio.Input("audio", optional=True), io.Vae.Input("audio_vae", optional=True),
                io.String.Input("audio_vae_label", default="H3 audio VAE (checkpoint unspecified)", optional=True)],
            outputs=[REFMOD.Output(), io.String.Output(display_name="details")])

    @classmethod
    def execute(cls, vae, frames, source_fps, name, description, vae_label, start_seconds,
                duration_seconds, max_edge, max_tokens, audio=None, audio_vae=None,
                audio_vae_label="H3 audio VAE (checkpoint unspecified)"):
        clip_controls(start_seconds, duration_seconds)
        if not isinstance(vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("Connect the full MiniMax H3 visual VAE.")
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4) or frames.shape[0] == 0:
            raise ValueError("Connect one video as an IMAGE batch [frames,H,W,3/4].")
        if not math.isfinite(source_fps) or not 0 < source_fps <= 240:
            raise ValueError("Connect the source video's real frame rate, not the generation frame rate.")
        if not torch.isfinite(frames).all():
            raise ValueError("Video contains non-finite pixels.")
        available = frames.shape[0] / source_fps - start_seconds
        count = math.floor(min(duration_seconds, available) * 24 + 1e-6)
        if count < 5:
            raise ValueError("The selected video interval needs at least 5 frames at 24 fps.")
        count = 5 + ((count - 5) // 17) * 17
        duration = count / 24
        h, w = frames.shape[1:3]
        tw, th = core.reference_size(w, h, max_edge)
        soundtrack = None
        if audio is not None:
            if audio_vae is None:
                raise ValueError("Connect the H3 audio VAE for the paired soundtrack, or disconnect audio for a silent reference.")
            if not isinstance(audio_vae.first_stage_model, MiniMaxH3AudioVAE):
                raise ValueError("The paired soundtrack needs the MiniMax H3 audio VAE.")
            soundtrack = prepare_audio(audio, start_seconds, duration, require_full=True)
        estimate = video_latent_t(count) * (tw // 32) * (th // 32) + (math.ceil(duration * 40) * 2 if soundtrack is not None else 0)
        if max_tokens > 0 and estimate > max_tokens:
            raise ValueError(f"Clip requires about {estimate:,} DiT tokens; budget is {max_tokens:,}. Shorten the clip or reduce max_edge before encoding.")
        # Nearest-frame time sampling keeps the audio on its original timeline.
        indices = torch.floor((start_seconds + torch.arange(count, dtype=torch.float64) / 24) * source_fps).long().clamp(max=frames.shape[0] - 1)
        selected = frames[indices.to(frames.device), ..., :3]
        resized = comfy.utils.common_upscale(selected.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
        with torch.inference_mode():
            z = vae.encode(resized).detach().cpu().contiguous()
        sampled = resized[::12]
        jpeg, offsets = core.video_presentation(sampled)
        entry = {"kind": "video", "name": name, "description": description, "source_index": 1,
            "vae_label": vae_label, "original_width": w, "original_height": h,
            "source_fps": source_fps, "start_seconds": start_seconds, "duration_seconds": duration,
            "frame_count": count, "strength": 1.0, "latent": z, "jpeg": core.jpeg_tensor(resized[:1]),
            "video_jpeg": jpeg, "video_offsets": offsets, "timestamps": [i / 2 for i in range(sampled.shape[0])]}
        if soundtrack is not None:
            entry.update(kind="video_audio", audio_latent=encode_audio(audio_vae, soundtrack), audio_vae_label=audio_vae_label)
        pack = {"entries": [entry]}
        core.validate(pack)
        core.check_budget(pack["entries"], max_tokens)
        note = f"\nSelected {count} frames at 24 fps ({duration:.3f}s), from {start_seconds:g}s; requested up to {duration_seconds:g}s. Alignment trims the tail; nothing is stretched."
        return io.NodeOutput(pack, core.describe(pack) + note)


class CreateAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabCreateAudio", display_name="Create H3 Audio RefMod", category=CATEGORY,
            description="Encode one standalone voice/sound reference with H3's audio VAE. Mono is duplicated to stereo; native encoding resamples to 32 kHz. Raw audio is not stored; Qwen sees an Audio label.",
            inputs=[io.Vae.Input("audio_vae"), io.Audio.Input("audio"),
                io.String.Input("name", default="subject_voice"), io.String.Input("description", default="", multiline=True),
                io.String.Input("vae_label", default="H3 audio VAE (checkpoint unspecified)"),
                io.Float.Input("start_seconds", default=0, min=0, max=86400, step=0.01),
                io.Float.Input("duration_seconds", default=3, min=0.21, max=15, step=0.01),
                io.Int.Input("max_tokens", default=8192, min=0, max=1048576)],
            outputs=[REFMOD.Output(), io.String.Output(display_name="details")])

    @classmethod
    def execute(cls, audio_vae, audio, name, description, vae_label, start_seconds, duration_seconds, max_tokens):
        clip_controls(start_seconds, duration_seconds)
        if not isinstance(audio_vae.first_stage_model, MiniMaxH3AudioVAE):
            raise ValueError("Connect the MiniMax H3 audio VAE.")
        prepared = prepare_audio(audio, start_seconds, duration_seconds)
        duration = prepared["waveform"].shape[-1] / prepared["sample_rate"]
        if duration < 0.21:
            raise ValueError("Select at least 0.21 seconds of reference audio.")
        estimate = math.ceil(duration * 40) * 2
        if max_tokens > 0 and estimate > max_tokens:
            raise ValueError(f"Audio requires about {estimate:,} DiT tokens; budget is {max_tokens:,}. Shorten the clip before encoding.")
        z = encode_audio(audio_vae, prepared)
        # Small waveform thumbnail, independent of the inference payload.
        from PIL import Image, ImageDraw
        import numpy as np
        tile = Image.new("RGB", (256, 256), (32, 32, 32))
        draw = ImageDraw.Draw(tile)
        signal = prepared["waveform"][0].mean(0).detach().cpu()
        amplitude = max(float(signal.abs().max()), 1e-6)
        for x in range(256):
            first, last = x * signal.numel() // 256, (x + 1) * signal.numel() // 256
            peak = float(signal[first:max(first + 1, last)].abs().max()) / amplitude * 70
            draw.line((x, 128 - peak, x, 128 + peak), fill=(100, 190, 230))
        draw.text((12, 12), f"AUDIO / {duration:.2f}s", fill=(240, 240, 240))
        thumbnail = torch.from_numpy(np.array(tile, copy=True)).float().div(255)[None]
        pack = {"entries": [{"kind": "audio", "name": name, "description": description,
            "source_index": 1, "vae_label": vae_label, "strength": 1.0,
            "start_seconds": start_seconds, "duration_seconds": duration,
            "audio_latent": z, "jpeg": core.jpeg_tensor(thumbnail)}]}
        core.validate(pack)
        core.check_budget(pack["entries"], max_tokens)
        return io.NodeOutput(pack, core.describe(pack))


class Save(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabSave", display_name="Save H3 RefMod", category=CATEGORY,
            is_output_node=True,
            inputs=[REFMOD.Input("refmod"), io.String.Input("filename", default="subject"),
                io.Boolean.Input("embed_workflow", default=True, optional=True,
                    tooltip="Embed the current creation graph and API prompt so ComfyUI can open this safetensors file as a workflow.")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[REFMOD.Output(), io.String.Output(display_name="saved_file")])

    @classmethod
    def execute(cls, refmod, filename, embed_workflow=True):
        workflow, prompt = None, None
        if embed_workflow and not args.disable_metadata:
            workflow = (cls.hidden.extra_pnginfo or {}).get("workflow")
            prompt = cls.hidden.prompt
        path = core.save_pack(refmod, folder_paths.get_folder_paths(FOLDER)[0], filename, workflow=workflow, prompt=prompt)
        return io.NodeOutput(refmod, str(path), ui={"text": [f"Saved {path}"]})


class Load(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabLoad", display_name="Load H3 RefMod", category=CATEGORY,
            inputs=[io.Combo.Input("filename", options=folder_paths.get_filename_list(FOLDER))],
            outputs=[REFMOD.Output(), io.String.Output(display_name="details")])

    @classmethod
    def fingerprint_inputs(cls, filename):
        path = Path(folder_paths.get_full_path_or_raise(FOLDER, filename))
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    @classmethod
    def execute(cls, filename):
        pack = core.load_pack(folder_paths.get_full_path_or_raise(FOLDER, filename))
        return io.NodeOutput(pack, core.describe(pack))


class Combine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabCombineV2", display_name="Combine H3 RefMods", category=CATEGORY,
            description="Combine up to 100 packs in numerical input order. Connect RefMod 1 to reveal RefMod 2, and so on. Every connected pack has the same strength slider. Zero strength omits it from H3 and Qwen. Replace Combine nodes in older workflows before using this version.",
            inputs=[io.Autogrow.Input("refmods", optional=True, template=io.Autogrow.TemplateNames(
                    input=REFMOD.Input("refmod"), names=[f"refmod_{i}" for i in range(1, 101)], min=0)),
                io.String.Input("strengths", default="{}", optional=True)],
            outputs=[REFMOD.Output(), io.String.Output(display_name="reference_map")])

    @classmethod
    def execute(cls, refmods=None, strengths="{}"):
        try:
            weights = json.loads(strengths)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid input strengths. Reset the Combine strengths before generating.") from error
        if not isinstance(weights, dict):
            raise ValueError("Input strengths must be a JSON object.")
        additional = refmods or {}
        valid_names = {f"refmod_{i}" for i in range(1, 101)}
        if not isinstance(additional, dict) or any(name not in valid_names for name in additional) or any(name not in valid_names for name in weights):
            raise ValueError("Invalid expanding RefMod input names.")
        inputs = [(additional[name], weights.get(name, 1.0)) for name in sorted(additional, key=lambda name: int(name[7:]))]
        pack = core.combine_many(inputs)
        return io.NodeOutput(pack, core.describe(pack))


class SelectSources(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabSelectSources", display_name="Select H3 RefMod Sources", category=CATEGORY,
            description="Select stored images, videos or audio without re-encoding. A video and its paired soundtrack are selected together. Selection is saved in the workflow. Empty selection omits this pack; reference numbers change.",
            is_output_node=True,
            inputs=[REFMOD.Input("refmod"), io.String.Input("selection", default="all")],
            outputs=[REFMOD.Output(), io.String.Output(display_name="reference_map")])

    @classmethod
    def execute(cls, refmod, selection):
        subset = core.select_sources(refmod, selection)
        details = core.describe(subset)
        return io.NodeOutput(subset, details, ui={"sources": core.source_catalog(refmod), "text": [details]})


class Apply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabApply", display_name="Apply H3 RefMod (Latents Only)", category=CATEGORY,
            description="Append latents to every conditioning entry, preserving guides and other refs. Qwen does not see these new images. Do not also use RefMod Text Encode for the same pack.",
            inputs=[REFMOD.Input("refmod"), io.Conditioning.Input("positive"),
                io.Float.Input("strength", default=1.0, min=0, max=2, step=0.05),
                io.Int.Input("max_tokens", default=8192, min=0, max=1048576)],
            outputs=[io.Conditioning.Output(display_name="positive")])

    @classmethod
    def execute(cls, refmod, positive, strength, max_tokens):
        core.validate(refmod, allow_empty=True)
        entries = [dict(e, strength=e.get("strength", 1.0) * strength) for e in core.active_entries(refmod)] if strength > 0 else []
        if entries:
            core.validate({"entries": entries})
        for _, meta in positive:
            existing = meta.get("minimax_refs", [])
            core.check_budget(entries + existing, max_tokens)
        return io.NodeOutput(core.append_conditioning(positive, entries))


class TextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabTextEncode", display_name="H3 RefMod Text Encode", category=CATEGORY,
            description="Present saved images/video frames and audio labels to H3's Qwen encoder AND attach saved latents. Picture, Video and Audio numbering follows active pack order, separately per type. Use Empty MiniMax H3 AV Latent separately. Do not Apply the same pack again.",
            inputs=[io.Clip.Input("clip"), REFMOD.Input("refmod"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("max_tokens", default=8192, min=0, max=1048576)],
            outputs=[io.Conditioning.Output(display_name="positive"), io.String.Output(display_name="reference_map")])

    @classmethod
    def execute(cls, clip, refmod, prompt, max_tokens):
        core.validate(refmod, allow_empty=True)
        entries = core.active_entries(refmod)
        core.check_budget(entries, max_tokens)
        items = core.text_items(entries)
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if any(meta.get("minimax_refs") for _, meta in cond):
            raise ValueError("Text encoder returned references already attached; refusing to duplicate them.")
        return io.NodeOutput(core.append_conditioning(cond, entries), core.describe(refmod))


class Inspect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3RefModLabInspect", display_name="Inspect H3 RefMod", category=CATEGORY,
            is_output_node=True, inputs=[REFMOD.Input("refmod")],
            outputs=[io.Image.Output(display_name="thumbnails"), io.String.Output(display_name="details")])

    @classmethod
    def execute(cls, refmod):
        core.validate(refmod, allow_empty=True)
        details = core.describe(refmod)
        return io.NodeOutput(core.thumbnails(refmod), details, ui={"text": [details]})


NODE_CLASSES = [Create, CreateVideo, CreateAudio, Save, Load, Combine, SelectSources, Apply, TextEncode, Inspect]
