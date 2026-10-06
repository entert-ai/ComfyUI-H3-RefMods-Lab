"""Run with ComfyUI's Python; all tests use synthetic media and CPU tensors."""
import importlib
import importlib.util
import asyncio
import json
from pathlib import Path
import sys
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
comfy_root_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COMFYUI_ROOT")
if not comfy_root_arg:
    raise SystemExit("Pass the ComfyUI installation directory as the first argument, or set COMFYUI_ROOT.")
COMFY = Path(comfy_root_arg).resolve()
if not (COMFY / "main.py").is_file():
    raise SystemExit("The supplied directory is not a ComfyUI installation.")
(ROOT / "artifacts").mkdir(exist_ok=True)
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, str(COMFY))
spec = importlib.util.spec_from_file_location("h3_refmods_lab", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from h3_refmods_lab import core, nodes
from h3_refmods_lab import picker_routes
import torch
from safetensors.torch import save_file
from safetensors import safe_open


def pack(name="alice", strength=1.0):
    return {"entries": [{"name": name, "description": "reference description", "source_index": 1,
        "vae_label": "test H3 VAE", "strength": strength,
        "latent": torch.ones(1, 24, 1, 4, 6),
        "jpeg": core.jpeg_tensor(torch.full((1, 64, 96, 3), 0.5))}]}


def video_pack(paired=False):
    frames = torch.full((2, 64, 96, 3), 0.4)
    jpeg, offsets = core.video_presentation(frames)
    entry = dict(pack("motion")["entries"][0], kind="video", latent=torch.ones(1, 24, 7, 4, 6),
        video_jpeg=jpeg, video_offsets=offsets, timestamps=[0.0, 0.5], frame_count=22,
        duration_seconds=22 / 24, source_fps=24, start_seconds=0)
    if paired:
        entry.update(kind="video_audio", audio_latent=torch.ones(1, 32, 2, 37))
    return {"entries": [entry]}


def audio_pack():
    entry = dict(pack("voice")["entries"][0], kind="audio", audio_latent=torch.ones(1, 32, 2, 80), duration_seconds=2)
    entry.pop("latent")
    return {"entries": [entry]}


class RefModTests(unittest.TestCase):
    def test_extra_model_paths_filter_refmod_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            paths = [directory, str(Path(directory) / "second")]
            with patch.dict(nodes.folder_paths.folder_names_and_paths, {nodes.FOLDER: (paths, set())}), patch.dict(nodes.folder_paths.filename_list_cache, {}, clear=True):
                saved = core.save_pack(pack(), directory, "demo")
                (Path(directory) / "notes.txt").write_text("not a pack")
                importlib.reload(nodes)
                self.assertIs(nodes.folder_paths.folder_names_and_paths[nodes.FOLDER][0], paths)
                self.assertEqual(nodes.folder_paths.folder_names_and_paths[nodes.FOLDER][1], {".safetensors"})
                self.assertEqual(nodes.folder_paths.get_filename_list(nodes.FOLDER), [saved.name])
                self.assertEqual(nodes.Load.define_schema().inputs[0].options, [saved.name])

    def test_saved_instructions_roundtrip_and_subject_identity(self):
        a = nodes.SetInstructions.execute(pack("Alice"), "a portrait", subject_name="Alice", retention_strategy="partially_preserved", retention_details="Keep facial identity; allow new clothing").result[0]
        b = nodes.SetInstructions.execute(pack("Alice"), "her body shape", subject_name="Alice", retention_strategy="fully_preserved").result[0]
        c = nodes.SetInstructions.execute(pack("Alice"), "a different person", subject_name="Alice Jones").result[0]
        combined = core.combine_many([(a, 1), (b, 1), (c, 1)])
        definitions, retention = core.reference_instructions(combined["entries"])
        self.assertEqual(definitions, ["<Subject 1> is Alice.", "<Picture 1> provides a portrait.", "<Picture 2> provides her body shape.", "<Subject 2> is Alice Jones.", "<Picture 3> provides a different person."])
        self.assertIn("<Picture 1>: partially_preserved - Keep facial identity; allow new clothing.", retention)
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            restored = core.load_pack(core.save_pack(combined, directory, "instructions"))
        self.assertEqual(core.reference_instructions(restored["entries"]), (definitions, retention))
        self.assertEqual(core.source_ids(restored), core.source_ids(combined))
        self.assertTrue(torch.equal(restored["entries"][0]["latent"], a["entries"][0]["latent"]))
        self.assertNotIn("subject_id", pack()["entries"][0])
        original = pack("Existing")
        changed = nodes.SetInstructions.execute(original, "a portrait", subject_name="Alice").result[0]
        self.assertEqual(core.source_ids(original), core.source_ids(changed))
        self.assertIs(original["entries"][0]["latent"], changed["entries"][0]["latent"])
        self.assertEqual(original["entries"][0]["description"], "reference description")
        empty_description = nodes.SetInstructions.execute(original, "", subject_name="Alice").result[0]
        self.assertIn("<Picture 1> provides a visual reference.", core.reference_instructions(empty_description["entries"])[0])

    def test_instruction_selection_renumbering_and_conflicts(self):
        a = nodes.SetInstructions.execute(pack("Alice"), "a portrait", subject_name="Alice").result[0]
        b = nodes.SetInstructions.execute(pack("Bob"), "a portrait", subject_name="Bob").result[0]
        combined = core.combine_many([(a, 0), (b, 1)])
        definitions, _ = core.reference_instructions(combined["entries"])
        self.assertEqual(definitions, ["<Subject 1> is Bob.", "<Picture 1> provides a portrait."])
        combined = core.combine_many([(a, 1), (b, 1)])
        selected = core.select_sources(combined, json.dumps([core.source_ids(combined)[1]]))
        self.assertEqual(core.reference_instructions(selected["entries"])[0], definitions)
        conflict = dict(a["entries"][0], subject_name="Bob")
        self.assertIn("<Subject 2> is Bob.", core.reference_instructions(a["entries"] + [conflict])[0])
        for marker in ("fully_copy", "invalid"):
            with self.assertRaises(ValueError):
                core.instruction_metadata("Alice", retention_strategy=marker)

    def test_instruction_audio_and_video_labels(self):
        paired = nodes.SetInstructions.execute(video_pack(True), "walking motion", subject_name="Alice", retention_strategy="attribute_transfer", audio_retention_strategy="reference", audio_retention_details="Voice timbre only").result[0]
        voice = nodes.SetInstructions.execute(audio_pack(), "a voice", subject_name="Alice", audio_retention_strategy="weak_reference").result[0]
        result = core.combine_many([(voice, 1), (paired, 1)])
        definitions, retention = core.reference_instructions(result["entries"])
        self.assertIn("<Audio 1> provides a voice.", definitions)
        self.assertIn("<Audio 2> provides the synchronized soundtrack.", definitions)
        self.assertIn("<Video 1> provides walking motion.", definitions)
        self.assertIn("<Audio 2>: reference - Voice timbre only.", retention)
        self.assertIn("<Video 1>: attribute_transfer.", retention)
        self.assertEqual(core.token_count(result["entries"]), core.token_count(voice["entries"] + paired["entries"]))

    def test_text_encode_final_prompt_and_opt_out(self):
        class Clip:
            def tokenize(self, prompt, minimax_ref_items):
                self.prompt, self.items = prompt, minimax_ref_items
                return "tokens"
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros(1), {}]]
        clip = Clip()
        refs = nodes.SetInstructions.execute(pack("Alice"), "a portrait", subject_name="Alice", retention_strategy="fully_preserved").result[0]
        body = "[Summary]\n<Subject 1> waves."
        result = nodes.TextEncode.execute(clip, refs, body, 8192, True).result
        self.assertEqual(result[2], clip.prompt)
        self.assertIn("[Subject Definitions]\n<Subject 1> is Alice.", clip.prompt)
        self.assertIn("[Retention Analysis]\n<Picture 1>: fully_preserved.", clip.prompt)
        self.assertTrue(clip.prompt.endswith(body))
        self.assertEqual(len(result[0][0][1]["minimax_refs"]), 1)
        self.assertEqual(nodes.TextEncode.execute(clip, refs, body, 8192, False).result[2], body)
        for heading in ("Subject Definitions", "Retention Analysis"):
            with self.assertRaisesRegex(ValueError, "already contains"):
                nodes.TextEncode.execute(clip, refs, f"[{heading}]\nmanual instructions", 8192, True)
        legacy = pack()
        self.assertNotIn("<Subject", core.build_prompt(legacy["entries"], body.replace("<Subject 1>", "The person"), True))
        self.assertEqual(core.build_prompt([], body, True), body)

    def test_one_subject_field_and_automatic_labels(self):
        a = nodes.SetInstructions.execute(pack("old-label"), "portrait", subject_name="Alice").result[0]
        b = nodes.SetInstructions.execute(audio_pack(), "voice", subject_name="  alice  ").result[0]
        c = nodes.SetInstructions.execute(video_pack(), "motion", subject_name="Alice Jones").result[0]
        combined = core.combine_many([(a, 1), (b, 1), (c, 1)])
        definitions, _ = core.reference_instructions(combined["entries"])
        self.assertEqual(sum(line.startswith("<Subject ") for line in definitions), 2)
        self.assertEqual(core.source_labels(combined["entries"]), ["Alice1", "Alice2", "Alice Jones1"])
        self.assertEqual(a["entries"][0]["subject_id"], b["entries"][0]["subject_id"])
        self.assertEqual(core.source_ids(a), core.source_ids(pack("old-label")))
        self.assertTrue(core.source_catalog(combined)[1]["label"].startswith("Alice2 / audio"))
        for cls in (nodes.Create, nodes.CreateVideo, nodes.CreateAudio, nodes.SetInstructions):
            inputs = cls.INPUT_TYPES()
            names = set(inputs.get("required", {})) | set(inputs.get("optional", {}))
            self.assertIn("subject_name", names)
            self.assertNotIn("name", names)
            self.assertNotIn("subject_key", names)
            self.assertNotIn("vae_label", names)
            self.assertNotIn("audio_vae_label", names)
        with self.assertRaisesRegex(ValueError, "Subject name"):
            core.instruction_metadata("  ")
        # Old saved IDs remain readable but names now control grouping.
        old = dict(a["entries"][0], subject_id="auto:legacy-other-id")
        self.assertEqual(sum(line.startswith("<Subject ") for line in core.reference_instructions(a["entries"] + [old])[0]), 1)

    def test_preview_instruction_labels_match_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            saved = core.save_pack(pack("old-label"), directory, "legacy")
            with patch.dict(nodes.folder_paths.folder_names_and_paths, {nodes.FOLDER: ([directory], {".safetensors"})}), patch.dict(nodes.folder_paths.filename_list_cache, {}, clear=True):
                plan = {"type": "instructions", "input": {"type": "load", "filename": saved.name}, "description": "portrait", "subject_name": "Alice"}
                result = picker_routes.resolve_plan(plan)
                expected = nodes.SetInstructions.execute(core.load_pack(saved), "portrait", "Alice").result[0]
                self.assertEqual(core.source_catalog(result), core.source_catalog(expected))
                self.assertEqual(core.source_ids(result), core.source_ids(core.load_pack(saved)))

    def test_prompt_groups_interleaved_sources_without_changing_labels(self):
        alice_portrait = core.set_instructions(pack("portrait"), "her three-quarters portrait", "Alice")
        bob = core.set_instructions(pack("bob"), "his portrait", "Bob")
        alice_body = core.set_instructions(pack("body"), "her body shape", "Alice")
        combined = core.combine_many([(alice_portrait, 1), (bob, 1), (alice_body, 1)])
        ids = core.source_ids(combined)
        definitions, _ = core.reference_instructions(combined["entries"])
        self.assertEqual(definitions, ["<Subject 1> is Alice.", "<Picture 1> provides her three-quarters portrait.", "<Picture 3> provides her body shape.", "<Subject 2> is Bob.", "<Picture 2> provides his portrait."])
        self.assertEqual(core.source_ids(combined), ids)
        self.assertEqual([entry["description"] for entry in combined["entries"]], ["her three-quarters portrait", "his portrait", "her body shape"])
        prompt = core.build_prompt(combined["entries"], "[Summary]\nAlice and Bob wave.", True)
        self.assertNotIn("for <Subject", prompt)
        self.assertLess(prompt.index("<Picture 3>"), prompt.index("<Subject 2>"))

    def test_uniform_combine_order_and_strength(self):
        original = pack()
        combined = nodes.Combine.execute(refmods={"refmod_10": audio_pack(), "refmod_3": video_pack(), "refmod_2": pack("beth"), "refmod_1": original}, strengths='{"refmod_2":0.5,"refmod_3":0,"refmod_10":0.25}').result[0]
        self.assertEqual([entry["name"] for entry in combined["entries"]], ["alice", "beth", "motion", "voice"])
        self.assertEqual([entry["strength"] for entry in combined["entries"]], [1, 0.5, 0, 0.25])
        self.assertEqual(original["entries"][0]["strength"], 1)
        self.assertEqual(nodes.Combine.execute().result[0]["entries"], [])
        for strengths in ('[]', 'invalid', '{"refmod_1":3}', '{"refmod_1":"1"}', '{"refmod_0":1}'):
            with self.assertRaises(ValueError):
                nodes.Combine.execute(refmods={"refmod_1": pack()}, strengths=strengths)
        for name in ("refmod_0", "refmod_101", "refmod_01", "refmod_a"):
            with self.assertRaises(ValueError):
                nodes.Combine.execute(refmods={name: pack()})
        self.assertEqual(len(nodes.Combine.execute(refmods={f"refmod_{i}": pack() for i in range(1, 101)}).result[0]["entries"]), 100)

    def test_native_autogrow_schema_parses_uniform_packs(self):
        from comfy_api.latest import _io
        live = {"refmods.refmod_1": pack(), "strengths": '{"refmod_1":0.5}',
            "refmods.refmod_2": video_pack(), "refmods.refmod_12": audio_pack()}
        schema, _, v3_data = _io.get_finalized_class_inputs(nodes.Combine.INPUT_TYPES(), live)
        self.assertIn("refmods.refmod_12", schema["optional"])
        nested = _io.build_nested_inputs(live, v3_data)
        self.assertEqual(set(nested["refmods"]), {"refmod_1", "refmod_2", "refmod_12"})
        result = nodes.Combine.execute(**nested).result[0]
        self.assertEqual([core.kind(entry) for entry in result["entries"]], ["image", "video", "audio"])
        self.assertEqual(result["entries"][0]["strength"], 0.5)

    def test_expanding_combine_preview_matches_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            saved = [core.save_pack(source, directory, f"source{i}") for i, source in enumerate((pack(), video_pack(), audio_pack()))]
            with patch.dict(nodes.folder_paths.folder_names_and_paths, {nodes.FOLDER: ([directory], {".safetensors"})}), patch.dict(nodes.folder_paths.filename_list_cache, {}, clear=True):
                plan = {"type": "combine", "inputs": [{"input": {"type": "load", "filename": path.name}, "strength": strength} for path, strength in zip(saved, (1, 0, 0.5))]}
                result = picker_routes.resolve_plan(plan)
                expected = nodes.Combine.execute(refmods={"refmod_1": pack(), "refmod_2": video_pack(), "refmod_3": audio_pack()}, strengths='{"refmod_2":0,"refmod_3":0.5}').result[0]
                self.assertEqual(core.source_ids(result), core.source_ids(expected))
                self.assertEqual(core.describe(result), core.describe(expected))
                legacy = picker_routes.resolve_plan({"type": "combine", "a": {"type": "load", "filename": saved[0].name}, "b": {"type": "load", "filename": saved[1].name}, "strength_a": 1, "strength_b": 1})
                self.assertEqual(len(legacy["entries"]), 2)

    def test_mixed_roundtrip_and_v1_compatibility(self):
        mixed = core.combine(core.combine(pack(), video_pack(True), 1, 1), audio_pack(), 1, 1)
        core.validate(mixed)
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = core.save_pack(mixed, directory, "mixed")
            loaded = core.load_pack(path)
            self.assertEqual(core.source_ids(mixed), core.source_ids(loaded))
            for a, b in zip(mixed["entries"], loaded["entries"]):
                for field in core.tensor_fields(a):
                    self.assertTrue(torch.equal(a[field], b[field]))
            old = pack()["entries"][0]
            save_file({"ref.0.latent": old["latent"], "ref.0.jpeg": old["jpeg"]}, str(Path(directory) / "v1.safetensors"),
                metadata={"format": core.FORMAT, "version": "1", "manifest": json.dumps([{k: v for k, v in old.items() if k not in ("latent", "jpeg")}])})
            legacy = core.load_pack(Path(directory) / "v1.safetensors")
            self.assertEqual(core.source_ids(legacy), core.source_ids(pack()))

    def test_multimodal_presentation_numbering_selection_and_layout(self):
        mixed = core.combine(core.combine(audio_pack(), video_pack(True), 1, 1), pack(), 1, 1)
        items = core.text_items(mixed["entries"])
        self.assertEqual([item["type"] for item in items], ["audio", "audio", "video", "image"])
        self.assertEqual(items[2]["timestamps"], [0, 0.5])
        self.assertEqual(tuple(items[2]["data"].shape), (2, 64, 96, 3))
        self.assertIn("<Audio 2> + <Video 1> = motion", core.describe(mixed))
        subset = core.select_sources(mixed, json.dumps(core.source_ids(mixed)[1:]))
        self.assertIn("<Audio 1> + <Video 1> = motion", core.describe(subset))
        from comfy.ldm.minimax.model import PackedLayout, patchify_video, pack_audio
        refs = core.blocks(mixed["entries"])
        layout = PackedLayout(3, 7, 4, 6, 37, refs=refs)
        ref_rows = sum(stop - start for start, stop, label in layout.segments if label in ("ref_img", "ref_audio"))
        self.assertEqual(ref_rows, core.token_count(mixed["entries"]))
        for entry in mixed["entries"]:
            actual = 0
            if "latent" in entry:
                actual += patchify_video(entry["latent"]).shape[0]
            if "audio_latent" in entry:
                actual += pack_audio(entry["audio_latent"]).shape[0]
            self.assertEqual(actual, core.token_count([entry]))

    def test_mixed_text_encoder_and_existing_budget(self):
        class Clip:
            def tokenize(self, prompt, minimax_ref_items):
                self.items = minimax_ref_items
                return prompt
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros(1), {}]]
        clip = Clip()
        mixed = core.combine(video_pack(True), audio_pack(), 1, 1)
        cond = nodes.TextEncode.execute(clip, mixed, "test", 8192).result[0]
        self.assertEqual([item["type"] for item in clip.items], ["audio", "video", "audio"])
        self.assertEqual([block["kind"] for block in cond[0][1]["minimax_refs"]], ["video_audio", "audio"])
        with self.assertRaisesRegex(ValueError, "budget"):
            nodes.Apply.execute(pack(), cond, 1, core.token_count(mixed["entries"]))
        result = nodes.Apply.execute(pack(), cond, 1, 8192).result[0]
        self.assertEqual(len(result[0][1]["minimax_refs"]), 3)

    def test_audio_and_video_invalid_payloads(self):
        bad = audio_pack()
        bad["entries"][0]["audio_latent"] = torch.ones(1, 32, 1, 80)
        with self.assertRaisesRegex(ValueError, "stereo"):
            core.validate(bad)
        bad = video_pack()
        bad["entries"][0]["timestamps"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "timestamps"):
            core.validate(bad)
        bad = video_pack()
        bad["entries"][0]["video_offsets"][-1] -= 1
        with self.assertRaisesRegex(ValueError, "offsets"):
            core.validate(bad)
        a, b = video_pack(), video_pack()
        b["entries"][0]["latent"][0, 0, 0, 0, 0] = 2
        self.assertNotEqual(core.source_ids(a), core.source_ids(b))

    def test_video_time_sampling_alignment_and_paired_audio(self):
        class Visual:
            first_stage_model = nodes.MiniMaxH3VideoVAE.__new__(nodes.MiniMaxH3VideoVAE)
            calls = 0
            def encode(self, frames):
                self.calls += 1
                self.frames = frames
                return torch.ones(1, 24, nodes.video_latent_t(frames.shape[0]), frames.shape[1] // 16, frames.shape[2] // 16)
        class Audio:
            first_stage_model = nodes.MiniMaxH3AudioVAE.__new__(nodes.MiniMaxH3AudioVAE)
            audio_sample_rate = 32000
            def encode(self, waveform):
                self.waveform = waveform
                return torch.ones(1, 32, 2, round(waveform.shape[1] / 800))
        vae, audio_vae = Visual(), Audio()
        frames = torch.arange(90).float().view(-1, 1, 1, 1).expand(-1, 64, 96, 3) / 100
        audio = {"waveform": torch.zeros(1, 1, 96000), "sample_rate": 32000}
        kwargs = dict(vae=vae, frames=frames, source_fps=30, subject_name="clip", description="",
            start_seconds=0.5, duration_seconds=1, max_edge=768, max_tokens=8192, audio=audio, audio_vae=audio_vae)
        created = nodes.CreateVideo.execute(**kwargs).result[0]
        entry = created["entries"][0]
        self.assertEqual(entry["frame_count"], 22)
        self.assertEqual(entry["kind"], "video_audio")
        self.assertAlmostEqual(float(vae.frames[0, 0, 0, 0]), 0.15, delta=1 / 255)
        self.assertAlmostEqual(float(vae.frames[-1, 0, 0, 0]), 0.41, delta=1 / 255)
        self.assertEqual(audio_vae.waveform.shape[2], 2)
        vae.calls = 0
        with self.assertRaisesRegex(ValueError, "budget"):
            nodes.CreateVideo.execute(**dict(kwargs, max_tokens=1))
        self.assertEqual(vae.calls, 0)
        with self.assertRaisesRegex(ValueError, "cover"):
            nodes.CreateVideo.execute(**dict(kwargs, audio={"waveform": torch.zeros(1, 2, 24000), "sample_rate": 32000}))

    def test_standalone_audio_trim_and_mono_conversion(self):
        class Audio:
            first_stage_model = nodes.MiniMaxH3AudioVAE.__new__(nodes.MiniMaxH3AudioVAE)
            audio_sample_rate = 32000
            def encode(self, waveform):
                return torch.ones(1, 32, 2, round(waveform.shape[1] / 800))
        source = {"waveform": torch.ones(1, 1, 64000) * 0.1, "sample_rate": 32000}
        result = nodes.CreateAudio.execute(Audio(), source, "voice", "", 0.5, 1, 8192).result[0]
        self.assertEqual(core.token_count(result["entries"]), 80)
        self.assertEqual(result["entries"][0]["duration_seconds"], 1)
        self.assertEqual(core.text_items(result["entries"]), [{"type": "audio"}])
        with self.assertRaisesRegex(ValueError, "empty"):
            nodes.CreateAudio.execute(Audio(), source, "voice", "", 3, 1, 8192)

    def test_cached_comfy_ui_preserves_source_ids(self):
        from comfy_execution.asset_enrichment import register_cached_outputs
        catalog = core.source_catalog(pack())
        cached = register_cached_outputs({"output": {"sources": catalog}}, "synthetic-test", SimpleNamespace(enabled=False))
        self.assertEqual(cached["output"]["sources"][0]["source_id"], core.source_ids(pack())[0])

    def test_source_selection_order_identity_and_no_mutation(self):
        original = core.combine(pack("alice"), pack("beth"), 1, 1)
        ids = core.source_ids(original)
        subset = core.select_sources(original, json.dumps([ids[1]]))
        self.assertEqual(len(subset["entries"]), 1)
        self.assertIs(subset["entries"][0]["latent"], original["entries"][1]["latent"])
        self.assertEqual(subset["entries"][0]["name"], "beth")
        self.assertEqual(len(original["entries"]), 2)
        self.assertEqual(core.token_count(subset["entries"]), 6)
        self.assertIn("<Picture 1> = beth", core.describe(subset))
        self.assertEqual(len(core.select_sources(original, "all")["entries"]), 2)
        self.assertEqual(len(core.select_sources(original, json.dumps(list(reversed(ids))))["entries"]), 2)
        self.assertEqual(core.select_sources(original, json.dumps(list(reversed(ids))))["entries"][0]["name"], "alice")
        duplicate = core.combine(pack(), pack(), 1, 1)
        self.assertEqual(len(set(core.source_ids(duplicate))), 2)

    def test_source_ids_survive_save_reload_and_strength(self):
        original = pack()
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            loaded = core.load_pack(core.save_pack(original, directory, "ids"))
            self.assertEqual(core.source_ids(original), core.source_ids(loaded))
            loaded["entries"][0]["strength"] = 0.5
            self.assertEqual(core.source_ids(original), core.source_ids(loaded))

    def test_stale_selector_publishes_current_sources_and_blocks_outputs(self):
        from comfy_execution.graph_utils import ExecutionBlocker
        original = pack("old")
        replacement = pack("new")
        saved_selection = json.dumps(core.source_ids(original))
        result = nodes.SelectSources.execute(replacement, saved_selection)
        self.assertTrue(all(isinstance(value, ExecutionBlocker) for value in result.result))
        self.assertEqual(result.ui["sources"], core.source_catalog(replacement))
        self.assertIn("Upstream sources changed", result.ui["text"][0])
        recovered = nodes.SelectSources.execute(replacement, "all")
        self.assertEqual(len(recovered.result[0]["entries"]), 1)
        recovered = nodes.SelectSources.execute(replacement, json.dumps(core.source_ids(replacement)))
        self.assertEqual(core.source_ids(recovered.result[0]), core.source_ids(replacement))
        # Partial overlap still requires explicit confirmation, not an implicit fallback.
        combined = core.combine_many([(original, 1), (replacement, 1)])
        stale = json.dumps(core.source_ids(original) + ["missing"])
        blocked = nodes.SelectSources.execute(combined, stale)
        self.assertIsInstance(blocked.result[0], ExecutionBlocker)
        for selection in ("invalid", "{}", "[1]"):
            with self.assertRaises(ValueError):
                nodes.SelectSources.execute(replacement, selection)

    def test_selection_stale_and_invalid(self):
        for selection in ('["missing"]', 'invalid', '{}', '[1]'):
            with self.assertRaises(ValueError):
                core.select_sources(pack(), selection)

    def test_empty_selection_all_downstream_paths(self):
        empty = nodes.SelectSources.execute(pack(), "[]").result[0]
        empty = nodes.SetInstructions.execute(empty, "her portrait", subject_name="Alice").result[0]
        self.assertEqual(empty["entries"], [])
        self.assertEqual(core.combine(empty, pack("beth"), 1, 1)["entries"][0]["name"], "beth")
        class Clip:
            def tokenize(self, prompt, minimax_ref_items):
                self.items = minimax_ref_items
                return prompt
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros(1), {}]]
        clip = Clip()
        result = nodes.TextEncode.execute(clip, empty, "A quiet room.", 8192).result[0]
        self.assertEqual(clip.items, [])
        self.assertEqual(result[0][1]["minimax_refs"], [])
        self.assertEqual(nodes.Apply.execute(empty, result, 1, 8192).result[0][0][1]["minimax_refs"], [])
        self.assertEqual(tuple(nodes.Inspect.execute(empty).result[0].shape), (0, 256, 256, 3))
        with self.assertRaises(ValueError):
            core.validate(empty)
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            with self.assertRaisesRegex(ValueError, "at least one"):
                core.save_pack(empty, directory, "empty")

    def test_selector_filters_both_qwen_and_latents(self):
        combined = core.combine(pack("alice"), pack("beth"), 1, 1)
        subset = nodes.SelectSources.execute(combined, json.dumps([core.source_ids(combined)[1]])).result[0]
        class Clip:
            def tokenize(self, prompt, minimax_ref_items):
                self.items = minimax_ref_items
                return prompt
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros(1), {}]]
        clip = Clip()
        cond, mapping, final_prompt = nodes.TextEncode.execute(clip, subset, "Beth", 8192).result
        self.assertEqual(len(clip.items), 1)
        self.assertEqual(len(cond[0][1]["minimax_refs"]), 1)
        self.assertIn("<Picture 1> = beth", mapping)
        self.assertNotIn("alice", mapping)
        catalog = core.source_catalog(combined)
        self.assertEqual(len(catalog), 2)
        self.assertEqual([source["source_id"] for source in catalog], core.source_ids(combined))
        self.assertTrue(catalog[0]["thumbnail"].startswith("data:image/jpeg;base64,"))

    def test_preview_route_selected_files_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = core.save_pack(pack(), directory, "preview")
            with patch.dict(nodes.folder_paths.folder_names_and_paths, {nodes.FOLDER: ([directory], {".safetensors"})}), patch.dict(nodes.folder_paths.filename_list_cache, {}, clear=True):
                result = picker_routes.resolve_plan({"type": "load", "filename": path.name})
                self.assertEqual(core.source_ids(result), core.source_ids(pack()))
                result = picker_routes.resolve_plan({"type": "select", "input": {"type": "load", "filename": path.name}, "selection": "[]"})
                self.assertEqual(result["entries"], [])
                for filename in ('../other.safetensors', 'C:\\private.safetensors'):
                    with self.assertRaises(ValueError):
                        picker_routes.resolve_plan({"type": "load", "filename": filename})
                class Request:
                    async def json(self):
                        return {"type": "load", "filename": path.name}
                response = asyncio.run(picker_routes.preview_sources(Request()))
                self.assertEqual(response.status, 200)
                self.assertEqual(len(json.loads(response.text)["sources"]), 1)

    def test_embedded_workflow_and_api_prompt(self):
        workflow = {"version": 0.4, "nodes": [{"id": 1, "title": "CrÃ©ation"}], "links": []}
        prompt = {"1": {"class_type": "H3RefModLabCreate", "inputs": {"name": "alice"}}}
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            saved = core.save_pack(pack(), directory, "embedded", workflow=workflow, prompt=prompt)
            with safe_open(str(saved), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            header_bytes = saved.read_bytes()[:4 * 1024 * 1024]
            header_size = int.from_bytes(header_bytes[:4], "little")
            frontend_metadata = json.loads(header_bytes[8:8 + header_size])["__metadata__"]
            self.assertEqual(json.loads(frontend_metadata["workflow"]), workflow)
            self.assertEqual(json.loads(metadata["workflow"]), workflow)
            self.assertEqual(json.loads(metadata["prompt"]), prompt)
            self.assertTrue(torch.equal(core.load_pack(saved)["entries"][0]["latent"], pack()["entries"][0]["latent"]))
            without = core.save_pack(pack(), directory, "plain")
            with safe_open(str(without), framework="pt", device="cpu") as handle:
                self.assertNotIn("workflow", handle.metadata())
                self.assertNotIn("prompt", handle.metadata())

    def test_save_node_embedding_toggle_and_disable_metadata(self):
        hidden = SimpleNamespace(prompt={"1": {"class_type": "test"}}, extra_pnginfo={"workflow": {"nodes": [], "links": []}})
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            with patch.object(nodes.Save, "hidden", hidden), patch.dict(nodes.folder_paths.folder_names_and_paths, {nodes.FOLDER: ([directory], {".safetensors"})}):
                for enabled, globally_disabled, expected in ((True, False, True), (False, False, False), (True, True, False)):
                    with patch.object(nodes.args, "disable_metadata", globally_disabled):
                        result = nodes.Save.execute(pack(), "saved", enabled).result
                    with safe_open(result[1], framework="pt", device="cpu") as handle:
                        self.assertEqual("workflow" in handle.metadata(), expected)
                        self.assertEqual("prompt" in handle.metadata(), expected)

    def test_roundtrip_and_no_overwrite(self):
        original = pack()
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            one = core.save_pack(original, directory, "alice")
            two = core.save_pack(original, directory, "alice")
            self.assertNotEqual(one, two)
            loaded = core.load_pack(one)
            self.assertTrue(torch.equal(loaded["entries"][0]["latent"], original["entries"][0]["latent"]))
            self.assertTrue(torch.equal(loaded["entries"][0]["jpeg"], original["entries"][0]["jpeg"]))
            self.assertEqual(loaded["entries"][0]["description"], "reference description")
            self.assertEqual(len(list(Path(directory).glob("*.safetensors"))), 2)

    def test_file_numbers_per_name_and_survive_deletion(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            one = core.save_pack(pack(), directory, "first")
            two = core.save_pack(pack(), directory, "second")
            self.assertEqual(one.name, "first_00001.safetensors")
            self.assertEqual(two.name, "second_00001.safetensors")
            one.unlink()
            two.unlink()
            self.assertEqual(core.save_pack(pack(), directory, "first").name, "first_00002.safetensors")
            self.assertEqual(core.save_pack(pack(), directory, "second").name, "second_00002.safetensors")

    def test_file_numbers_seed_existing_folder_and_imports_and_expand(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            root = Path(directory)
            (root / "old_name_00042.safetensors").touch()
            self.assertEqual(core.save_pack(pack(), root, "new").name, "new_00001.safetensors")
            (root / "old_name_00042.safetensors").unlink()
            self.assertEqual(core.save_pack(pack(), root, "old_name").name, "old_name_00043.safetensors")
            (root / "imported_99999.safetensors").touch()
            self.assertEqual(core.save_pack(pack(), root, "new").name, "new_00002.safetensors")
            (root / "new_99999.safetensors").touch()
            self.assertEqual(core.save_pack(pack(), root, "new").name, "new_100000.safetensors")

    def test_file_numbers_upgrade_legacy_shared_counter(self):
        import sqlite3
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            root = Path(directory)
            with sqlite3.connect(root / ".h3-refmods-counter.sqlite3") as database:
                database.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, last INTEGER)")
                database.execute("INSERT INTO counter VALUES (1, 400)")
            database.close()
            (root / "existing_00004.safetensors").touch()
            self.assertEqual(core.save_pack(pack(), root, "fresh").name, "fresh_00001.safetensors")
            self.assertEqual(core.save_pack(pack(), root, "existing").name, "existing_00005.safetensors")

    def test_file_numbers_case_insensitive_and_exact_prefix(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            self.assertEqual(core.save_pack(pack(), directory, "Alice").name, "Alice_00001.safetensors")
            self.assertEqual(core.save_pack(pack(), directory, "alice").name, "alice_00002.safetensors")
            self.assertEqual(core.save_pack(pack(), directory, "alice_extra").name, "alice_extra_00001.safetensors")

    def test_concurrent_file_number_reservations_are_unique(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            with ThreadPoolExecutor(max_workers=8) as executor:
                numbers = list(executor.map(lambda _: core._next_file_number(Path(directory), "shared"), range(24)))
            self.assertEqual(sorted(numbers), list(range(1, 25)))
            self.assertEqual(core._next_file_number(Path(directory), "shared"), 25)

    def test_failed_publication_does_not_reuse_number(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            with patch.object(core.os, "link", side_effect=OSError("Synthetic publication failure")):
                with self.assertRaises(OSError):
                    core.save_pack(pack(), directory, "test")
            self.assertEqual(core.save_pack(pack(), directory, "test").name, "test_00002.safetensors")
            self.assertEqual(list(Path(directory).glob(".refmod-*.tmp")), [])

    def test_foreign_format_and_bad_shape(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            file = Path(directory) / "foreign.safetensors"
            save_file({"latent": torch.ones(1)}, file)
            with self.assertRaisesRegex(ValueError, "not an H3"):
                core.load_pack(file)
        bad = pack()
        bad["entries"][0]["latent"] = torch.ones(1, 16, 1, 4, 6)
        with self.assertRaisesRegex(ValueError, "Expected H3"):
            core.validate(bad)
        bad = pack()
        bad["entries"][0]["latent"][0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            core.validate(bad)

    def test_filename_traversal(self):
        for name in ("../other", "C:\\other", "folder/name", "", "bad.name"):
            with self.assertRaisesRegex(ValueError, "Filename"):
                core.save_pack(pack(), ROOT / "artifacts", name)

    def test_budget_and_geometry(self):
        self.assertEqual(core.token_count(pack()["entries"]), 6)
        with self.assertRaisesRegex(ValueError, "Nothing was dropped"):
            core.check_budget(pack()["entries"], 5)
        self.assertEqual(core.reference_size(1920, 1080, 768), (768, 448))
        self.assertEqual(core.reference_size(64, 96, 768), (64, 96))

    def test_combining_strength_and_reference_order(self):
        a, b = pack(), pack("beth")
        result = core.combine(a, b, 0, 0.5)
        self.assertEqual(core.active_entries(result)[0]["name"], "beth")
        self.assertIn("<Picture 1> = beth", core.describe(result))
        self.assertNotIn("<Picture 2>", core.describe(result))
        self.assertEqual(b["entries"][0]["strength"], 1)
        self.assertTrue(torch.equal(core.blocks(core.active_entries(result))[0]["latent"], b["entries"][0]["latent"] * 0.5))

    def test_append_preserves_every_conditioning_entry(self):
        tensor = torch.zeros(1)
        old_ref = {"kind": "image", "latent": torch.ones(1, 24, 1, 2, 2)}
        source = [[tensor, {"minimax_refs": [old_ref], "minimax_keyframes": ["guide"], "tag": 1}], [tensor, {"tag": 2}]]
        result = core.append_conditioning(source, pack()["entries"])
        self.assertEqual(len(result[0][1]["minimax_refs"]), 2)
        self.assertEqual(len(result[1][1]["minimax_refs"]), 1)
        self.assertEqual(result[0][1]["minimax_keyframes"], ["guide"])
        self.assertEqual(len(source[0][1]["minimax_refs"]), 1)

    def test_text_encoder_receives_images_and_attaches_once(self):
        class Clip:
            def tokenize(self, prompt, minimax_ref_items):
                self.items = minimax_ref_items
                self.prompt = prompt
                return "tokens"
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros(1), {"minimax_token_tags": "kept"}]]
        clip = Clip()
        refs = core.combine(pack(), pack("beth"), 1, 0)
        result = nodes.TextEncode.execute(clip, refs, "The person in <Picture 1> waves.", 8192).result
        self.assertEqual(len(clip.items), 1)
        self.assertEqual(tuple(clip.items[0]["data"].shape), (1, 64, 96, 3))
        self.assertEqual(len(result[0][0][1]["minimax_refs"]), 1)
        self.assertEqual(result[0][0][1]["minimax_token_tags"], "kept")

    def test_apply_counts_existing_refs_and_zero_strength(self):
        source = [[torch.zeros(1), {"minimax_refs": core.blocks(pack()["entries"])}]]
        with self.assertRaisesRegex(ValueError, "budget"):
            nodes.Apply.execute(pack(), source, 1, 6)
        result = nodes.Apply.execute(pack(), source, 0, 6).result[0]
        self.assertEqual(len(result[0][1]["minimax_refs"]), 1)

    def test_create_independent_images_and_budget_before_encode(self):
        class VAE:
            first_stage_model = nodes.MiniMaxH3VideoVAE.__new__(nodes.MiniMaxH3VideoVAE)
            calls = 0
            def encode(self, images):
                self.calls += 1
                return torch.ones(1, 24, 1, images.shape[1] // 16, images.shape[2] // 16)
        vae = VAE()
        images = torch.full((2, 64, 96, 3), 0.5)
        with self.assertRaisesRegex(ValueError, "budget"):
            nodes.Create.execute(vae, images, "alice", "", 768, 11)
        self.assertEqual(vae.calls, 0)
        result = nodes.Create.execute(vae, images, "alice", "", 768, 12).result[0]
        self.assertEqual(vae.calls, 2)
        self.assertNotIn("vae_label", result["entries"][0])
        self.assertEqual(len(result["entries"]), 2)
        self.assertEqual(result["entries"][0]["subject_id"], result["entries"][1]["subject_id"])

    def test_thumbnails_and_schemas(self):
        self.assertEqual(tuple(core.thumbnails(pack()).shape), (1, 256, 256, 3))
        schemas = {}
        for cls in nodes.NODE_CLASSES:
            schema = cls.GET_SCHEMA()
            schemas[schema.node_id] = cls.INPUT_TYPES()
        (ROOT / "artifacts/refmod_node_inputs.json").write_text(json.dumps(schemas, indent=2), encoding="utf-8")
        self.assertEqual(len(schemas), 11)

    def test_example_workflow_links_and_types(self):
        import nodes as comfy_nodes

        async def check():
            for module in ("nodes_minimax_h3.py", "nodes_custom_sampler.py", "nodes_model_advanced.py", "nodes_audio.py", "nodes_video.py", "nodes_resolution.py", "nodes_primitive.py", "nodes_preview_any.py"):
                self.assertTrue(await comfy_nodes.load_custom_node(str(COMFY / "comfy_extras" / module), module_parent="comfy_extras"))
            known_types = set(comfy_nodes.NODE_CLASS_MAPPINGS)
            known_types.update(cls.GET_SCHEMA().node_id for cls in nodes.NODE_CLASSES)
            known_types.add("Reroute")  # Frontend-only passthrough.
            paths = sorted((ROOT / "workflows").glob("*.json"))
            self.assertEqual([path.name for path in paths], ["01_create_refmod_Alice.json", "02_generate_with_refmod_Alice.json"])
            for path in paths:
                workflow = json.loads(path.read_text())
                subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
                graph_types = known_types | {graph["id"] for graph in subgraphs}
                for graph in [workflow, *subgraphs]:
                    graph_nodes = {node["id"]: node for node in graph["nodes"]}
                    self.assertEqual(len(graph_nodes), len(graph["nodes"]), path.name)
                    links = {}
                    for raw in graph["links"]:
                        link = raw if isinstance(raw, dict) else dict(zip(("id", "origin_id", "origin_slot", "target_id", "target_slot", "type"), raw))
                        self.assertNotIn(link["id"], links, path.name)
                        links[link["id"]] = link
                        if link["origin_id"] == -10:
                            self.assertIn(graph, subgraphs)
                            self.assertIn(link["id"], graph["inputs"][link["origin_slot"]]["linkIds"])
                        else:
                            source = graph_nodes[link["origin_id"]]["outputs"][link["origin_slot"]]
                            self.assertIn(link["id"], source.get("links") or [], path.name)
                        if link["target_id"] == -20:
                            self.assertIn(graph, subgraphs)
                            self.assertIn(link["id"], graph["outputs"][link["target_slot"]]["linkIds"])
                        else:
                            target = graph_nodes[link["target_id"]]["inputs"][link["target_slot"]]
                            self.assertEqual(target["link"], link["id"], path.name)
                    for node in graph["nodes"]:
                        self.assertIn(node["type"], graph_types, (path.name, node["type"]))
                        for slot, port in enumerate(node.get("inputs", [])):
                            if port.get("link") is not None:
                                link = links[port["link"]]
                                self.assertEqual((link["target_id"], link["target_slot"]), (node["id"], slot), path.name)
                        for slot, port in enumerate(node.get("outputs", [])):
                            for link_id in port.get("links") or []:
                                link = links[link_id]
                                self.assertEqual((link["origin_id"], link["origin_slot"]), (node["id"], slot), path.name)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)

