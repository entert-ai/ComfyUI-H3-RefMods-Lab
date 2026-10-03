# H3 RefMods Lab

Image, video and audio reference packs for experimenting in ComfyUI.
Uses ComfyUI's native MiniMax H3 visual/audio VAEs and reference conditioning. No training,
model patches, cloud calls or weight downloads.

## Install and start

Clone this repository into your ComfyUI `custom_nodes` folder:

```text
git clone https://github.com/entert-ai/ComfyUI-H3-RefMods-Lab.git
```

Alternatively copy this folder into your ComfyUI `custom_nodes` folder. Dependencies
are already present in a standard ComfyUI installation. Restart ComfyUI and refresh
its browser page. Nodes appear under **MiniMax H3 / RefMods Lab**.

Drag the **UI JSON** files from `workflows/` into ComfyUI. Files ending `.api.json`
are companion API graphs, not canvas workflows. Choose your own reference files;
placeholder filenames deliberately contain no personal content.

1. **01_create_refmod.json** — choose an image, name and description. Select the
   H3 visual VAE. Queue to encode, inspect
   and save. Saves go to `ComfyUI/models/h3_refmods_lab`, with numbered filenames
   that never overwrite existing packs. Source images are embedded in the pack.
   **Embed workflow** is on by default: drag a newly saved `.safetensors` onto
   ComfyUI to restore its creation graph, just like a generated image. This saves
   the graph used at creation time, not a later generation workflow. Original
   Load Image files must still be available to rerun that graph; embedded JPEGs
   remain usable through Load RefMod even without the original files. The graph
   includes its filenames, prompts and settings. Turn the option off if unwanted;
   ComfyUI's `--disable-metadata` flag also disables embedding. API-only saves
   contain the API prompt but need `extra_pnginfo.workflow` for the canvas graph.
   Filename suffixes use a persistent counter per name within the save folder:
   `subject_00001.safetensors`, `voice_00001.safetensors`, then
   `subject_00002.safetensors`, and so on. Deleting
   old packs does not reuse their numbers. The hidden `.h3-refmods-counter.sqlite3`
   file retains the counter; include it when backing up or moving the folder.
   An existing folder seeds each name from its highest numbered filename. Names
   differing only in letter case share one counter. Upgrading from the former
   shared counter preserves existing filenames; new names start at 00001. That
   old shared counter cannot reconstruct the names of previously deleted files.
   Failed saves may leave harmless gaps. Five digits are minimum padding: after
   99999 the number becomes 100000. Saves still never overwrite existing files.
2. Refresh ComfyUI's node/model definitions (reload the browser if needed), then
   load **02_generate_with_refmod.json**. Select the saved pack and check the
   Inspect node's Picture map. Edit the prompt's subject definitions accordingly.
   On **Select H3 RefMod Sources**, click **Refresh sources** to show thumbnails,
   then use the checkboxes to include or exclude individual sources. A video and
   its paired soundtrack are one selectable source. **Select all**
   and **Clear all** are available. The selected count and active DiT token count
   update immediately, without VAE encoding. Checkboxes are saved in the workflow;
   the original pack stays intact. Refresh again after changing an upstream pack.
3. **03_two_refmods.json** provides two loaders and independent strength controls.
   Its example assumes one picture per pack. A pack containing three images uses
   Picture 1–3, so the next pack starts at Picture 4. The Inspect map is authoritative.
4. **04_latents_only.json** is an alternate experimental path: Qwen sees only the
   text, and Apply appends saved latents to native H3 conditioning.
5. **05_create_video_refmod.json** — Load Video → Get Video Components → Create H3
   Video RefMod. Connect the actual source fps, visual VAE and optionally the
   soundtrack plus H3 audio VAE. Disconnect audio to create a silent video reference.
   Default: first 3 seconds, max_edge 512. H3 frame alignment trims this to 56 frames
   (~2.33 seconds). Inspect reports the actual interval. This is one temporal
   reference, not many pictures. Combine with existing packs before Save to add it.
6. **06_create_audio_refmod.json** — Load Audio → Create H3 Audio RefMod using the
   H3 audio VAE. Default: first 3 seconds. Combine with a subject's visual pack to
   store appearance and voice together. Describe the voice's role explicitly in
   the generation prompt using Inspect's Audio labels.
7. **07_combine_refmods.json** — four packs feeding one expanding Combine node.
   Inputs are numbered from 1. Connect another pack to the next empty socket;
   ComfyUI adds another automatically, up to 100 inputs total. Every connected
   pack has the same strength slider, saved with the workflow. Unconnected sockets
   are ignored; reference order follows input numbers. Refresh sources supports
   the whole combined set.

8. **08_saved_instructions.json** — two packs supplying Alice's portrait and body shape.
   Set Instructions attaches matching subject names and separate retention details;
   Text Encode builds the reference instructions. Save the edited packs to persist them.

The generation workflows use installed H3 Ref2VA INT8 weights, H3 Qwen3-VL NVFP4,
both VAEs, Comfy Kitchen attention, native sol-attn, Euler/simple, 20 steps and
fixed seed 42. Initial output is 768×448 at 24 fps and 124 frames (~5.17 seconds).
Adjust model filenames if using another installation. Generation workflow 02 also
accepts mixed packs; revise its image-only example prompt to match the actual
Picture, Video and Audio labels. The output workflows decode and save audio.

## Nodes

| Node | Behaviour |
| --- | --- |
| Create H3 RefMod | Sequentially encode every input image as a separate picture. |
| Create H3 Video RefMod | Encode one frame batch at 24 fps, optionally with its synchronized soundtrack. |
| Create H3 Audio RefMod | Encode one mono/stereo standalone audio interval. |
| Save H3 RefMod | Save a numbered `.safetensors` pack and return its exact path. |
| Load H3 RefMod | Load a Lab v1 or v2 pack on CPU; changed files invalidate its cache. |
| Combine H3 RefMods | Combine up to 100 packs through expanding inputs, with individual strengths; uniform inputs numbered from 1 and matching sliders for every connected pack. |
| Select H3 RefMod Sources | Choose a subset using thumbnails and checkboxes; no re-encoding or file changes. |
| Apply H3 RefMod (Latents Only) | Append latents while preserving existing refs, guides and conditioning entries. |
| H3 RefMod Text Encode | Optionally generate saved reference instructions; present media to H3 Qwen, attach latents and output the final prompt. |
| Set H3 RefMod Instructions | Edit one subject's source descriptions and retention metadata without re-encoding. |
| Inspect H3 RefMod | Show active reference labels, resolution, VAE label and DiT token count; output thumbnails for Preview Image. |

**Text Encode already attaches the references. Do not also Apply the same pack.**
Text Encode creates fresh conditioning and does not merge an existing reference
workflow. Add native guides downstream if needed. Inspect's numbering describes
the pack's own Text Encode path; latent-only Apply cannot assign Qwen labels to
images Qwen has not seen.

**Combine node update:** Replace existing Combine nodes and reconnect their packs in older workflows. The inputs are now `refmod_1` through `refmod_100`, with matching 0–2 strength sliders. Existing saved RefMod files remain compatible. Updated example workflows use the new layout. API inputs use `refmods.refmod_1`, etc., and a `strengths` JSON object such as `{"refmod_1":1,"refmod_2":0.5}`.

Saved instructions are optional metadata, carried by each source inside the `.safetensors`
file. All Create nodes have one **Subject name** field, plus the source description
and retention controls. Sources with the same subject name group automatically,
ignoring capitalization and extra spaces. Use different names for different subjects,
e.g. Alice Smith and Alice Jones. A Create batch shares one subject; use separate
branches for different descriptions. Source labels such as Alice1 and Alice2 are
assigned automatically in pack order. They are display labels, not filenames or
reference tokens, and may renumber after selection/reordering. Internal selection
IDs remain unchanged. Save's filename field still controls the output filename.
Create nodes do not ask for VAE labels: the connected VAE determines encoding.
Existing VAE labels in saved packs remain readable.
Older saved packs remain readable; subject metadata is grouped by name even if
it contains former grouping keys. Old canvas node fields migrate when loaded. API callers should use `subject_name`;
the former `name` and `subject_key` inputs have been removed.

Enable **include_saved_instructions** in Text Encode to prepend `[Subject Definitions]`
and `[Retention Analysis]` to your scene prompt. It is off by default for existing
manual prompts. Remove competing manual sections when enabled; duplicate sections
produce an explicit error. `final_prompt` is the exact string sent to H3. Subject,
Picture, Video and Audio numbers follow the active sources after combining and
selection; zero strength omits a source. Check the reference map before writing
numbered references in your scene. Metadata descriptions/details should use plain
text rather than hard-coded reference numbers or ambiguous pronouns.

Generated source descriptions are grouped directly beneath their subject definition,
without a repeated `for <Subject N>` suffix. Start descriptions with **his/her/its**,
e.g. `her three-quarters portrait` or `its surface texture`. The text is used as written;
pronouns are not inferred. Subject grouping may put Picture 1 and Picture 3 together
before another subject's Picture 2. Their numbers still match the actual reference order.

Visual retention presets are `unspecified`, `fully_preserved`, `partially_preserved`,
`attribute_transfer`, and `weak_reference`. Audio uses `unspecified`, `fully_copy`,
`partially_copy`, `reference`, and `weak_reference`. Paired video/audio has separate
soundtrack retention controls. These are prompt instructions, not guaranteed outcomes
or changes to latent strength. Text grows slightly; latent tensors and DiT budgets
remain unchanged. Old packs load without subject metadata and can still contribute
source descriptions when enabled.

**Set H3 RefMod Instructions** edits an existing pack without a VAE. It replaces
metadata on every incoming source, assigning them one subject; use Select Sources
first for a subset or one subject from a combined pack. Empty description/details
clear those fields. For standalone audio in this editor, use the audio retention
controls. Save the output to persist the changes. Original packs remain intact.

Place a selector after each subject's Load node and before Combine or Text Encode.
Refresh supports saved packs through Load, Combine and other Select Sources nodes.
For an unsaved in-memory pack, queue only the selector first to populate its cards.
An empty selection omits that pack from both Qwen and the DiT; if every pack is
empty, Text Encode supplies text-only conditioning. Saving an empty pack is rejected.
Picture, Video and Audio numbers each close up after exclusions; prompts are not rewritten automatically.
Inspect after Combine shows the final map. Stable source IDs preserve explicit
selections across reloads and strength changes. A changed pack with missing selected
sources raises an error until you choose again. Preview reads only explicitly
connected Lab files and uses stored JPEGs, not the VAE or Qwen.

For multiple same-size pictures, connect an IMAGE batch to Create. Each batch
element remains one image, not a video frame. For different sizes, use separate
Load Image → Create branches sharing one VAE, Combine, then Save. The same
technique combines different subjects; names/descriptions are metadata and are
not automatic subject bindings or trigger words.

## Encoding and experiments

`max_edge` limits the longest side, with no deliberate enlargement except rounding
dimensions to H3's 32-pixel grid. Rounding can slightly change aspect ratio.
References retain full VAE latents; there is no pooling, refinement or latent
compression. Prepared source images are stored as quality-95 JPEG bytes for Qwen,
so reference presentation is slightly lossy while saved latent tensors are exact.
The VAE label is user-supplied provenance, not an automatically verified file hash.

Token budgets reject overflow before encoding/saving rather than dropping images.
One image uses `(width/32) × (height/32)` DiT tokens; budgets exclude Qwen vision
tokens and target-video tokens. `max_tokens=0` explicitly disables the budget.
Apply counts existing native image, video and audio references as well as new sources.
Video uses `latent_T × (width/32) × (height/32)` tokens; stereo audio uses
`2 × audio_latent_T` (about 80 tokens per second). Paired clips use their sum.

Strength 1 is the baseline. Other strengths scale latent values; they are an
experimental control, not a measured identity-retention percentage. Combine
multiplies existing strengths, with an allowed effective range of 0–2. Zero-strength
references are omitted from both Text Encode presentation and DiT conditioning,
and remaining Picture numbers close up. Inspect thumbnails include every stored
source, including inactive ones. Learned compression remains future work.
Existing community RefMods are rejected clearly: this is a distinct,
versioned Lab format, not a claim of cross-format compatibility.

For comparison, keep model, seed, prompt, dimensions, steps and attention settings
fixed. Compare ordinary native image references against Lab Text Encode first,
then compare the latent-only workflow. Use matching reference resize settings:
native `match` sizes by generation area, whereas Create sizes by `max_edge`.

## Video and audio caveats

- New packs use Lab v2; existing v1 image packs and image source selections still
  load. Older copies of these nodes cannot read v2 packs.
- Connect the source's real fps. Videos are sampled by time to 24 fps using the
  nearest preceding source frame, without optical flow. Low-fps input repeats
  frames. Variable-frame-rate sources should be normalized to constant fps first.
  This initial path decodes the source frame batch before trimming, so long source
  files can consume substantial RAM even when only a short interval is encoded.
- Video intervals trim down to `5+17k` frames and retain their original timing.
  Input controls allow up to 15 seconds; reference quality at very short durations
  has not been established. No hidden generation-length truncation occurs after
  loading a pack. The paired soundtrack must cover the retained video interval.
- Video adds temporal tokens at every diffusion step. For example, 56 frames at
  512×288 use 17×16×9 = 2,448 visual tokens; its soundtrack adds about 187 tokens.
  The same duration at 768×448 uses 5,712 visual tokens. Start with one short video;
  large reference sets may dominate other subjects and exceed VRAM during generation.
- H3 audio encoding uses its separate 32 kHz stereo VAE. Mono is duplicated to
  stereo. Packs retain exact audio latents, not the original recording. A waveform
  thumbnail identifies audio sources; in-node playback is not implemented.
- Qwen sees sampled video JPEGs at 2 fps with timestamps. It receives only labels
  for audio; no transcript is inferred or embedded. Describe which subject should
  use `<Audio 1>` and whether you want its voice, ambience or another sound property.
  A reference is conditioning, not a guarantee of verbatim audio reproduction or
  exact frame copying. Use native guides when a source must be anchored on the
  output timeline.
- Video/audio packs will be larger than small image packs. The full latent payload
  is retained; video also includes sparse JPEG frames for Qwen. Selecting a paired
  video currently includes/excludes its soundtrack together; create silent video
  and standalone audio packs if separate runtime control is needed.

Example bindings for a pack with one image, one video with soundtrack, and one
standalone voice reference:

```text
[Subject Definitions]
<Subject 1> is the person in <Picture 1>, with the voice of <Audio 2>.

[Retention Analysis]
Retain the appearance of <Subject 1> from <Picture 1>.
Use <Video 1> as the reference for motion.

[Soundscape]
<Subject 1> speaks in the voice of <Audio 2>.
```

The paired soundtrack has Audio 1 because its label precedes Video 1. Always use
Inspect's actual map rather than assuming labels from this example.

## Verification

From this repository, use ComfyUI's Python and pass your ComfyUI installation
directory explicitly. Replace the example installation path below with your own:

```powershell
& 'C:\path\to\ComfyUI\.venv\Scripts\python.exe' tests/test_refmods.py 'C:\path\to\ComfyUI'
& 'C:\path\to\ComfyUI\.venv\Scripts\python.exe' tools/smoke_refmod_vae.py 'C:\path\to\ComfyUI'
& 'C:\path\to\ComfyUI\.venv\Scripts\python.exe' tools/smoke_refmod_av.py 'C:\path\to\ComfyUI'
```

The CPU suite covers format round trips, no-overwrite saves, invalid inputs,
budgets, conditioning preservation, active label order, image presentation and
all seven graphs through ComfyUI's actual prompt validator using synthetic inputs.
The GPU smoke test compares a synthetic image's real H3 VAE latent to the native
reference node, then verifies an exact save/load round trip. These checks do not
establish generated identity quality or benchmark large reference packs. The AV
smoke compares real video, paired audio and standalone audio latents to the native
reference node and verifies the exact mixed-pack round trip. It checks ComfyUI's
queue before using the GPU and leaves existing generations undisturbed.

## Repository privacy

Git uses an explicit file allowlist: only this package's source code, synthetic
example workflows, documentation and test tools are tracked. Local recordings,
RefMod packs, models, credentials, logs, test outputs and other personal artifacts
are excluded. Add new source files to `.gitignore`'s allowlist deliberately.

Saved RefMods may embed the creation workflow, including prompts and filenames.
Keep personal RefMods and workflow exports outside the tracked example files.
