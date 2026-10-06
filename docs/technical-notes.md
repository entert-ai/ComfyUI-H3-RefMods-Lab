# Technical notes

[Back to the README](../README.md).

## Dependencies and installation

The node imports ComfyUI's native H3 modules and V3 node API. Missing H3 modules
usually mean ComfyUI needs updating. If a dependency import is missing, install
[requirements.txt](../requirements.txt) with the Python environment used by ComfyUI.
SQLite is part of Python's standard library; it needs no separate pip dependency.
It is used only for persistent save counters, not subject descriptions or references.

Copying the complete node folder into `custom_nodes` still works, but a Git clone
makes updates easier. Avoid nesting the repository inside another copy of its folder.

## Older workflows and packs

- Current packs use Lab format v2; v1 image packs remain readable. The pack format
  version is separate from the package's `0.1.2` release version.
- Replace old A/B Combine nodes and reconnect their inputs. Current inputs are
  numbered 1–100 with matching strength sliders.
- Subject grouping uses the subject name, ignoring capitalization and extra spaces.
  Old canvas naming fields migrate on load. API callers should use `subject_name`;
  `name` and `subject_key` are no longer Create inputs.
- Create nodes no longer expose VAE labels. The connected VAE determines encoding;
  labels stored in older packs remain readable as unverified metadata.

## API and saved metadata

Combine API ports are `refmods.refmod_1` through `refmods.refmod_100`.
The `strengths` input is a JSON object, for example
`{"refmod_1":1,"refmod_2":0.5}`. Multipliers compound with existing source strengths;
the effective strength must remain within 0–2. One input at strength 1 preserves
its references and metadata.

Prompt metadata lives in each source inside the RefMod. The stored source IDs
preserve selections across reloads and strength changes. Recreated or replaced
sources may have different IDs; stale explicit selections block downstream outputs
until the user chooses sources again. `all` follows the pack's current source list.

Text Encode assigns separate Picture, Video and Audio counters in active source
order. It groups description lines beneath their subjects without changing those
media numbers. For paired video/audio, the audio label is emitted before the video
label; their counters are independent.

Set Instructions edits every incoming source without changing its tensors. Empty
descriptions or details clear those fields. It returns a new in-memory pack; Save
persists the result.

Save embeds the canvas graph and API prompt when **Embed workflow** is enabled.
ComfyUI's `--disable-metadata` disables embedding. API-only callers must supply
`extra_pnginfo.workflow` to include a reopenable canvas graph.

Filename counters are case-insensitive per prefix and save folder. Existing files
seed the counter; failed saves can leave gaps. Five digits are minimum padding,
so numbering continues beyond 99999. Keep `.h3-refmods-counter.sqlite3` in backups.
The former shared counter cannot recover prefix counts for previously deleted files.

## Encoding and experiments

`max_edge` limits the longest side, with no deliberate enlargement except rounding
dimensions to H3's 32-pixel grid. Rounding can slightly change aspect ratio.
References retain full VAE latents; there is no pooling, refinement or latent
compression. Prepared source images are stored as quality-95 JPEG bytes for Qwen,
so reference presentation is slightly lossy while saved latent tensors are exact.

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
source, including inactive ones.

For comparison, keep model, seed, prompt, dimensions, steps and attention settings
fixed. Compare ordinary native image references against Lab Text Encode first,
then compare the latent-only Apply path. Use matching reference resize settings:
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
the supplied canvas workflow node types and link consistency, including subgraphs.
The GPU smoke test compares a synthetic image's real H3 VAE latent to the native
reference node, then verifies an exact save/load round trip. These checks do not
establish generated identity quality or benchmark large reference packs. The AV
smoke compares real video, paired audio and standalone audio latents to the native
reference node and verifies the exact mixed-pack round trip. It checks ComfyUI's
queue before using the GPU and leaves existing generations undisturbed.

## Repository privacy

Git uses an explicit allowlist for source code, documentation, the two reviewed
workflow examples, their screenshots and test tools. Add new files deliberately.
Models, saved RefMods, credentials, logs and local test artifacts are excluded.

RefMods with embedded workflows may contain filenames and prompts. Keep personal
packs and workflow exports outside the tracked example files.
