# QuickTag

Batch image tagger for Linux and Windows. QuickTag scans a folder of images, scores them against a predefined tag list using [SigLIP2](https://huggingface.co/horrible/siglip2-base-patch16-224) via ONNX Runtime, and writes the selected tags into image metadata. Tagged copies are saved to an output folder; originals are left unchanged.

## User folder layout

After downloading a release (or building locally), arrange your working folder like this.

**Windows:**

```
quicktag/
├── quicktag.exe
├── _internal/          ← bundled libraries (do not delete)
├── exiftool/           ← created on first run if missing (see below)
│   ├── exiftool.exe
│   └── exiftool_files/
├── input/              ← place images here
├── output/             ← tagged copies appear here
├── tags.yaml           ← list of possible tags
├── config.yaml         ← paths and scoring settings
├── README.md
└── .cache/             ← created on first run (downloaded model)
```

**Linux:**

```
quicktag/
├── quicktag              ← the executable
├── _internal/            ← bundled libraries (do not delete)
├── input/
├── output/
├── tags.yaml
├── config.yaml
├── README.md
└── .cache/
```

Linux requires [ExifTool](https://exiftool.org/) on your `PATH` (not bundled). Windows downloads ExifTool into `exiftool/` on first run when it is missing and an internet connection is available.

## Quick start

1. Copy example configs if needed:
   - `config.example.yaml` → `config.yaml`
   - `tags.example.yaml` → `tags.yaml`
2. Edit `tags.yaml` with the tags you want to detect.
3. Put images in `input/`.
4. Run `./quicktag` (Linux) or `quicktag.exe` (Windows).
5. Find tagged copies in `output/`.

**First run:** SigLIP2 ONNX bundle (~1.5 GB) downloads from [horrible/siglip2-base-patch16-224](https://huggingface.co/horrible/siglip2-base-patch16-224) into `.cache/huggingface`. An internet connection is required once; later runs work offline.

If the Hugging Face CLI (`hf`) is installed, QuickTag prefers your global Hugging Face cache and reuses models already downloaded there.

## Configuration reference

### config.yaml

```yaml
paths:
  input: input          # relative to install dir, or absolute
  output: output

model:
  name: horrible/siglip2-base-patch16-224
  cache_dir: .cache/huggingface

scoring:
  min_score: 0.05       # ignore tags below this probability
  top_k: 10             # max tags per image (null = no limit)
  top_p: 0.9            # nucleus cutoff on normalized scores (null = disabled)
  prompt_template: "a photo of {tag}"  # placeholders: {tag}, {label}, {prompt}
  prompt_overrides_template: false      # if true: explicit prompts on a tag skip the template

metadata:
  fields: [Keywords, "XMP:Subject"]
  merge_existing: false # true = union with existing keywords

processing:
  extensions: [jpg, jpeg, png, webp, tiff, tif]
  preserve_timestamps: true
  on_error: skip        # skip | fail

tags_file: tags.yaml    # optional; default tags.yaml
```

The shipped `config.example.yaml` uses `min_score: 0.001` for a more permissive starting point; the code default when the key is omitted is `0.05`.

### tags.yaml

Simple list:

```yaml
tags:
  - cat
  - dog
  - landscape
```

With custom prompts (improves accuracy):

```yaml
tags:
  - label: cat
    prompt: "a cat"
  - label: close-up
    prompt: "a close-up photo"
    override: true
```

Prompts are lowercased automatically (SigLIP2 expects lowercase text).

#### Prompt templates

Set `scoring.prompt_template` in `config.yaml` to wrap every tag prompt in a shared format before classification. Placeholders:

- `{tag}` / `{label}` — the tag label
- `{prompt}` — the tag's base prompt (label lowercased, or the explicit `prompt` field)

Example:

```yaml
scoring:
  prompt_template: "a photo of {tag}"
```

The default for `prompt_overrides_template` is `false`. When `false`, the template applies to every tag, including those with custom prompts. Set it to `true` so tags with an explicit `prompt` keep that text as-is (simple string tags still use the template). Per-tag `override: true` also skips the template for that tag's custom prompt.

```yaml
# prompt_overrides_template: false  → template applies to all tags
scoring:
  prompt_template: "a photo of {prompt}"
  prompt_overrides_template: false
tags:
  - person                       # "a photo of person"
  - label: feline
    prompt: "a cat"               # "a photo of a cat"
  - label: dog
    prompt: "contains dogs"
    override: true                # "contains dogs" (override wins)
```

```yaml
# prompt_overrides_template: true  → explicit prompts kept as-is
scoring:
  prompt_template: "a photo of {prompt}"
  prompt_overrides_template: true
tags:
  - person                       # "a photo of person"
  - label: feline
    prompt: "a cat"               # "a cat"
  - label: dog
    prompt: "contains dogs"
    override: true                # "contains dogs"
```

### Scoring

SigLIP2 returns independent sigmoid scores per tag (multi-label). Tags are selected in this order:

1. Drop tags below `min_score`
2. Sort by score descending
3. If `top_p` is set, keep the smallest prefix whose normalized cumulative score ≥ `top_p`
4. Cap at `top_k` if set

## CLI

```
quicktag                           # uses ./config.yaml beside the executable
quicktag --config path.yaml        # custom config
quicktag --root /path/to/quicktag  # override install directory (dev/smoke tests)
quicktag -v                        # verbose logging and per-image tag output
quicktag -q                        # warnings only
quicktag -qq                       # warnings only, no progress bar
```

Exit codes:

- `0` — success
- `1` — config or validation error
- `2` — one or more files failed (when `on_error: skip`)

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp config.example.yaml config.yaml
cp tags.example.yaml tags.yaml
mkdir -p input output

python -m quicktag
pytest
```

Or use the Makefile shortcuts (`make install`, `make test`, `make build-linux`).

Integration tests (model download) are opt-in:

```bash
pytest -m integration
pytest -m ""    # run everything
```

Download the ONNX bundle locally (optional, for offline dev):

```bash
python scripts/download_onnx_model.py \
  --output .cache/huggingface/onnx-export/horrible--siglip2-base-patch16-224
```

CI build smoke tests use a tiny committed ONNX bundle (`tests/fixtures/onnx-smoke-bundle/`) with `config.smoke.yaml` — no Hugging Face download in the Build workflow. Regenerate the bundle after changing the ONNX I/O contract in `onnx_tagger.py`:

```bash
pip install -e ".[onnx]"
python scripts/generate_smoke_onnx_bundle.py
```

On Linux/macOS, install [ExifTool](https://exiftool.org/) and ensure `exiftool` is on `PATH` for metadata writing during development.

Stage images in `.tmp/input/` before a local build to copy them into the dist bundle's `input/` folder automatically (see `.cursor/rules/post-build-input-copy.mdc`).

## Building

### Linux / macOS

From the repo root with Python 3.11 installed:

```bash
chmod +x build.sh
./build.sh
```

If ExifTool is not installed, the script prints a warning with install instructions and continues. Output lands in `dist/linux/quicktag/` with `docs/DIST_README_LINUX.md` copied as `README.md`.

### Windows

From a Windows machine with Python 3.11 installed:

```powershell
.\build.ps1
```

Output lands in `dist/win/quicktag/` with `docs/DIST_README_WIN.md` copied as `README.md`. ExifTool is not bundled at build time; the executable downloads it on first run when missing.

GitHub Actions runs the Test workflow on pull requests and the Build workflow on push to `main`, `master`, or `release`, producing Linux and Windows artifacts.

## Releasing

Version lives in `pyproject.toml` and `src/quicktag/__init__.py` (keep both in sync). Use semver (`MAJOR.MINOR.PATCH`).

1. Bump the version in both files and merge to `master`.
2. Wait for the [Build workflow](.github/workflows/build.yml) to succeed on that commit.
3. Create the release using either method:
   - **Actions → Release → Run workflow** — leave `version` blank to use `pyproject.toml`, or pass an explicit version.
   - **Push a tag** — `git tag v0.2.0 <commit-sha> && git push origin v0.2.0`

The release workflow downloads Linux and Windows artifacts from that commit's successful Build run (no rebuild). Assets are published as `quicktag-v{version}-linux.tar.gz` and `quicktag-v{version}-windows.zip`.

Release fails if CI did not pass for the commit, or if the tag/input version does not match `pyproject.toml`.

## License

MIT. ExifTool is bundled under its own license; see [exiftool.org](https://exiftool.org/).
