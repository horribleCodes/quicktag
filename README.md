# QuickTag

Batch image tagger for Windows. QuickTag scans a folder of images, scores them against a predefined tag list using [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224), and writes the selected tags into image metadata. Tagged copies are saved to an output folder; originals are left unchanged.

## User folder layout

After downloading a release (or building locally), arrange your working folder like this:

```
quicktag/
├── quicktag.exe
├── exiftool/
│   ├── exiftool.exe
│   └── exiftool_files/
├── input/          ← place images here
├── output/         ← tagged copies appear here
├── tags.yaml       ← list of possible tags
├── config.yaml     ← paths and scoring settings
└── .cache/         ← created on first run (downloaded model)
```

## Quick start

1. Copy example configs if needed:
   - `config.example.yaml` → `config.yaml`
   - `tags.example.yaml` → `tags.yaml`
2. Edit `tags.yaml` with the tags you want to detect.
3. Put images in `input/`.
4. Run `quicktag.exe`.
5. Find tagged copies in `output/`.

**First run:** SigLIP2 weights (~400 MB) download from Hugging Face into `.cache/huggingface`. An internet connection is required once; later runs work offline.

## Configuration reference

### config.yaml

```yaml
paths:
  input: input          # relative to install dir, or absolute
  output: output

model:
  name: google/siglip2-base-patch16-224
  cache_dir: .cache/huggingface

scoring:
  min_score: 0.05       # ignore tags below this probability
  top_k: 10             # max tags per image (null = no limit)
  top_p: 0.9            # nucleus cutoff on normalized scores (null = disabled)

metadata:
  fields: [Keywords, "XMP:Subject"]
  merge_existing: false # true = union with existing keywords

processing:
  extensions: [jpg, jpeg, png, webp, tiff, tif]
  preserve_timestamps: true
  on_error: skip        # skip | fail

tags_file: tags.yaml    # optional; default tags.yaml
```

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
    prompt: "a photo of a cat"
  - dog
```

Prompts are lowercased automatically (SigLIP2 expects lowercase text).

### Scoring

SigLIP2 returns independent sigmoid scores per tag (multi-label). Tags are selected in this order:

1. Drop tags below `min_score`
2. Sort by score descending
3. If `top_p` is set, keep the smallest prefix whose normalized cumulative score ≥ `top_p`
4. Cap at `top_k` if set

## CLI

```
quicktag.exe                     # uses ./config.yaml
quicktag.exe --config path.yaml  # custom config
quicktag.exe --root C:\myapp    # override install directory
quicktag.exe -v                 # verbose logging
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
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

cp config.example.yaml config.yaml
cp tags.example.yaml tags.yaml
mkdir -p input output

python -m quicktag
pytest
```

On Linux/macOS, install [ExifTool](https://exiftool.org/) and ensure `exiftool` is on `PATH` for metadata writing during development.

## Building

### Linux / macOS

From the repo root with Python 3.11 installed:

```bash
chmod +x build.sh
./build.sh
```

If ExifTool is not installed, the script prints a warning with install instructions and continues. Output lands in `dist/linux/quicktag/`.

### Windows

From a Windows machine with Python 3.11 installed:

```powershell
.\build.ps1
```

ExifTool is downloaded and bundled automatically. Output lands in `dist/win/quicktag/`.

GitHub Actions builds both Linux and Windows artifacts on push to `master`.

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
