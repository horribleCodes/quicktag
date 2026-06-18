# QuickTag

QuickTag automatically tags your images using AI and writes the tags into each file's metadata. Put images in the input folder, run QuickTag, and collect tagged copies from the output folder. Your originals are never modified.

## Folder layout

Keep these files and folders together in the same directory:

```
quicktag/
├── quicktag            ← the executable
├── _internal/          ← bundled libraries (do not delete)
├── input/              ← put your images here
├── output/             ← tagged copies appear here
├── tags.yaml           ← tags to detect
├── config.yaml         ← settings
└── README.md           ← this file
```

On first run, QuickTag downloads the SigLIP2 ONNX bundle (~1.5 GB) from Hugging Face (`horrible/siglip2-base-patch16-224`) into `.cache/huggingface`. You need an internet connection once; later runs work offline.

If the Hugging Face CLI (`hf`) is installed, QuickTag prefers your global Hugging Face cache and reuses models already downloaded there.

## Prerequisites

QuickTag requires **ExifTool** to write metadata. Install it before running:

| Distribution | Install command |
|--------------|-----------------|
| Arch Linux | `sudo pacman -S perl-image-exiftool` |
| Debian / Ubuntu | `sudo apt install libimage-exiftool-perl` |
| Fedora | `sudo dnf install perl-Image-ExifTool` |
| macOS | `brew install exiftool` |

Verify with:

```bash
exiftool -ver
```

More options: https://exiftool.org/install.html

## Quick start

1. Edit `tags.yaml` with the tags you want to detect (see below).
2. Copy images into `input/`.
3. From this folder, run:

   ```bash
   ./quicktag
   ```

4. Open `output/` to find your tagged copies.

Progress is printed to the console: each file shows the tags applied and their confidence scores.

## Customizing tags (`tags.yaml`)

This file lists every tag QuickTag may apply. Only tags from this list are considered — the AI does not invent new ones.

```yaml
tags:
  - person
  - label: woman
  - label: cat
    prompt: "a photo of a cat"
  - dog
  # ...
```

When you use the short form (`- dog`), the prompt defaults to the tag name. Prompts are lowercased automatically.

## Customizing settings (`config.yaml`)

### Input and output paths

Paths are relative to this folder unless you use an absolute path.

```yaml
paths:
  input: input
  output: output
```

### Scoring

Controls which tags get applied to each image.

```yaml
scoring:
  min_score: 0.05   # ignore tags below this confidence (0–1)
  top_k: 10         # max tags per image (use null for no limit)
  top_p: 0.9        # nucleus cutoff (use null to disable)
```

Tags must pass `min_score` first, then `top_p` selects the smallest set whose combined weight reaches the threshold, and `top_k` caps the final count.

Lower `min_score` applies more tags; raise it to be more selective.

### Prompt templates

Set `scoring.prompt_template` in `config.yaml` to wrap every tag prompt in a shared format before classification. Supports placeholders:

- `{tag}` / `{label}` — the tag label
- `{prompt}` — the tag's base prompt (`prompt` field, or label lowercased if missing)

Example:

```yaml
scoring:
  prompt_template: "a photo of {tag}"
```

The default for `prompt_overrides_template` is `false`. When `false`, the template applies to every tag, including those with custom prompts. Set it to `true` so tags with an explicit `prompt` keep that text as-is (simple string tags still use the template). Per-tag `override: true` also skips the template for that tag's custom prompt.

```yaml
# prompt_overrides_template: false
scoring:
  prompt_template: "a photo of {prompt}"
  prompt_overrides_template: false
tags:
  - person                    # "a photo of person"
  - label: feline
    prompt: "a cat"           # "a photo of a cat"
  - label: dog
    prompt: "contains dogs"
    override: true            # "contains dogs"
```

```yaml
# prompt_overrides_template: true
scoring:
  prompt_template: "a photo of {prompt}"
  prompt_overrides_template: true
tags:
  - person                     # "a photo of person"
  - label: feline
    prompt: "a cat"            # "a cat"
  - label: dog
    prompt: "contains dogs"
    override: true             # "contains dogs"
```

### Metadata

```yaml
metadata:
  fields: [Keywords, "XMP:Subject"]
  merge_existing: false
```

Tags are written to both EXIF Keywords and XMP Subject so they show up in most photo apps. Set `merge_existing: true` to add new tags without removing existing ones.

### Processing

```yaml
processing:
  extensions: [jpg, jpeg, png, webp, tiff, tif]
  preserve_timestamps: true
  on_error: skip
```

Only files with listed extensions in `input/` are processed (not subfolders). With `on_error: skip`, a failed file is logged and the rest continue.

### Other options

```yaml
tags_file: tags.yaml   # path to your tag list (relative or absolute)

model:
  name: horrible/siglip2-base-patch16-224
  cache_dir: .cache/huggingface
```

## Command-line options

```bash
./quicktag                     # use config.yaml in this folder
./quicktag --config other.yaml # use a different config file
./quicktag -v                  # verbose logging and per-image tag output
./quicktag -q                  # warnings only
./quicktag -qq                 # warnings only, no progress bar
```

## Supported formats

- JPEG
- PNG
- WebP
- TIFF

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| "ExifTool not found" | Install ExifTool (see Prerequisites) and ensure `exiftool` is on your PATH |
| No tags applied | Lower `min_score` in `config.yaml`, or improve prompts in `tags.yaml` |
| Too many tags | Raise `min_score`, lower `top_k`, or lower `top_p` |
| First run is slow | The model is downloading; wait for it to finish |
| Permission denied | Run `chmod +x quicktag` |

## Exit codes

- `0` — finished successfully
- `1` — config or setup error
- `2` — one or more files failed (when `on_error: skip`)
