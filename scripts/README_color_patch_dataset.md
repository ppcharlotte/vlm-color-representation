# Pure Color Patch Dataset Builder

Use `build_color_patch_dataset.py` to turn a baseline RGB/name CSV into a pure color patch dataset for the VLM color alignment experiment.

## Expected Input

Default auto-detection accepts columns like:

```csv
r,g,b,name
255,0,0,red
0,0,255,blue
128,128,128,grey
```

Supported basic colors:

```text
black, blue, brown, gray, green, orange, pink, purple, red, white, yellow
```

`grey` is normalized to `gray`.

## Run

From the thesis directory:

```bash
python3 scripts/build_color_patch_dataset.py baseline_color.csv \
  --output-dir data/color_patches \
  --samples-per-color 100 \
  --patch-size 224 \
  --seed 42
```

If the CSV uses different column names:

```bash
python3 scripts/build_color_patch_dataset.py baseline_color.csv \
  --r-col R --g-col G --b-col B --name-col color_name
```

If names are descriptive rather than exact, for example `dark red` or `light blue`, use:

```bash
python3 scripts/build_color_patch_dataset.py baseline_color.csv \
  --match-mode contains
```

## Output

The script writes:

- `data/color_patches/images/*.png`: generated solid color patches.
- `data/color_patches/ground_truth.csv`: image path, basic color label, raw name, RGB, hex, and source row.
- `data/color_patches/metadata.json`: selected counts, skipped rows, column mapping, RGB scale, and shortfalls by color.

Each basic color keeps at most 100 samples. If a color has fewer than 100 rows, all available rows are used.

