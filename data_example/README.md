# Data Example

This directory is reserved for one collected example episode and the outputs
produced by the tool scripts.

Large data files (parquet, mp4, npy, png, json, etc.) are excluded from the
repository via `.gitignore`. Only this README is tracked. You need to collect
your own episode data and place it under `episode/` before running the tool
scripts below.

## Layout

```text
data_example/
  episode/
    # One collected episode, for example:
    # episode_000.parquet
    # sensor_index.json
    # actions/
  tool_results/
    parquet_visualization/
      # Output from tools/visualize/visualize_parquet.py
    mp4_npy/
      # Output from tools/data_processing/parquet_to_mp4_npy.py
    reward_inference/
      # Output from tools/reward_inference/qwen3_reward_inference.py
```

## Offline Verification Commands (No ROS Required)

All commands should be run from the repository root (`cd /home/agilex/DAgger`).

### 1. Generate Parquet Subsets

Split synced parquet episodes into human/rollout parquet subsets:

```bash
python tools/data_processing/split_parquet_by_mode.py \
  --input_base data_example/episode \
  --output_base data_example/tool_results/split_parquet
```

### 2. Convert Parquet to MP4 + NPY

Export parquet episodes to mp4 videos and npy arrays:

```bash
python tools/data_processing/parquet_to_mp4_npy.py \
  --input_base data_example/episode \
  --output_base data_example/tool_results/mp4_npy \
  --fps 30
```

> **Note:** The default encoder is `libsvtav1` (AV1). If OpenCV cannot decode
> AV1 on your platform, transcode the output mp4 files to h264:
>
> ```bash
> # Create an h264 directory alongside the original
> mkdir -p data_example/tool_results/mp4_npy/human/episode_000_h264
>
> # Transcode AV1 -> h264
> for f in data_example/tool_results/mp4_npy/human/episode_000/*.mp4; do
>   name=$(basename "$f")
>   ffmpeg -y -i "$f" -c:v libx264 -crf 23 -pix_fmt yuv420p \
>     "data_example/tool_results/mp4_npy/human/episode_000_h264/$name"
> done
>
> # Symbolic-link remaining files (npy, json, etc.)
> for f in data_example/tool_results/mp4_npy/human/episode_000/*; do
>   [[ "$f" == *.mp4 ]] && continue
>   ln -s "$(realpath "$f")" \
>     "data_example/tool_results/mp4_npy/human/episode_000_h264/$(basename "$f")"
> done
> ```

### 3. Visualize Parquet Data

Save image videos, numeric plots, qpos and action joint-by-joint plots:

```bash
python tools/visualize/visualize_parquet.py \
  data_example/episode/episode_000.parquet \
  --save-plots \
  --save-qpos \
  --save-action \
  --output-dir data_example/tool_results/parquet_visualization
```

### 4. Offline Qwen3-VL Reward Inference

Run Qwen3-VL reward inference on the exported mp4/npy data. Requires the
`fluxdagger` conda environment (Python >= 3.9, transformers >= 4.51) and a
Qwen3-VL reward checkpoint.

```bash
conda run -n fluxdagger python tools/reward_inference/qwen3_reward_inference.py \
  --src_dir data_example/tool_results/mp4_npy/human/episode_000_h264 \
  --save_mp4 data_example/tool_results/reward_inference/episode_000_labeled.mp4 \
  --start 0 \
  --end 619 \
  --model_path /path/to/qwen3_vl_reward_checkpoint
```

> Replace `--end` with the actual total frame count of your video (obtained via
> `cv2.VideoCapture(...).get(cv2.CAP_PROP_FRAME_COUNT)`), and
> `--model_path` with your local checkpoint directory.

For more options, run:

```bash
python <script> --help
```
