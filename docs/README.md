# FluxDAgger: A Model-Decoupled DAgger Pipeline for Dual-Arm Robotic Manipulation

<div align="center">
<a href="https://hqr-robotic.github.io/FluxDAgger-project-page/"><img src="https://img.shields.io/badge/Project_Page-Purple?color=8A2BE2&logo=githubpages" alt="Project Page"></a>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

**FluxDAgger** is a model-decoupled DAgger pipeline for dual-arm robotic manipulation. It decouples the policy model, reward model, data collection controller, and dataset processing stack, making the system easier to adapt to different VLA models and reward models.

The system is built around a modular ROS architecture with synchronized multi-camera observations, online human takeover, DAgger-style correction collection, Qwen3-VL reward visualization, and host-side dataset processing.

<p align="center">
  <img src="../assets/fluxdagger_system.svg" alt="FluxDAgger system overview" width="92%">
</p>

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Unified Configuration (YAML)](#unified-configuration-yaml)
- [Real-Robot Collection Workflow](#real-robot-collection-workflow)
- [Usage](#usage)
- [Framework](#framework)
- [Data Format](#data-format)
- [Reward](#reward)
- [Tool Scripts](#tool-scripts)
- [FAQ](#faq)
- [Related Projects](#related-projects)
- [Support](#support)
- [Citation & Acknowledgements](#citation--acknowledgements)

## <a id="features"></a>🌟 Features

- **Model-decoupled design**: policy inference runs in an external project; this repository only consumes agreed ROS topics.
- **Human-in-the-loop DAgger**: supports autonomous rollout, online takeover, and corrective data collection.
- **Synchronized multi-camera observations**: timestamp-based synchronization aligns multi-camera streams and robot state.
- **Four-arm Piper control**: manages front/rear and master/slave arms through ROS nodes.
- **Reward visualization**: supports online and offline Qwen3-VL reward inference.
- **Dataset processing pipeline**: exports raw Parquet, segmented videos, NumPy arrays, and reward annotations.

## Repository Layout

```text
FluxDAgger
├── src/
│   ├── dagger/
│   │   ├── config/      # default.yaml — single source of defaults
│   │   ├── dagger/      # Python package (nodes import dagger.* )
│   │   ├── launch/      # roslaunch files
│   │   └── scripts/     # ROS node entrypoints
│   └── dagger_msgs/     # Custom ROS messages
├── tools/
│   ├── data_processing/ # Parquet/video/NumPy data processing
│   ├── hardware/        # CAN and Piper initialization tools
│   ├── reward_inference/ # Offline reward inference scripts
│   └── visualize/       # Real-time and offline visualization tools
├── data_example/        # Example episode and tool output workspace
├── requirements.txt
└── setup_env.sh
```

## <a id="installation"></a>🛠️ Installation

System requirements:

- Ubuntu 20.04 + ROS Noetic.
- Python 3.10 conda environment or a compatible environment.
- Real-robot collection requires Piper SDK, camera nodes, and CAN devices.
- Qwen3-VL reward requires an available GPU and a local checkpoint.

<details>
<summary><b>1. Run the one-step environment script</b></summary>

```bash
export DAGGER_REPO=/path/to/FluxDAgger
cd "$DAGGER_REPO"
bash setup_env.sh
```

</details>

<details>
<summary><b>2. Enter the environment</b></summary>

```bash
conda activate fluxdagger
source /opt/ros/noetic/setup.bash
source "$DAGGER_REPO/devel/setup.bash"
```

</details>

<details>
<summary><b>Mainland China / slow PyPI</b></summary>

`setup_env.sh` defaults to official PyPI. To use a mirror, set `DAGGER_DISABLE_PIP_MIRROR=0`; `DAGGER_PIP_INDEX_URL` defaults to the Tsinghua mirror. The `--extra-index-url` in `requirements.txt` remains, so CUDA PyTorch wheels still resolve from PyTorch's index.

```bash
# Enable Tsinghua mirror
DAGGER_DISABLE_PIP_MIRROR=0 bash setup_env.sh

# Use another mirror
DAGGER_DISABLE_PIP_MIRROR=0 \
  DAGGER_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  bash setup_env.sh
```

Manual install with a PyPI mirror:

```bash
pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r "$DAGGER_REPO/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

You can also configure a global `index-url` via `~/.pip/pip.conf` or `pip config set`.

</details>

<details>
<summary><b>3. Build the ROS workspace</b></summary>

```bash
cd "$DAGGER_REPO"
catkin_make
source devel/setup.bash
```

</details>

## Unified Configuration (YAML)

Runtime defaults live in `src/dagger/config/default.yaml` and are loaded by `dagger.runtime_config`. Each node receives a private ROS param `config_file` (launch default: `$(find dagger)/config/default.yaml`). Launch `<arg>` values map to node `<param>` entries; the placeholder `__config__` means “use the value from YAML for this key.” Explicit launch arguments override YAML for one run only.

Placeholders `{home_dir}` and `{repo_dir}` are expanded when the file loads (`repo_dir` is inferred from the config file path).

## Real-Robot Collection Workflow

Camera nodes and Piper SDK come from official external projects. This repository depends only on their ROS topics and CAN interfaces. Camera code, Piper SDK code, and external policy inference code are not vendored here.

External dependency sources:

- Camera node: official `astra_camera` ROS node. Start with `roslaunch astra_camera nmulti_camera.launch`.
- Piper SDK: AgileX official SDK at <https://github.com/agilexrobotics/piper_sdk>.
- Policy inference: external VLA/policy project. The current interface matches FluxVLA and must continuously publish the command and raw-action topics consumed by FluxDAgger.

If USB-CAN devices were replugged, reconfigure CAN ports:

```bash
cd "$DAGGER_REPO"
sudo ./tools/hardware/can_config_modified.sh
```

`tools/hardware/can_config_modified.sh` is derived from the Piper ROS project’s `can_config.sh`.

Key CAN configuration changes:

| Item             | Before `can_config.sh` | After `can_config_modified.sh` |
| ---------------- | ---------------------- | ------------------------------ |
| CAN module count | `EXPECTED_CAN_COUNT=3` | `EXPECTED_CAN_COUNT=4`         |
| `1-13:1.0`       | `can_left:1000000`     | `c_left_slave:1000000`         |
| `1-12:1.0`       | `can_right:1000000`    | `c_left_master:1000000`        |
| `1-5:1.0`        | Not configured         | `c_right_master:1000000`       |
| `1-2:1.0`        | `can0:500000`          | `c_right_slave:1000000`        |

These interface names must match the `can_port` values for the four `arm_node` entries in `src/dagger/launch/dagger.launch`:

| DAgger arm node   | CAN port         |
| ----------------- | ---------------- |
| `arm_front_left`  | `c_left_slave`   |
| `arm_front_right` | `c_right_slave`  |
| `arm_rear_left`   | `c_left_master`  |
| `arm_rear_right`  | `c_right_master` |

Recommended startup order:

1. Start the camera node.

```bash
roslaunch astra_camera nmulti_camera.launch
```

2. Configure the four arms into the expected master/slave relationship and move them to the parallel initial pose. `piper_initial_pose.py` can be run repeatedly until all four arms are parallel and horizontal.

```bash
cd "$DAGGER_REPO"
python tools/hardware/piper_set_slave.py
python tools/hardware/piper_initial_pose.py
```

3. Start external policy inference. Replace environment, project, config, and checkpoint with your own setup.

```bash
conda activate <policy_inference_env>
cd <policy_inference_project>

python <inference_entrypoint>.py \
  --config <policy_config> \
  --ckpt-path <policy_checkpoint>
```

<details>
<summary><b>4. Build and launch DAgger collection</b></summary>

```bash
cd "$DAGGER_REPO"
catkin_make
source ./devel/setup.bash
roslaunch dagger dagger.launch
```

</details>

Collection keys:

| Key | Action                                         |
| --- | ---------------------------------------------- |
| `s` | Start an episode and enter inference mode      |
| `h` | Switch to human mode; rear arms act as masters |
| `i` | Switch back to inference mode                  |
| `r` | Stop and save, then enter success and score    |
| `d` | Discard the current episode                    |
| `q` | Exit and broadcast shutdown                    |

## Usage

<details>
<summary><b>Pre-launch checks</b></summary>

```bash
source /opt/ros/noetic/setup.bash
source "$DAGGER_REPO/devel/setup.bash"
roscore
```

</details>

<details>
<summary><b>Verify sensor topics</b></summary>

```bash
rostopic hz /camera_h/color/image_raw
rostopic hz /camera_l/color/image_raw
rostopic hz /camera_r/color/image_raw
rostopic hz /puppet/joint_left
rostopic hz /puppet/joint_right
```

</details>

<details>
<summary><b>Pure DAgger data collection</b></summary>

```bash
roslaunch dagger dagger.launch \
  scene_dir:=fold_towels \
  ckpt_dir:=/path/to/checkpoint \
  task_id:=policy_name \
  save_data_base_dir:=/path/to/dagger_data/data_buffer
```

</details>

<details>
<summary><b>Optional: point all nodes at a custom config file</b></summary>

```bash
roslaunch dagger dagger.launch \
  config_file:=/path/to/my_dagger.yaml \
  scene_dir:=fold_towels \
  ckpt_dir:=/path/to/checkpoint \
  task_id:=policy_name \
  save_data_base_dir:=/path/to/dagger_data/data_buffer
```

</details>

<details>
<summary><b>Full collection stack + Qwen3-VL reward</b></summary>

```bash
roslaunch dagger dagger_with_reward.launch \
  scene_dir:=fold_towels \
  ckpt_dir:=/path/to/checkpoint \
  task_id:=policy_name \
  qwen3_model_path:=$DAGGER_REPO/ckpt/reward_model/qwen3_reward/step35000 \
  save_data_base_dir:=/path/to/dagger_data/data_buffer
```

</details>

Common `dagger.launch` parameters (defaults match `default.yaml` unless overridden):

| Parameter               | Default                              | Description                                                                   |
| ----------------------- | ------------------------------------ | ----------------------------------------------------------------------------- |
| `config_file`           | `$(find dagger)/config/default.yaml` | Unified YAML for nodes in this launch                                         |
| `scene_dir`             | `fold_towels`                        | Task/scene directory name                                                     |
| `ckpt_dir`              | `/tmp/test_sync`                     | Policy checkpoint path, used to build save paths                              |
| `task_id`               | `test_policy`                        | Policy or experiment name                                                     |
| `save_data_base_dir`    | `~/dagger_data/data_buffer`          | Data root (YAML `paths.data_buffer_dir`)                                      |
| `collect_frame_data`    | `true`                               | Whether to start saving                                                       |
| `sync_observation_rate` | `90.0`                               | Sync node loop rate (Hz)                                                      |
| `data_collection_rate`  | `30.0`                               | Collector thread rate; each tick saves at most one latest `SyncedObservation` |
| `sync_slop`             | `0.03`                               | Max time span across synchronized streams (seconds)                           |
| `img_front_topic`       | empty                                | Optional front camera                                                         |
| `img_head_topic`        | `/camera_h/color/image_raw`          | Head camera                                                                   |
| `img_left_topic`        | `/camera_l/color/image_raw`          | Left wrist camera                                                             |
| `img_right_topic`       | `/camera_r/color/image_raw`          | Right wrist camera                                                            |

## Framework

Precise temporal alignment across multiple camera streams is critical for consistent multi-view observations. FluxDAgger follows the timestamp synchronization implementation from the official AgileX data collection system: each camera maintains a frame buffer, and frames are aligned to a common sync timestamp. Frames arriving before the sync time are discarded, ensuring all camera views correspond to the same physical moment during rollout and human takeover.

<p align="center">
  <img src="../assets/timestamp_sync_strategy.svg" alt="Timestamp synchronization strategy" width="92%">
</p>

Node topology:

```text
raw sensors
  /camera_h/color/image_raw
  /camera_l/color/image_raw
  /camera_r/color/image_raw
  /camera_f/color/image_raw (optional)
  /puppet/joint_left
  /puppet/joint_right
  /puppet/end_pose_left
  /puppet/end_pose_right
       |
       v
sync_observation_node
       |
       +--> /robot/observation_sync
                |
                +--> dagger_collector_node -> episode parquet + sensor_index.json
                |
                +--> qwen3_reward_node -> /robot/reward_signal

dagger_controller_node
       |
       +--> /dagger/global_state
       +--> /dagger/control/command
       +--> /dagger/collector/command
       +--> /dagger/arm/<name>/mode
       +--> /dagger/arm/<name>/subscribe_target
       +--> /dagger/arm/<name>/enable
                |
                +--> arm_node x 4
```

Core modules:

| Module                                | Description                                                                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dagger.runtime_config`               | Loads ROS topics, camera names, paths, four-arm profiles, and home poses from `src/dagger/config/default.yaml`.                                         |
| `dagger.infra.ros_observation_buffer` | Subscribes to camera, arm, command, and raw-action topics and maintains message queues.                                                                 |
| `dagger.infra.frame_sync`             | Selects aligned frames within the sync window and returns `FrameData`.                                                                                  |
| `sync_observation_node.py`            | Publishes `SyncedObservation` and tracks `episode_id` / `step_id`.                                                                                      |
| `dagger_collector_node.py`            | Consumes `SyncedObservation`, aligns command/raw-action queues, writes parquet; worker samples at `data_collection_rate` (latest observation per tick). |
| `dagger_controller_node.py`           | Keyboard input, global state, episode lifecycle, four-arm mode switching.                                                                               |
| `qwen3_reward_node.py`                | Asynchronous Qwen3 reward inference and publishing.                                                                                                     |
| `arm_node.py`                         | Per-arm mode, target subscription, enable state.                                                                                                        |

Episode lifecycle:

1. User presses `s`.
2. Controller publishes `start_collect`, `start_episode`, and global state.
3. Collector creates a new episode directory and publishes `EpisodeInfo(is_active=True)`.
4. SyncNode resets `step_id` after receiving `EpisodeInfo`.
5. Each `SyncedObservation` carries the current `episode_id` and `step_id`.
6. User presses `h` / `i` to switch between human and inference modes.
7. User presses `r` to finish; collector prompts for success and score, then writes disk.
8. User presses `d` to discard; collector deletes the episode directory.
9. User presses `q`; controller broadcasts shutdown.

Main topics:

| Topic                              | Type                | Publisher  | Subscribers                 |
| ---------------------------------- | ------------------- | ---------- | --------------------------- |
| `/robot/observation_sync`          | `SyncedObservation` | SyncNode   | Collector, Reward           |
| `/robot/reward_signal`             | `RewardSignal`      | Reward     | Visualization               |
| `/dagger/global_state`             | `GlobalState`       | Controller | Collector, ArmNode          |
| `/dagger/control/command`          | `String`            | Controller | ArmNode, SyncNode, Reward   |
| `/dagger/collector/command`        | `String`            | Controller | Collector, SyncNode, Reward |
| `/dagger/episode_info`             | `EpisodeInfo`       | Collector  | SyncNode                    |
| `/dagger/controller/input_request` | `InputRequest`      | Collector  | Controller                  |
| `/dagger/controller/input_result`  | `InputResult`       | Controller | Collector                   |
| `/dagger/arm/<name>/status`        | `ArmStatus`         | ArmNode    | Controller                  |

Hardware and policy topics:

| Topic                                              | Description                         |
| -------------------------------------------------- | ----------------------------------- |
| `/camera_h/color/image_raw`                        | Head camera                         |
| `/camera_l/color/image_raw`                        | Left wrist camera                   |
| `/camera_r/color/image_raw`                        | Right wrist camera                  |
| `/camera_f/color/image_raw`                        | Optional front camera               |
| `/puppet/joint_left` / `/puppet/joint_right`       | Front-arm joint states              |
| `/puppet/end_pose_left` / `/puppet/end_pose_right` | Front-arm end-effector poses        |
| `/master/joint_left` / `/master/joint_right`       | Joint commands from policy or human |
| `/left_arm/trajectory` / `/right_arm/trajectory`   | Policy raw action chunks            |

## Data Format

Collector writes data to:

```text
<save_data_base_dir>/<scene_dir>/<YYYYMMDD>/<checkpoint_folder>/<task_id>/episode_XXX/
```

Example:

```text
<save_data_base_dir>/
  fold_towels/
    20260511/
      policy_ckpt_folder/
        policy_name/
          episode_000/
            episode_000.parquet
            sensor_index.json
            actions/
```

Parquet columns:

| Column                                 | Type          | Description                               |
| -------------------------------------- | ------------- | ----------------------------------------- |
| `/observations/qpos`                   | `float32[14]` | Left and right front-arm joint positions  |
| `/observations/qvel`                   | `float32[14]` | Left and right front-arm joint velocities |
| `/observations/effort`                 | `float32[14]` | Left and right front-arm joint effort     |
| `/observations/eepose`                 | `float32[14]` | Left/right front-arm poses, 7 values each |
| `/observations/images/cam_high`        | PNG bytes     | Head camera                               |
| `/observations/images/cam_left_wrist`  | PNG bytes     | Left wrist camera                         |
| `/observations/images/cam_right_wrist` | PNG bytes     | Right wrist camera                        |
| `/observations/images/cam_front`       | PNG bytes     | Optional front camera                     |
| `/action`                              | `float32[]`   | Joint command aligned to the frame        |
| `/data_flag`                           | semantic flag | Rollout vs human flag                     |
| `/success`                             | `int`         | Episode-level success label               |
| `/score`                               | `int`         | Episode score; failures often `-1`        |

Frames with incomplete dual-arm state, velocity, effort, or end-effector poses are **skipped** (no zero padding). The pipeline does **not** write `base_action` / base velocity columns.

`sensor_index.json` records success/score, frame mode, synchronization timestamps, and references to raw action chunk files. `actions/` stores raw chunks for analyzing policy output vs executed motion.

Pipeline outputs:

- human/rollout parquet subsets;
- full-trajectory MP4 and NumPy arrays;
- videos segmented around human takeover;
- offline reward annotations and visualization.

## Reward

The mainline uses Qwen3-VL reward. The node reads `/robot/observation_sync` and publishes `/robot/reward_signal` for visualization, logging, and upstream modules.

Full stack with online reward:

```bash
roslaunch dagger dagger_with_reward.launch \
  scene_dir:=fold_towels \
  ckpt_dir:=/path/to/checkpoint \
  task_id:=policy_name \
  qwen3_model_path:=$DAGGER_REPO/ckpt/reward_model/qwen3_reward/step35000 \
  save_data_base_dir:=/path/to/dagger_data/data_buffer
```

`Qwen3RewardModel` caches `cam_high`, `cam_left_wrist`, and `cam_right_wrist`. Common parameters:

| Parameter        | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| `f_interval`     | Step interval between neighboring frames in one inference window |
| `t_interval`     | Minimum steps between inference calls                            |
| `prompt`         | Task description for Qwen3-VL                                    |
| `max_new_tokens` | Max generated tokens                                             |

Parsed status labels:

| Status       | Reward |
| ------------ | ------ |
| `Normal`     | `1.0`  |
| `Static`     | `0.3`  |
| `Grasp fail` | `0.0`  |
| `Shaking`    | `0.1`  |
| `Drop`       | `0.0`  |

Offline reward:

```bash
python tools/reward_inference/qwen3_reward_inference.py \
  --src_dir /path/to/mp4_npy_data \
  --save_mp4 /path/to/save_dir \
  --start 0 \
  --end 100 \
  --model_path "$DAGGER_REPO/ckpt/reward_model/qwen3_reward/step35000"
```

## Tool Scripts

`tools/` holds offline processing, visualization, replay, reward inference, and hardware scripts. Commands assume the repository root:

```bash
cd "$DAGGER_REPO"
```

Overview:

| Tool                                               | Purpose                                         |
| -------------------------------------------------- | ----------------------------------------------- |
| `tools/hardware/can_config_modified.sh`            | Configure four-arm CAN interfaces.              |
| `tools/hardware/piper_set_slave.py`                | Piper master/slave setup.                       |
| `tools/hardware/piper_initial_pose.py`             | Move four arms to the parallel initial pose.    |
| `tools/data_processing/split_parquet_by_mode.py`   | Split raw parquet into human/rollout subsets.   |
| `tools/data_processing/parquet_to_mp4_npy.py`      | Convert parquet subsets to MP4 + NumPy.         |
| `tools/visualize/visualize_parquet.py`             | Inspect one parquet episode; save plots/images. |
| `tools/visualize/visualize_ros_reward.py`          | Live ROS cameras + reward.                      |
| `tools/replay_qpos_npy.py`                         | Replay saved qpos to ROS command topics.        |
| `tools/reward_inference/qwen3_reward_inference.py` | Offline Qwen3 reward on MP4/NPY data.           |

Common commands:

```bash
python tools/data_processing/split_parquet_by_mode.py \
  --input_base /path/to/raw_dataset \
  --output_base /path/to/sub_parquet_dataset

python tools/data_processing/parquet_to_mp4_npy.py \
  --input_base /path/to/sub_parquet_dataset \
  --output_base /path/to/mp4_npy_dataset \
  --fps 30

python tools/visualize/visualize_parquet.py \
  /path/to/episode_000.parquet \
  --save-plots \
  --save-qpos \
  --save-action \
  --output-dir /tmp/episode_000_vis

python tools/visualize/visualize_ros_reward.py \
  --cameras h,l,r \
  --reward_topic /robot/reward_signal

python tools/reward_inference/qwen3_reward_inference.py \
  --src_dir /path/to/mp4_npy_dataset \
  --save_mp4 /path/to/reward_output \
  --start 0 \
  --end 100 \
  --model_path /path/to/qwen3_reward/checkpoint
```

Per-script options:

```bash
python <script> --help
```

## FAQ

`dagger_msgs` import errors:

<details>
<summary><b>Rebuild and source the workspace</b></summary>

```bash
cd "$DAGGER_REPO"
catkin_make
source devel/setup.bash
```

</details>

Sensor topics empty:

```bash
rostopic hz /camera_h/color/image_raw
rostopic hz /camera_l/color/image_raw
rostopic hz /camera_r/color/image_raw
rostopic hz /puppet/joint_left
rostopic hz /puppet/joint_right
```

Qwen3 reward not publishing:

- Confirm `qwen3_reward_node.py` is running.
- Confirm `/robot/observation_sync` is publishing.
- Confirm the model path points to a local checkpoint.
- Confirm GPU memory is sufficient.

## Related Projects

- [FluxVLA](https://github.com/FluxVLA/FluxVLA): full-stack platform for VLA training, evaluation, acceleration, and real-robot inference.
- [ARM](https://arxiv.org/abs/2604.03037): Advantage Reward Modeling for long-horizon manipulation.

## Support

If you encounter any issues while using this repository, feel free to contact us. You can reach us directly at [ryan.hu@limxdynamics.com](mailto:ryan.hu@limxdynamics.com) and [wayne@limxdynamics.com](mailto:wayne@limxdynamics.com), or open a GitHub issue for help.

## <a id="citation--acknowledgements"></a>🙏 Citation & Acknowledgements

If FluxDAgger is useful to your research or engineering work, please cite or link to the project page.

Acknowledgement: FluxDAgger builds on the Piper robot ecosystem and uses [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk) for Piper arm integration.
