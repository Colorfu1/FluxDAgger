# FluxDAgger: A Model-Decoupled DAgger Pipeline for Dual-Arm Robotic Manipulation

<div align="center">
<a href="https://hqr-robotic.github.io/FluxDAgger-project-page/"><img src="https://img.shields.io/badge/Project_Page-Purple?color=8A2BE2&logo=githubpages" alt="Project Page"></a>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

**FluxDAgger** is a model-decoupled DAgger pipeline for dual-arm robotic manipulation. It decouples the policy model, reward model, data collection controller, and dataset processing stack, making the system easier to adapt to different VLA models and reward models.

https://github.com/user-attachments/assets/73ac0c18-d7d8-4326-b260-36df47fe9cee

The system is built around a modular ROS architecture with synchronized multi-camera observations, online human takeover, DAgger-style correction collection, reward visualization, and dataset processing.

<p align="center">
  <img src="assets/fluxdagger_system.svg" alt="FluxDAgger system overview" width="92%">
</p>

## 🌟 Features

- **Model-decoupled design**: policy inference runs in an external project; this repository only consumes agreed ROS topics.
- **Human-in-the-loop DAgger**: supports autonomous rollout, online takeover, and corrective data collection.
- **Synchronized multi-camera observations**: timestamp-based synchronization aligns multi-camera streams and robot state.
- **Four-arm Piper control**: manages front/rear and master/slave arms through ROS nodes.
- **Reward visualization**: supports online and offline Qwen3-VL reward inference.
- **Dataset processing pipeline**: exports raw Parquet, segmented videos, NumPy arrays, and reward annotations.

## 📢 Latest News

**\[2026/05/29\]** 🔥 FluxDAgger has been open-sourced.

## 🛠️ Installation

<details>
<summary><b>1. Clone the repository</b></summary>

```bash
git clone https://github.com/FluxVLA/FluxDAgger.git
cd FluxDAgger
```

</details>

<details>
<summary><b>2. Set up the environment</b></summary>

```bash
bash setup_env.sh
```

`setup_env.sh` uses official PyPI by default. If PyPI is slow in mainland China, enable a mirror via `DAGGER_DISABLE_PIP_MIRROR=0` — see [Documentation](docs/README.md). Manual installs are covered under **Installation**.

</details>

<details>
<summary><b>3. Build the ROS workspace</b></summary>

```bash
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

</details>

## Usage

<details>
<summary><b>Launch FluxDAgger data collection</b></summary>

```bash
roslaunch dagger dagger.launch \
  scene_dir:=fold_towels \
  ckpt_dir:=/path/to/checkpoint \
  task_id:=policy_name \
  save_data_base_dir:=/home/agilex/dagger_data/data_buffer
```

</details>

<details>
<summary><b>Launch FluxDAgger with Qwen3 reward</b></summary>

```bash
roslaunch dagger dagger_with_reward.launch \
  scene_dir:=fold_towels \
  ckpt_dir:=/path/to/checkpoint \
  task_id:=policy_name \
  qwen3_model_path:=/path/to/qwen3_reward/checkpoint \
  save_data_base_dir:=/home/agilex/dagger_data/data_buffer
```

</details>

## Framework

### CAN Bus Control

For four-arm teleoperation and DAgger collection, FluxDAgger uses a modified CAN wiring and configuration. The original two USB-CAN setup is extended to four independent USB-CAN interfaces, so each Piper arm can be addressed through its own CAN channel instead of sharing a coupled control path. This allows the front/rear and master/slave arms to be enabled, switched, and commanded independently by the four arm_node instances.

| Original CAN Bus Control | Modified CAN Bus Control |
| :----------------------: | :----------------------: |
| <img src="assets/original-can-bus-control.png" alt="Original CAN bus control"> | <img src="assets/modified-can-bus-control.png" alt="Modified CAN bus control"> |

<br>

### Timestamp Synchronization

Precise temporal alignment across multiple camera streams is critical for consistent multi-view observations. FluxDAgger follows the timestamp synchronization implementation from the official AgileX data collection system: each camera maintains a frame buffer, and frames are aligned to a common sync timestamp. Frames arriving before the sync time are discarded, ensuring all camera views correspond to the same physical moment during rollout and human takeover.

<p align="center">
  <img src="assets/timestamp_sync_strategy.svg" alt="Timestamp synchronization strategy" width="92%">
</p>

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
├── docs/                # Technical documentation
├── requirements.txt
└── setup_env.sh
```

## Core Modules

| Module                      | Description                                                            |
| --------------------------- | ---------------------------------------------------------------------- |
| `sync_observation_node.py`  | Publishes synchronized multi-camera and robot-state observations.      |
| `dagger_controller_node.py` | Manages episode state, keyboard commands, and arm mode switching.      |
| `dagger_collector_node.py`  | Collects synchronized observations, actions, and episode metadata.     |
| `qwen3_reward_node.py`      | Runs asynchronous Qwen3 reward inference and publishes reward signals. |
| `arm_node.py`               | Controls per-arm mode, target subscription, and enable state.          |

## Data Processing

FluxDAgger stores each collected episode as Parquet data with camera and robot states, then processes the data into multiple downstream formats:

- human/rollout parquet subsets
- full-trajectory MP4 videos and NumPy arrays
- videos segmented around human takeover periods
- offline reward annotations and visualization results

## Related Projects

- [FluxVLA](https://github.com/FluxVLA/FluxVLA): a full-stack engineering platform for VLA training, evaluation, acceleration, and real-robot inference.
- [ARM](https://arxiv.org/abs/2604.03037): Advantage Reward Modeling for long-horizon robotic manipulation.

## Support

If you encounter any issues while using this repository, feel free to contact us. You can reach us directly at [ryan.hu@limxdynamics.com](mailto:ryan.hu@limxdynamics.com) and [wayne@limxdynamics.com](mailto:wayne@limxdynamics.com), or open a GitHub issue for help.

## 🙏 Citation & Acknowledgements

If you find FluxDAgger useful, please consider citing or linking to the project page.

Acknowledgement: FluxDAgger builds on the Piper robot ecosystem and uses [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk) for Piper arm integration.
