# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Typed loader for the unified FluxDAgger YAML configuration.

All runtime defaults are defined in ``src/dagger/config/default.yaml``. A
caller may pass a different YAML file to ``load_runtime_config``; that file is
merged over the default config and becomes the only override source.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_VALUE_SENTINEL = '__config__'

RawConfig = dict[str, Any]
PathContext = dict[str, str]
OptionalPath = str | Path | None
ConfigData = tuple[RawConfig, Path]

# ---------------------------------------------------------------------------
# Typed config model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeParams:
    """Common node rates and runtime flags."""

    debug: bool
    master_publish_rate: float
    puppet_publish_rate: float
    sync_observation_rate: float
    data_collection_rate: float
    sync_slop: float
    synced_observation_pub_queue_size: int
    synced_observation_queue_size: int
    synced_observation_drain_timeout: float
    task_text: str


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations resolved from the YAML ``paths`` section."""

    home_dir: Path
    repo_dir: Path
    data_buffer_dir: Path
    captures_dir: Path
    screenshots_dir: Path
    sync_fail_log_dir: Path
    act_checkpoint_dir: Path
    qwen3_reward_model: Path


@dataclass(frozen=True)
class TopicsConfig:
    """ROS topic names and templates."""

    puppet_arm_left: str
    puppet_arm_right: str
    puppet_ee_pose_left: str
    puppet_ee_pose_right: str
    master_joint_left: str
    master_joint_right: str
    human_joint_left: str
    human_joint_right: str
    raw_action_left: str
    raw_action_right: str
    dagger_control_command: str
    dagger_global_state: str
    dagger_collector_command: str
    dagger_episode_info: str
    dagger_episode_result: str
    dagger_input_request: str
    dagger_input_result: str
    robot_observation_sync: str
    robot_reward_signal: str
    arm_mode_template: str
    arm_subscribe_template: str
    arm_enable_template: str
    arm_status_template: str

    def arm_mode(self, arm_name: str) -> str:
        return self.arm_mode_template.format(arm_name=arm_name)

    def arm_subscribe(self, arm_name: str) -> str:
        return self.arm_subscribe_template.format(arm_name=arm_name)

    def arm_enable(self, arm_name: str) -> str:
        return self.arm_enable_template.format(arm_name=arm_name)

    def arm_status(self, arm_name: str) -> str:
        return self.arm_status_template.format(arm_name=arm_name)


@dataclass(frozen=True)
class CameraConfig:
    """One configured camera across ROS, datasets, and tools."""

    key: str
    name: str
    topic: str
    ros_param: str
    short_key: str
    msg_field: str
    label: str
    sync_enabled: bool

    @property
    def sync_topic(self) -> str:
        """Topic used by the sync node; empty means disabled."""
        return self.topic if self.sync_enabled else ''


@dataclass(frozen=True)
class ArmProfile:
    """One arm's role, topics, CAN port, and home pose."""

    name: str
    role: str
    joint_pub_topic: str | None
    joint_sub_topic: str
    joint_state_pub_topic: str | None
    end_pose_pub_topic: str | None
    human_sub_topic: str | None
    home_joint: tuple[float, ...]
    home_end_pose: tuple[float, ...]
    can_port: str

    @property
    def is_front(self) -> bool:
        return self.role == 'front'

    @property
    def is_rear(self) -> bool:
        return self.role == 'rear'


@dataclass(frozen=True)
class CollectorConfig:
    """Defaults for the data collector node."""

    ckpt_dir: str | None
    task_id: str | None
    scene_dir: str


@dataclass(frozen=True)
class Qwen3RewardConfig:
    """Defaults for the online Qwen3-VL reward node."""

    model_path: Path
    prompt: str
    f_interval: int
    t_interval: int
    max_new_tokens: int
    source_tag: str


@dataclass(frozen=True)
class RewardConfig:
    qwen3: Qwen3RewardConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Fully resolved config consumed by nodes, collectors, and tools."""

    runtime: RuntimeParams
    paths: PathsConfig
    topics: TopicsConfig
    cameras: dict[str, CameraConfig]
    camera_order: tuple[str, ...]
    default_saved_camera_keys: tuple[str, ...]
    arm_profiles: dict[str, ArmProfile]
    collector: CollectorConfig
    reward: RewardConfig
    source_path: Path

    def camera_sync_topics(self) -> dict[str, str]:
        return {
            key: self.cameras[key].sync_topic
            for key in self.camera_order if key in self.cameras
        }

    def camera_topics_by_short_key(self) -> dict[str, str]:
        return {
            camera.short_key: camera.topic
            for camera in self.cameras.values()
        }

    def camera_labels_by_short_key(self) -> dict[str, str]:
        return {
            camera.short_key: camera.label
            for camera in self.cameras.values()
        }

    def camera_short_key_order(self) -> tuple[str, ...]:
        return tuple(self.cameras[key].short_key for key in self.camera_order)

    def default_saved_camera_names(self) -> tuple[str, ...]:
        return tuple(self.cameras[key].name
                     for key in self.default_saved_camera_keys)

    def camera_name(self, key: str) -> str:
        return self.cameras[key].name


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_runtime_config(config_file: OptionalPath = None) -> RuntimeConfig:
    """Load ``default.yaml`` plus an optional YAML override file."""
    data, source_path = _load_config_data(config_file)
    paths, path_context = _build_paths(data, source_path)
    cameras, camera_order, default_saved = _build_cameras(data)

    return RuntimeConfig(
        runtime=_build_runtime(data),
        paths=paths,
        topics=_build_topics(data),
        cameras=cameras,
        camera_order=camera_order,
        default_saved_camera_keys=default_saved,
        arm_profiles=_build_arms(data),
        collector=_build_collector(data),
        reward=_build_reward(data, path_context),
        source_path=source_path,
    )


def _load_config_data(config_file: OptionalPath) -> ConfigData:
    default_path = _require_default_config_path()
    data = _read_yaml(default_path)

    if not config_file or is_config_sentinel(config_file):
        return data, default_path

    override_path = Path(config_file).expanduser()
    if not override_path.is_file():
        raise FileNotFoundError(
            f'FluxDAgger config file not found: {override_path}')

    if override_path.resolve() == default_path.resolve():
        return data, default_path

    return _deep_update(data, _read_yaml(override_path)), override_path


def _read_yaml(path: Path) -> RawConfig:
    with path.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f'Configuration root must be a mapping: {path}')
    return loaded


def _deep_update(base: RawConfig, override: RawConfig) -> RawConfig:
    """Recursively merge an override YAML mapping onto the default mapping."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require_default_config_path() -> Path:
    if DEFAULT_CONFIG_PATH is None:
        raise FileNotFoundError('Unable to locate dagger config/default.yaml')
    return DEFAULT_CONFIG_PATH


def _source_default_config_path() -> Path | None:
    candidates = [
        _package_dir_from_file() / 'config' / 'default.yaml',
        _repo_root_from_file() / 'src' / 'dagger' / 'config' / 'default.yaml',
    ]

    try:
        import rospkg  # type: ignore

        candidates.append(
            Path(rospkg.RosPack().get_path('dagger')) / 'config' /
            'default.yaml')
    except Exception:
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _package_dir_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_CONFIG_PATH: Path | None = _source_default_config_path()

# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_runtime(data: RawConfig) -> RuntimeParams:
    raw = _section(data, 'runtime')
    return RuntimeParams(
        debug=bool(raw['debug']),
        master_publish_rate=float(raw['master_publish_rate']),
        puppet_publish_rate=float(raw['puppet_publish_rate']),
        sync_observation_rate=float(raw['sync_observation_rate']),
        data_collection_rate=float(raw['data_collection_rate']),
        sync_slop=float(raw['sync_slop']),
        synced_observation_pub_queue_size=int(
            raw['synced_observation_pub_queue_size']),
        synced_observation_queue_size=int(
            raw['synced_observation_queue_size']),
        synced_observation_drain_timeout=float(
            raw['synced_observation_drain_timeout']),
        task_text=str(raw['task_text']),
    )


def _build_paths(data: RawConfig,
                 source_path: Path) -> tuple[PathsConfig, PathContext]:
    raw = _section(data, 'paths')
    context = _initial_path_context(source_path)

    home_dir = _yaml_path(raw, 'home_dir', context)
    repo_dir = _yaml_path(raw, 'repo_dir', context)
    context.update({
        'home_dir': str(home_dir),
        'repo_dir': str(repo_dir),
    })

    return (
        PathsConfig(
            home_dir=home_dir,
            repo_dir=repo_dir,
            data_buffer_dir=_yaml_path(raw, 'data_buffer_dir', context),
            captures_dir=_yaml_path(raw, 'captures_dir', context),
            screenshots_dir=_yaml_path(raw, 'screenshots_dir', context),
            sync_fail_log_dir=_yaml_path(raw, 'sync_fail_log_dir', context),
            act_checkpoint_dir=_yaml_path(raw, 'act_checkpoint_dir', context),
            qwen3_reward_model=_yaml_path(raw, 'qwen3_reward_model', context),
        ),
        context,
    )


def _build_topics(data: RawConfig) -> TopicsConfig:
    return TopicsConfig(**_section(data, 'topics'))


def _build_cameras(
    data: RawConfig
) -> tuple[dict[str, CameraConfig], tuple[str, ...], tuple[str, ...]]:
    raw = _section(data, 'cameras')
    camera_items = raw.get('items', {})
    cameras = {
        key: CameraConfig(key=key, **values)
        for key, values in camera_items.items()
    }

    camera_order = tuple(raw.get('order', tuple(cameras.keys())))
    default_saved = tuple(raw.get('default_saved', tuple(cameras.keys())))
    return cameras, camera_order, default_saved


def _build_arms(data: RawConfig) -> dict[str, ArmProfile]:
    raw = _section(data, 'arms')
    home_end_pose = _float_tuple(raw.get('home_end_pose', ()))
    home_joints = {
        name: _float_tuple(values)
        for name, values in raw.get('home_joints', {}).items()
    }

    return {
        arm_name: _build_arm_profile(
            arm_name=arm_name,
            raw_profile=raw_profile,
            home_joints=home_joints,
            home_end_pose=home_end_pose,
        )
        for arm_name, raw_profile in raw.get('profiles', {}).items()
    }


def _build_arm_profile(
    arm_name: str,
    raw_profile: RawConfig,
    home_joints: dict[str, tuple[float, ...]],
    home_end_pose: tuple[float, ...],
) -> ArmProfile:
    home_joint_ref = raw_profile.get('home_joint', ())
    home_joint = _resolve_home_joint(home_joint_ref, home_joints)

    return ArmProfile(
        name=arm_name,
        role=str(raw_profile['role']),
        can_port=str(raw_profile['can_port']),
        joint_pub_topic=raw_profile.get('joint_pub_topic'),
        joint_sub_topic=str(raw_profile['joint_sub_topic']),
        joint_state_pub_topic=raw_profile.get('joint_state_pub_topic'),
        end_pose_pub_topic=raw_profile.get('end_pose_pub_topic'),
        human_sub_topic=raw_profile.get('human_sub_topic'),
        home_joint=home_joint,
        home_end_pose=home_end_pose,
    )


def _build_collector(data: RawConfig) -> CollectorConfig:
    raw = _section(data, 'collector')
    return CollectorConfig(
        ckpt_dir=raw.get('ckpt_dir'),
        task_id=raw.get('task_id'),
        scene_dir=str(raw['scene_dir']),
    )


def _build_reward(data: RawConfig, context: PathContext) -> RewardConfig:
    raw = _section(_section(data, 'reward'), 'qwen3')
    return RewardConfig(
        qwen3=Qwen3RewardConfig(
            model_path=_expand_path(raw['model_path'], context),
            prompt=str(raw['prompt']),
            f_interval=int(raw['f_interval']),
            t_interval=int(raw['t_interval']),
            max_new_tokens=int(raw['max_new_tokens']),
            source_tag=str(raw['source_tag']),
        ))


# ---------------------------------------------------------------------------
# Small conversion helpers
# ---------------------------------------------------------------------------


def _section(data: RawConfig, name: str) -> RawConfig:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f'Missing or invalid YAML section: {name}')
    return section


def _initial_path_context(source_path: Path) -> PathContext:
    return {
        'home_dir': str(Path.home()),
        'repo_dir': str(_repo_root_from_config_path(source_path)),
    }


def _repo_root_from_config_path(config_path: Path) -> Path:
    """Infer ``{repo_dir}`` from the selected YAML file location."""
    config_dir = config_path.resolve().parent
    package_dir = config_dir.parent
    if package_dir.name == 'dagger' and package_dir.parent.name == 'src':
        return package_dir.parent.parent
    return package_dir


def _yaml_path(raw: RawConfig, name: str, context: PathContext) -> Path:
    return _expand_path(raw[name], context)


def _expand_path(value: Any, context: PathContext) -> Path:
    text = str(value).format(**context)
    return Path(text).expanduser()


def _float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in (values or ()))


def _resolve_home_joint(
    home_joint_ref: Any,
    home_joints: dict[str, tuple[float, ...]],
) -> tuple[float, ...]:
    if isinstance(home_joint_ref, str):
        return home_joints.get(home_joint_ref, ())
    return _float_tuple(home_joint_ref)


# ---------------------------------------------------------------------------
# ROS helpers
# ---------------------------------------------------------------------------


def is_config_sentinel(value: Any) -> bool:
    """Return whether a ROS param means 'use the YAML config value'."""
    return value == CONFIG_VALUE_SENTINEL


def get_ros_param(name: str, default: Any, ns: str = '~') -> Any:
    """Read a ROS private param while respecting ``CONFIG_VALUE_SENTINEL``."""
    import rospy

    value = rospy.get_param(f'{ns}{name}', CONFIG_VALUE_SENTINEL)
    return default if is_config_sentinel(value) else value


def get_ros_override(
    name: str,
    default: Any,
    *,
    value_type: type | None = None,
    ns: str = '~',
) -> Any:
    """Read a ROS override and normalize it to the requested type."""
    value = get_ros_param(name, default, ns)

    if value_type is None:
        return value
    if value is None:
        return None
    if value_type is bool:
        return _as_bool(value)
    if value_type is list:
        return _as_list(value)
    return value_type(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('1', 'true', 'yes', 'on'):
            return True
        if normalized in ('0', 'false', 'no', 'off', ''):
            return False
    return bool(value)


def _as_list(value: Any) -> list:
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    return list(value)


def load_runtime_config_from_ros(ns: str = '~') -> RuntimeConfig:
    """Load the config selected by the node-private ``config_file`` param."""
    return load_runtime_config(get_ros_param('config_file', None, ns))


def as_str(path: Path) -> str:
    """Convert a ``Path`` to ``str`` for argparse and ROS params."""
    return str(path)
