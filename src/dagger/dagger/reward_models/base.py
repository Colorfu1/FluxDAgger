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
"""Abstract reward model interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseRewardModel(ABC):
    """Common interface for reward model implementations."""

    @abstractmethod
    def setup(self, config: Dict[str, Any]) -> None:
        """Load model weights and initialize runtime resources.

        Args:
            config: Model-specific configuration dictionary.
        """
        ...

    @abstractmethod
    def on_episode_reset(self) -> None:
        """Reset internal state when an episode changes."""
        ...

    @abstractmethod
    def predict(self, observation: Any) -> Optional[Dict[str, Any]]:
        """Return a reward result for one observation, or ``None``.

        Result dictionaries follow ``RewardSignal.msg`` fields:
            - value (float): reward value.
            - done (bool): whether the task is complete.
            - confidence (float): prediction confidence.
            - aux_values (list[float]): auxiliary values.
            - source (str): reward source identifier.

        Return ``None`` when the model is not ready to infer yet.

        Args:
            observation: Implementation-specific observation payload.
        """
        ...

    @property
    @abstractmethod
    def source_tag(self) -> str:
        """Reward model identifier written to ``RewardSignal.source``."""
        ...
