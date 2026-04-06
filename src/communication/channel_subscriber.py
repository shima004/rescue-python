from __future__ import annotations

from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo, ScenarioInfoKeys
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.component.communication.channel_subscriber import (
  ChannelSubscriber,
)


class PlatoonChannelSubscriber(ChannelSubscriber):
  def subscribe(
    self,
    agent_info: AgentInfo,
    world_info: WorldInfo,
    scenario_info: ScenarioInfo,
  ) -> list[int]:
    agent = world_info.get_entity(agent_info.get_entity_id())
    if agent is None:
      return []

    number_of_channels: int = (
      scenario_info.get_value(ScenarioInfoKeys.COMMUNICATION_CHANNELS_COUNT, 1) - 1
    )

    subscriber_channel_limit: int = scenario_info.get_value(
      ScenarioInfoKeys.COMMUNICATION_CHANNELS_MAX_PLATOON, 1
    )

    channels = [
      i for i in range(1, min(number_of_channels + 1, subscriber_channel_limit + 1))
    ]

    return channels
