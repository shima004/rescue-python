from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.communication.standard.bundle.standard_message import (
  StandardMessage,
)
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo, ScenarioInfoKeys
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.component.communication.communication_message import (
  CommunicationMessage,
)
from adf_core_python.core.component.communication.message_coordinator import (
  MessageCoordinator,
)


class PlatoonMessageCoordinator(MessageCoordinator):
  def coordinate(
    self,
    agent_info: AgentInfo,
    world_info: WorldInfo,
    scenario_info: ScenarioInfo,
    message_manager: MessageManager,
    send_message_list: list[CommunicationMessage],
    channel_send_message_list: list[list[CommunicationMessage]],
  ) -> None:
    radio_messages: list[StandardMessage] = []
    voice_messages: list[StandardMessage] = []

    for msg in send_message_list:
      if not isinstance(msg, StandardMessage):
        continue
      if msg.is_wireless_message():
        radio_messages.append(msg)
      else:
        voice_messages.append(msg)

    radio_messages.sort(key=lambda m: m.get_priority(), reverse=True)
    voice_messages.sort(key=lambda m: m.get_priority(), reverse=True)

    number_of_channels: int = (
      scenario_info.get_value(ScenarioInfoKeys.COMMUNICATION_CHANNELS_COUNT, 1) - 1
    )
    subscriber_channel_limit: int = scenario_info.get_value(
      ScenarioInfoKeys.COMMUNICATION_CHANNELS_MAX_PLATOON, 1
    )
    size_of_channels = [
      scenario_info.get_value("comms.channels." + str(channel) + ".bandwidth", 0)
      for channel in range(1, min(number_of_channels + 1, subscriber_channel_limit + 1))
    ]

    for radio_message in radio_messages:
      # Assign to the channel with the most remaining capacity
      max_capacity_channel_index = max(
        range(1, min(number_of_channels + 1, subscriber_channel_limit + 1)),
        key=lambda c: size_of_channels[c - 1],
      )
      channel_send_message_list[max_capacity_channel_index].append(radio_message)
      size_of_channels[max_capacity_channel_index - 1] -= radio_message.get_bit_size()

    channel_send_message_list[0].extend(voice_messages)
