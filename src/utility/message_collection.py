from dataclasses import dataclass

from adf_core_python.core.agent.info.world_info import WorldInfo
from rcrscore.entities import Human


@dataclass
class MessageHumanInfo:
  human: Human
  step_ago: int


def increment_step_ago_for_human_messages(
  messages: list[MessageHumanInfo],
) -> list[MessageHumanInfo]:
  for message in messages:
    message.step_ago += 1
  return messages


def clear_percepted_human_messages(
  messages: list[MessageHumanInfo],
  world_info: WorldInfo,
) -> list[MessageHumanInfo]:
  return [
    message
    for message in messages
    if message.human.get_entity_id()
    not in world_info.get_change_set().get_changed_entities()
  ]
