from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from rcrscore.entities import AmbulanceTeam, Civilian, EntityID, Human, Refuge

HUMAN_MAX_HP = 10000


def is_buried(entity: Human) -> bool:
  buriedness = entity.get_buriedness()
  if buriedness is None:
    return False
  return buriedness > 0


def is_damaged(entity: Human) -> bool:
  hp = entity.get_hp()
  if hp is None:
    return False
  damage = entity.get_damage()
  if damage is None:
    return False
  return damage > 0 or hp < HUMAN_MAX_HP


def is_alived(entity: Human) -> bool:
  hp = entity.get_hp()
  if hp is None:
    return False
  return hp > 0


def is_civilian(entity: Human) -> bool:
  return isinstance(entity, Civilian)


def is_transporting(entity: Human, world_info: WorldInfo) -> bool:
  position_entity_id = entity.get_position()
  if position_entity_id is None:
    return False
  position_entity = world_info.get_entity(position_entity_id)
  if position_entity is None:
    return False
  return isinstance(position_entity, AmbulanceTeam)


def is_transporting_by_another_ambulance(
  entity: Human, world_info: WorldInfo, ambulance_entity_id: EntityID
) -> bool:
  position_entity_id = entity.get_position()
  if position_entity_id is None:
    return False
  position_entity = world_info.get_entity(position_entity_id)
  if position_entity is None:
    return False
  return (
    isinstance(position_entity, AmbulanceTeam)
    and not position_entity_id == ambulance_entity_id
  )


def is_transported_to_refuge(entity: Human, world_info: WorldInfo) -> bool:
  position_entity_id = entity.get_position()
  if position_entity_id is None:
    return False
  position_entity = world_info.get_entity(position_entity_id)
  if position_entity is None:
    return False
  return isinstance(position_entity, Refuge)


def is_ghost_human(human: Human, world_info: WorldInfo, agent_info: AgentInfo) -> bool:
  human_position_entity_id = human.get_position()
  my_position_entity_id = agent_info.get_position_entity_id()
  if human_position_entity_id is None or my_position_entity_id is None:
    return False
  if human_position_entity_id != my_position_entity_id:
    return False

  if human.get_entity_id() in world_info.get_change_set().get_changed_entities():
    return False

  return True
