from adf_core_python.core.agent.communication.standard.bundle.information.message_ambulance_team import (
  MessageAmbulanceTeam,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_building import (
  MessageBuilding,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_civilian import (
  MessageCivilian,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_fire_brigade import (
  MessageFireBrigade,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_police_force import (
  MessagePoliceForce,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_road import (
  MessageRoad,
)
from rcrscore.entities import (
  AmbulanceTeam,
  Blockade,
  Building,
  Civilian,
  Entity,
  FireBrigade,
  Human,
  PoliceForce,
  Road,
)


def message_to_object(
  message: MessageAmbulanceTeam
  | MessageFireBrigade
  | MessagePoliceForce
  | MessageCivilian
  | MessageBuilding
  | MessageRoad,
) -> Entity | None:
  if isinstance(message, MessageAmbulanceTeam):
    return ambulance_team_message_to_object(message)
  elif isinstance(message, MessageFireBrigade):
    return fire_brigade_message_to_object(message)
  elif isinstance(message, MessagePoliceForce):
    return police_force_message_to_object(message)
  elif isinstance(message, MessageCivilian):
    return civilian_message_to_object(message)
  elif isinstance(message, MessageBuilding):
    return building_message_to_object(message)
  elif isinstance(message, MessageRoad):
    return road_message_to_object(message)
  else:
    raise ValueError(f"Unsupported message type: {type(message).__name__}")


def message_to_object_of_human(
  message: MessageAmbulanceTeam
  | MessageFireBrigade
  | MessagePoliceForce
  | MessageCivilian,
) -> Human | None:
  if isinstance(message, MessageAmbulanceTeam):
    return ambulance_team_message_to_object(message)
  elif isinstance(message, MessageFireBrigade):
    return fire_brigade_message_to_object(message)
  elif isinstance(message, MessagePoliceForce):
    return police_force_message_to_object(message)
  elif isinstance(message, MessageCivilian):
    return civilian_message_to_object(message)
  else:
    raise ValueError(f"Unsupported message type: {type(message).__name__}")


def ambulance_team_message_to_object(
  message_ambulance_team: MessageAmbulanceTeam,
) -> AmbulanceTeam | None:
  entity_id = message_ambulance_team.get_ambulance_team_entity_id()
  if entity_id is None:
    return None

  ambulance = AmbulanceTeam(entity_id.get_value())
  if (hp := message_ambulance_team.get_ambulance_team_hp()) is not None:
    ambulance.set_hp(hp)
  if (damage := message_ambulance_team.get_ambulance_team_damage()) is not None:
    ambulance.set_damage(damage)
  if (buriedness := message_ambulance_team.get_ambulance_team_buriedness()) is not None:
    ambulance.set_buriedness(buriedness)
  if (position := message_ambulance_team.get_ambulance_team_position()) is not None:
    ambulance.set_position(position)

  return ambulance


def fire_brigade_message_to_object(
  message_fire_brigade: MessageFireBrigade,
) -> FireBrigade | None:
  entity_id = message_fire_brigade.get_fire_brigade_entity_id()
  if entity_id is None:
    return None

  fire_brigade = FireBrigade(entity_id.get_value())
  if (hp := message_fire_brigade.get_fire_brigade_hp()) is not None:
    fire_brigade.set_hp(hp)
  if (damage := message_fire_brigade.get_fire_brigade_damage()) is not None:
    fire_brigade.set_damage(damage)
  if (buriedness := message_fire_brigade.get_fire_brigade_buriedness()) is not None:
    fire_brigade.set_buriedness(buriedness)
  if (position := message_fire_brigade.get_fire_brigade_position()) is not None:
    fire_brigade.set_position(position)
  if (water := message_fire_brigade.get_fire_brigade_water()) is not None:
    fire_brigade.set_water(water)

  return fire_brigade


def police_force_message_to_object(
  message_police_force: MessagePoliceForce,
) -> PoliceForce | None:
  entity_id = message_police_force.get_police_force_entity_id()
  if entity_id is None:
    return None

  police_force = PoliceForce(entity_id.get_value())
  if (hp := message_police_force.get_police_force_hp()) is not None:
    police_force.set_hp(hp)
  if (damage := message_police_force.get_police_force_damage()) is not None:
    police_force.set_damage(damage)
  if (buriedness := message_police_force.get_police_force_buriedness()) is not None:
    police_force.set_buriedness(buriedness)
  if (position := message_police_force.get_police_force_position()) is not None:
    police_force.set_position(position)

  return police_force


def civilian_message_to_object(
  message_civilian: MessageCivilian,
) -> Civilian | None:
  entity_id = message_civilian.get_civilian_entity_id()
  if entity_id is None:
    return None

  civilian = Civilian(entity_id.get_value())
  if (hp := message_civilian.get_civilian_hp()) is not None:
    civilian.set_hp(hp)
  if (damage := message_civilian.get_civilian_damage()) is not None:
    civilian.set_damage(damage)
  if (buriedness := message_civilian.get_civilian_buriedness()) is not None:
    civilian.set_buriedness(buriedness)
  if (position := message_civilian.get_civilian_position()) is not None:
    civilian.set_position(position)

  return civilian


def building_message_to_object(
  message_building: MessageBuilding,
) -> Building | None:
  entity_id = message_building.get_building_entity_id()
  if entity_id is None:
    return None

  building = Building(entity_id.get_value())
  if (fieryness := message_building.get_building_fireyness()) is not None:
    building.set_fieryness(fieryness)
  if (brokenness := message_building.get_building_brokenness()) is not None:
    building.set_brokenness(brokenness)
  if (temperature := message_building.get_building_temperature()) is not None:
    building.set_temperature(temperature)

  return building


def road_message_to_object(
  message_road: MessageRoad,
) -> Road | None:

  entity_id = message_road.get_road_entity_id()
  if entity_id is None:
    return None

  blockade_entity_id = message_road.get_road_blockade_entity_id()
  if blockade_entity_id is None:
    return None

  road_blockade = Blockade(blockade_entity_id.get_value())
  if (repair_cost := message_road.get_road_blockade_repair_cost()) is not None:
    road_blockade.set_repair_cost(repair_cost)
  if (x := message_road.get_road_blockade_x()) is not None:
    road_blockade.set_x(x)
  if (y := message_road.get_road_blockade_y()) is not None:
    road_blockade.set_y(y)
