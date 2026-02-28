from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_JETBOT_USD = Path(__file__).resolve().parents[1] / "assets" / "Jetbot" / "jetbot.usd"
_JETBOT_NUCLEUS_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
_JETBOT_USD_PATH = str(_JETBOT_USD) if _JETBOT_USD.exists() else _JETBOT_NUCLEUS_USD

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=_JETBOT_USD_PATH),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
