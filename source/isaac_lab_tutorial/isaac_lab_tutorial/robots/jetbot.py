from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_JETBOT_USD = Path(__file__).resolve().parents[1] / "assets" / "Jetbot" / "jetbot.usd"

if not _JETBOT_USD.exists():
    raise FileNotFoundError(
        f"JetBot USD not found: {_JETBOT_USD}\n"
        "Run './launch.sh fetch-assets --robot jetbot' to download robot assets."
    )

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=str(_JETBOT_USD)),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
