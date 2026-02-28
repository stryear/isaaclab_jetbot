from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_TURTLEBOT3_BURGER_USD = (
    Path(__file__).resolve().parents[1] / "assets" / "Turtlebot" / "turtlebot3_burger.usd"
)

if not _TURTLEBOT3_BURGER_USD.exists():
    raise FileNotFoundError(
        f"TurtleBot3 Burger USD not found: {_TURTLEBOT3_BURGER_USD}\n"
        "Run './launch.sh fetch-assets --robot turtlebot3_burger' to download robot assets."
    )


TURTLEBOT3_BURGER_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=str(_TURTLEBOT3_BURGER_USD)),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            damping=None,
            stiffness=None,
        )
    },
)
