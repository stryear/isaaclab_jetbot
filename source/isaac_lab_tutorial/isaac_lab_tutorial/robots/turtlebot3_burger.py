from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_TURTLEBOT3_BURGER_USD = (
    Path(__file__).resolve().parents[1] / "assets" / "Turtlebot" / "turtlebot3_burger.usd"
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
