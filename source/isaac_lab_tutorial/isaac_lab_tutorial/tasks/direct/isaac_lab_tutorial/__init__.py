# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Isaac-Lab-Tutorial-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:IsaacLabTutorialEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:IsaacLabTutorialEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-SphereFollow-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:SphereFollowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:SphereFollowEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sphere_follow_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-SphereFollow-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:SphereFollowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:SphereFollowTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sphere_follow_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarNavTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_nav_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarCorridor-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarCorridorTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_corridor_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarJunctionTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_junction_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavJunction-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavJunctionTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_junction_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavUnified-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavUnifiedTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_unified_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavCurriculumP1-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavUnifiedCurriculumTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_curriculum_p1_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_curriculum_p1_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV14-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV14TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v14_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV15-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV15TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v15_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV15A-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV15ATurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v15a_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV15B-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV15BTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v15b_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV16-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV16TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v16_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV17-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV17TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v17_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV18-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV18TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v18_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavV18A-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavV18ATurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v18a_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemo-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoHard-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoHardTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_hard_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoHardV1-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoHardV1TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_hard_v1_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoMultiV1-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoMultiV1TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoMultiV1Bridge-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoMultiV1BridgeTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoMultiV1BridgeLSTM-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoMultiV1BridgeTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_lstm_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoScene4Easy-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoScene4EasyTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoScene4MidTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid2-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoScene4Mid2TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid2Obs1-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoScene4Mid2Obs1TurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoScene4Mid2Obs1Open-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoScene4Mid2Obs1OpenTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDemoScene4WallOnly-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDemoScene4WallOnlyTurtleBot3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_demo_multi_v1_bridge_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectDwbOscillationTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_dwb_finetune_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillationReordered-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectDwbOscillationTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_dwb_reordered_obs_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleSymmetricGap-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleSymmetricGapTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSmallObstaclePretrain-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSmallObstaclePretrainTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleWaypointBridge-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleWaypointBridgeTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearOffsetBridge-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleRearOffsetBridgeTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearCenteringBridge-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleRearCenteringBridgeTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearTransition-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleRearTransitionTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearFixedTransition-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleRearFixedTransitionTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleStage12Transition-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleStage12TransitionTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleFinalCurriculum-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectSingleObstacleFinalCurriculumTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_single_obstacle_gap_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectMppiCornerFail-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectMppiCornerFailTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v18a_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectDoorDeadlock-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectDoorDeadlockTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_defect_door_deadlock_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lab-Tutorial-LidarShortNavDefectUShapeTrap-TurtleBot3-Direct-v0",
    entry_point=f"{__name__}.isaac_lab_tutorial_env:LidarNavTurtleBot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.isaac_lab_tutorial_env_cfg:LidarShortNavDefectUShapeTrapTurtleBot3EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lidar_short_nav_v18a_ppo_cfg.yaml",
    },
)
