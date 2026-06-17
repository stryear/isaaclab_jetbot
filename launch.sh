#!/usr/bin/env bash
# ==============================================================================
# IsaacLabTutorial - Launch Script
# ==============================================================================
# Usage: ./launch.sh <command> [options]
#
# Prerequisites:
#   - NVIDIA Isaac Sim 4.5.0+ installed
#   - Isaac Lab framework installed
#   - NVIDIA GPU with CUDA support
# ==============================================================================

set -euo pipefail

# ---- Configuration -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME_JETBOT="Isaac-Lab-Tutorial-SphereFollow-Direct-v0"
TASK_NAME_TURTLEBOT3="Isaac-Lab-Tutorial-SphereFollow-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_LIDAR_NAV="Isaac-Lab-Tutorial-LidarNav-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_LIDAR_JUNCTION="Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_JUNCTION="Isaac-Lab-Tutorial-LidarShortNavJunction-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_UNIFIED="Isaac-Lab-Tutorial-LidarShortNavUnified-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_CURRICULUM_P1="Isaac-Lab-Tutorial-LidarShortNavCurriculumP1-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_V14="Isaac-Lab-Tutorial-LidarShortNavV14-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_V15="Isaac-Lab-Tutorial-LidarShortNavV15-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_V15A="Isaac-Lab-Tutorial-LidarShortNavV15A-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_V15B="Isaac-Lab-Tutorial-LidarShortNavV15B-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_V16="Isaac-Lab-Tutorial-LidarShortNavV16-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_V18="Isaac-Lab-Tutorial-LidarShortNavV18-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_V18A="Isaac-Lab-Tutorial-LidarShortNavV18A-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_DWB_OSCILLATION="Isaac-Lab-Tutorial-LidarShortNavDefectDwbOscillation-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleSymmetricGap-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SMALL_OBSTACLE_PRETRAIN="Isaac-Lab-Tutorial-LidarShortNavDefectSmallObstaclePretrain-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_WAYPOINT_BRIDGE="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleWaypointBridge-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_OFFSET_BRIDGE="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearOffsetBridge-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_CENTERING_BRIDGE="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearCenteringBridge-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_TRANSITION="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearTransition-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_FIXED_TRANSITION="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleRearFixedTransition-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_STAGE12_TRANSITION="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleStage12Transition-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_FINAL_CURRICULUM="Isaac-Lab-Tutorial-LidarShortNavDefectSingleObstacleFinalCurriculum-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_MPPI_CORNER_FAIL="Isaac-Lab-Tutorial-LidarShortNavDefectMppiCornerFail-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_DOOR_DEADLOCK="Isaac-Lab-Tutorial-LidarShortNavDefectDoorDeadlock-TurtleBot3-Direct-v0"
TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_U_SHAPE_TRAP="Isaac-Lab-Tutorial-LidarShortNavDefectUShapeTrap-TurtleBot3-Direct-v0"
DEFAULT_ROBOT="jetbot"
DEFAULT_MODE="sphere_follow"
DEFAULT_NUM_ENVS=100
DEFAULT_ALGORITHM="PPO"
DEFAULT_DEVICE="cuda:0"
DEFAULT_ML_FRAMEWORK="torch"

# ---- Colors ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---- Helper Functions --------------------------------------------------------
print_header() {
    echo -e "${CYAN}"
    echo "=================================================================="
    echo "  IsaacLabTutorial - Mobile Robot Navigation RL Environment"
    echo "=================================================================="
    echo -e "${NC}"
}

print_usage() {
    print_header
    echo -e "${GREEN}Usage:${NC} ./launch.sh <command> [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  install            Install the isaac_lab_tutorial package (editable mode)"
    echo "  fetch-assets       Download robot assets from .collect.mapping.json manifests"
    echo "  list-envs          List all registered environments"
    echo "  random-agent       Run environment with random actions"
    echo "  zero-agent         Run environment with zero actions"
    echo "  train              Train RL agent (PPO/AMP/IPPO/MAPPO/SAC) using skrl"
    echo "  play               Evaluate a trained agent checkpoint"
    echo "  select-ckpt        Select latest healthy LiDAR-nav checkpoint"
    echo "  ros2-bridge        Run LiDAR nav env with ROS2 cmd_vel/scan/odom bridge"
    echo "  help               Show this help message"
    echo ""
    echo -e "${YELLOW}Options (for random-agent / zero-agent):${NC}"
    echo "  --num_envs N       Number of parallel environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --device DEVICE    Compute device (default: ${DEFAULT_DEVICE})"
    echo "  --robot ROBOT      Robot profile: jetbot | turtlebot3_burger (default: ${DEFAULT_ROBOT})"
    echo "  --mode MODE        Task mode: sphere_follow | lidar_nav | lidar_junction | short_nav_junction | short_nav_unified | short_nav_curriculum_p1 | short_nav_v14 | short_nav_v15 | short_nav_v15a | short_nav_v15b | short_nav_v16 | short_nav_v18 | short_nav_v18a | dwb_oscillation | single_obstacle_symmetric_gap | small_obstacle_pretrain | single_obstacle_waypoint_bridge | single_obstacle_rear_offset_bridge | single_obstacle_rear_transition | single_obstacle_rear_fixed_transition | single_obstacle_stage12_transition | mppi_corner_fail | door_deadlock | u_shape_trap (default: ${DEFAULT_MODE})"
    echo ""
    echo -e "${YELLOW}Options (for fetch-assets):${NC}"
    echo "  --robot ROBOT      Asset profile: jetbot | turtlebot3_burger | all (default: all)"
    echo "  --force            Re-download assets even if files already exist"
    echo ""
    echo -e "${YELLOW}Options (for train):${NC}"
    echo "  --algorithm ALG    RL algorithm: PPO, AMP, IPPO, MAPPO, SAC (default: ${DEFAULT_ALGORITHM})"
    echo "  --num_envs N       Number of parallel environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --max_iterations N Maximum training iterations"
    echo "  --seed N           Random seed"
    echo "  --checkpoint PATH  Resume training from checkpoint"
    echo "  --ml_framework FW  ML framework: torch, jax, jax-numpy (default: ${DEFAULT_ML_FRAMEWORK})"
    echo "  --distributed      Enable multi-GPU training"
    echo "  --video            Record training videos"
    echo "  --robot ROBOT      Robot profile: jetbot | turtlebot3_burger (default: ${DEFAULT_ROBOT})"
    echo "  --mode MODE        Task mode: sphere_follow | lidar_nav | lidar_junction | short_nav_junction | short_nav_unified | short_nav_curriculum_p1 | short_nav_v14 | short_nav_v15 | short_nav_v15a | short_nav_v15b | short_nav_v16 | short_nav_v18 | short_nav_v18a | dwb_oscillation | single_obstacle_symmetric_gap | small_obstacle_pretrain | single_obstacle_waypoint_bridge | single_obstacle_rear_offset_bridge | single_obstacle_rear_transition | single_obstacle_rear_fixed_transition | single_obstacle_stage12_transition | mppi_corner_fail | door_deadlock | u_shape_trap (default: ${DEFAULT_MODE})"
    echo ""
    echo -e "${YELLOW}Options (for play):${NC}"
    echo "  --checkpoint PATH  Path to trained model checkpoint"
    echo "  --algorithm ALG    Algorithm used during training (default: ${DEFAULT_ALGORITHM})"
    echo "  --num_envs N       Number of environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --real-time        Run evaluation in real-time"
    echo "  --video            Record evaluation video"
    echo "  --robot ROBOT      Robot profile: jetbot | turtlebot3_burger (default: ${DEFAULT_ROBOT})"
    echo "  --mode MODE        Task mode: sphere_follow | lidar_nav | lidar_junction | short_nav_junction | short_nav_unified | short_nav_curriculum_p1 | short_nav_v14 | short_nav_v15 | short_nav_v15a | short_nav_v15b | short_nav_v16 | short_nav_v18 | short_nav_v18a | dwb_oscillation | single_obstacle_symmetric_gap | small_obstacle_pretrain | single_obstacle_waypoint_bridge | single_obstacle_rear_offset_bridge | single_obstacle_rear_transition | single_obstacle_rear_fixed_transition | single_obstacle_stage12_transition | mppi_corner_fail | door_deadlock | u_shape_trap (default: ${DEFAULT_MODE})"
    echo ""
    echo -e "${YELLOW}Options (for select-ckpt):${NC}"
    echo "  --root DIR         Log root (default: logs/skrl/lidar_nav_direct)"
    echo "  --strict           Only accept healthy-signature checkpoints"
    echo "  --verbose          Print skip reasons to stderr"
    echo ""
    echo -e "${YELLOW}Options (for ros2-bridge):${NC}"
    echo "  --robot ROBOT      Robot profile (default: turtlebot3_burger)"
    echo "  --mode MODE        Task mode (default: lidar_nav)"
    echo "  --topic_scan NAME  LaserScan topic (default: /scan)"
    echo "  --topic_odom NAME  Odometry topic (default: /odom)"
    echo "  --topic_cmd NAME   Twist command topic (default: /cmd_vel)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./launch.sh install"
    echo "  ./launch.sh fetch-assets --robot all"
    echo "  ./launch.sh train --algorithm PPO --num_envs 100"
    echo "  ./launch.sh train --algorithm PPO --num_envs 100 --robot turtlebot3_burger"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode lidar_nav --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode lidar_junction --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_junction --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_unified --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_curriculum_p1 --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_v14 --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_v15 --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_v15a --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_v15b --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_v16 --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_v18 --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_v18a --num_envs 64 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode single_obstacle_symmetric_gap --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode small_obstacle_pretrain --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode single_obstacle_waypoint_bridge --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode single_obstacle_rear_offset_bridge --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode single_obstacle_rear_transition --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode single_obstacle_rear_fixed_transition --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode single_obstacle_stage12_transition --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode mppi_corner_fail --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode door_deadlock --num_envs 100 --algorithm PPO"
    echo "  ./launch.sh train --robot turtlebot3_burger --mode short_nav_curriculum_p1 --num_envs 64 --algorithm SAC"
    echo "  ./launch.sh play --robot turtlebot3_burger --mode short_nav_junction --checkpoint logs/skrl/.../checkpoints/best_agent.pt"
    echo "  ./launch.sh ros2-bridge --robot turtlebot3_burger --mode lidar_nav"
    echo "  ./launch.sh train --algorithm AMP --num_envs 50 --max_iterations 1000"
    echo "  ./launch.sh play --checkpoint logs/skrl/.../checkpoints/best_agent.pt"
    echo "  ./launch.sh random-agent --num_envs 64"
    echo "  ./launch.sh select-ckpt"
    echo ""
}

check_prerequisites() {
    echo -e "${CYAN}[CHECK]${NC} Verifying prerequisites..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} Python 3 is not installed."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "  Python version: ${GREEN}${PYTHON_VERSION}${NC}"

    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo -e "  GPU: ${GREEN}${GPU_NAME}${NC}"
    else
        echo -e "  ${YELLOW}[WARN]${NC} nvidia-smi not found. NVIDIA GPU required for Isaac Sim."
    fi

    # Check if Isaac Sim / Isaac Lab are importable
    if python3 -c "import isaaclab" 2>/dev/null; then
        echo -e "  Isaac Lab: ${GREEN}available${NC}"
    else
        echo -e "  Isaac Lab: ${RED}not found${NC} - install Isaac Lab first"
        echo -e "  See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
    fi

    echo ""
}

resolve_task_name() {
    local robot="${1,,}"
    local mode="${2,,}"

    case "${mode}" in
        sphere_follow|sphere|follow|"")
            case "${robot}" in
                jetbot|jb)
                    echo "${TASK_NAME_JETBOT}"
                    ;;
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} Unknown robot profile: ${1}" >&2
                    echo -e "  Supported values: ${YELLOW}jetbot${NC}, ${YELLOW}turtlebot3_burger${NC}" >&2
                    exit 1
                    ;;
            esac
            ;;
        lidar_nav|lidar|nav)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_LIDAR_NAV}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} lidar_nav mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        lidar_junction|junction)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_LIDAR_JUNCTION}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} lidar_junction mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_junction|short_junction|short_nav)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_JUNCTION}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_junction mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_unified|unified|short_unified)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_UNIFIED}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_unified mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_curriculum_p1|curriculum_p1|short_curriculum_p1)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_CURRICULUM_P1}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_curriculum_p1 mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_v14|v14|short_v14)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_V14}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_v14 mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_v15|v15|short_v15)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_V15}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_v15 mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_v15a|v15a|short_v15a)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_V15A}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_v15a mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_v15b|v15b|short_v15b)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_V15B}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_v15b mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_v16|v16|short_v16)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_V16}"
                ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_v16 mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_v18|v18|short_v18)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_V18}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_v18 mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_v18a|v18a|short_v18a)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_V18A}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} short_nav_v18a mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_dwb_oscillation|dwb_oscillation|defect_dwb)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_DWB_OSCILLATION}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} dwb_oscillation mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_symmetric_gap|single_obstacle_symmetric_gap|single_obstacle_gap|single_gap|defect_single_gap)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_SYMMETRIC_GAP}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_symmetric_gap mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_small_obstacle_pretrain|small_obstacle_pretrain|small_obstacle_gap_pretrain|easy_single_gap)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SMALL_OBSTACLE_PRETRAIN}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} small_obstacle_pretrain mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_waypoint_bridge|single_obstacle_waypoint_bridge|single_obstacle_bridge|waypoint_bridge)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_WAYPOINT_BRIDGE}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_waypoint_bridge mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_rear_offset_bridge|single_obstacle_rear_offset_bridge|rear_offset_bridge|rear_bridge)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_OFFSET_BRIDGE}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_rear_offset_bridge mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_rear_centering_bridge|single_obstacle_rear_centering_bridge|rear_centering_bridge|centering_bridge)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_CENTERING_BRIDGE}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_rear_centering_bridge mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_rear_transition|single_obstacle_rear_transition|rear_transition|transition_bridge)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_TRANSITION}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_rear_transition mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_rear_fixed_transition|single_obstacle_rear_fixed_transition|rear_fixed_transition|fixed_rear_transition)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_REAR_FIXED_TRANSITION}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_rear_fixed_transition mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_stage12_transition|single_obstacle_stage12_transition|stage12_transition|final_stage12_transition)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_STAGE12_TRANSITION}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_stage12_transition mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_single_obstacle_final_curriculum|single_obstacle_final_curriculum|final_curriculum_bridge|final_curriculum)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_SINGLE_OBSTACLE_FINAL_CURRICULUM}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} single_obstacle_final_curriculum mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_mppi_corner_fail|mppi_corner_fail|defect_mppi|mppi)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_MPPI_CORNER_FAIL}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} mppi_corner_fail mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_door_deadlock|door_deadlock|defect_door|door)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_DOOR_DEADLOCK}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} door_deadlock mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        short_nav_defect_u_shape_trap|u_shape_trap|defect_u_shape|ushape)
            case "${robot}" in
                turtlebot3_burger|turtlebot3|turtlebot|tb3|burger)
                    echo "${TASK_NAME_TURTLEBOT3_SHORT_NAV_DEFECT_U_SHAPE_TRAP}"
                    ;;
                *)
                    echo -e "${RED}[ERROR]${NC} u_shape_trap mode currently supports only turtlebot3_burger." >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown mode: ${2}" >&2
            echo -e "  Supported values: ${YELLOW}sphere_follow${NC}, ${YELLOW}lidar_nav${NC}, ${YELLOW}lidar_junction${NC}, ${YELLOW}short_nav_junction${NC}, ${YELLOW}short_nav_unified${NC}, ${YELLOW}short_nav_curriculum_p1${NC}, ${YELLOW}short_nav_v14${NC}, ${YELLOW}short_nav_v15${NC}, ${YELLOW}short_nav_v15a${NC}, ${YELLOW}short_nav_v15b${NC}, ${YELLOW}short_nav_v16${NC}, ${YELLOW}short_nav_v18${NC}, ${YELLOW}short_nav_v18a${NC}, ${YELLOW}dwb_oscillation${NC}, ${YELLOW}single_obstacle_symmetric_gap${NC}, ${YELLOW}small_obstacle_pretrain${NC}, ${YELLOW}single_obstacle_waypoint_bridge${NC}, ${YELLOW}single_obstacle_rear_offset_bridge${NC}, ${YELLOW}single_obstacle_rear_transition${NC}, ${YELLOW}single_obstacle_rear_fixed_transition${NC}, ${YELLOW}single_obstacle_stage12_transition${NC}, ${YELLOW}mppi_corner_fail${NC}, ${YELLOW}door_deadlock${NC}, ${YELLOW}u_shape_trap${NC}" >&2
            exit 1
            ;;
    esac
}

SELECTED_TASK="${TASK_NAME_JETBOT}"
FILTERED_ARGS=()

extract_profile_args() {
    local default_mode="${1}"
    shift
    local robot="${DEFAULT_ROBOT}"
    local mode="${default_mode}"
    FILTERED_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --robot)
                if [[ $# -lt 2 ]]; then
                    echo -e "${RED}[ERROR]${NC} Missing value for --robot"
                    exit 1
                fi
                robot="$2"
                shift 2
                ;;
            --robot=*)
                robot="${1#*=}"
                shift
                ;;
            --mode|--task_mode)
                if [[ $# -lt 2 ]]; then
                    echo -e "${RED}[ERROR]${NC} Missing value for --mode"
                    exit 1
                fi
                mode="$2"
                shift 2
                ;;
            --mode=*|--task_mode=*)
                mode="${1#*=}"
                shift
                ;;
            *)
                FILTERED_ARGS+=("$1")
                shift
                ;;
        esac
    done

    SELECTED_TASK="$(resolve_task_name "${robot}" "${mode}")"
}

# ---- Commands ----------------------------------------------------------------

cmd_install() {
    echo -e "${CYAN}[INSTALL]${NC} Installing isaac_lab_tutorial package..."
    check_prerequisites
    cd "${SCRIPT_DIR}/source/isaac_lab_tutorial"
    pip install -e .
    echo -e "${GREEN}[DONE]${NC} Package installed successfully."
    echo -e "  Run '${YELLOW}./launch.sh list-envs${NC}' to verify."
}

cmd_fetch_assets() {
    echo -e "${CYAN}[ASSETS]${NC} Fetching robot assets..."
    python3 -u "${SCRIPT_DIR}/scripts/download_assets.py" "$@"
}

cmd_list_envs() {
    echo -e "${CYAN}[LIST]${NC} Listing registered environments..."
    env PYTHONNOUSERSITE=1 PYTHONPATH= \
        "${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}" -p "${SCRIPT_DIR}/scripts/list_envs.py"
}

cmd_random_agent() {
    extract_profile_args "${DEFAULT_MODE}" "$@"
    echo -e "${CYAN}[RUN]${NC} Running random action agent..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    env PYTHONNOUSERSITE=1 PYTHONPATH= \
        "${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}" -p "${SCRIPT_DIR}/scripts/random_agent.py" \
        --task "${SELECTED_TASK}" \
        --num_envs "${DEFAULT_NUM_ENVS}" \
        "${FILTERED_ARGS[@]}"
}

cmd_zero_agent() {
    extract_profile_args "${DEFAULT_MODE}" "$@"
    echo -e "${CYAN}[RUN]${NC} Running zero action agent..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    env PYTHONNOUSERSITE=1 PYTHONPATH= \
        "${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}" -p "${SCRIPT_DIR}/scripts/zero_agent.py" \
        --task "${SELECTED_TASK}" \
        --num_envs "${DEFAULT_NUM_ENVS}" \
        "${FILTERED_ARGS[@]}"
}

cmd_train() {
    extract_profile_args "${DEFAULT_MODE}" "$@"
    echo -e "${CYAN}[TRAIN]${NC} Starting RL training..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    env PYTHONNOUSERSITE=1 PYTHONPATH= \
        "${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}" -p "${SCRIPT_DIR}/scripts/skrl/train.py" \
        --task "${SELECTED_TASK}" \
        "${FILTERED_ARGS[@]}"
}

cmd_play() {
    extract_profile_args "${DEFAULT_MODE}" "$@"
    echo -e "${CYAN}[PLAY]${NC} Evaluating trained agent..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    env PYTHONNOUSERSITE=1 PYTHONPATH= \
        "${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}" -p "${SCRIPT_DIR}/scripts/skrl/play.py" \
        --task "${SELECTED_TASK}" \
        "${FILTERED_ARGS[@]}"
}

cmd_ros2_bridge() {
    extract_profile_args "lidar_nav" --robot turtlebot3_burger "$@"
    echo -e "${CYAN}[ROS2]${NC} Starting ROS2 bridge loop..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    python3 "${SCRIPT_DIR}/scripts/ros2_bridge_lidar_nav.py" \
        --task "${SELECTED_TASK}" \
        --num_envs 1 \
        "${FILTERED_ARGS[@]}"
}

cmd_select_ckpt() {
    local root_dir="${SCRIPT_DIR}/logs/skrl/lidar_nav_direct"
    local strict_mode=0
    local verbose_mode=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --root)
                if [[ $# -lt 2 ]]; then
                    echo -e "${RED}[ERROR]${NC} Missing value for --root"
                    exit 1
                fi
                root_dir="$2"
                shift 2
                ;;
            --root=*)
                root_dir="${1#*=}"
                shift
                ;;
            --strict)
                strict_mode=1
                shift
                ;;
            --verbose)
                verbose_mode=1
                shift
                ;;
            *)
                echo -e "${RED}[ERROR]${NC} Unknown option for select-ckpt: $1"
                exit 1
                ;;
        esac
    done
    local args=(--root "${root_dir}")
    [[ "${strict_mode}" == "1" ]] && args+=(--strict)
    [[ "${verbose_mode}" == "1" ]] && args+=(--verbose)
    "${SCRIPT_DIR}/scripts/skrl/select_healthy_checkpoint.sh" "${args[@]}"
}

# ---- Main Entry Point --------------------------------------------------------

if [[ $# -eq 0 ]]; then
    print_usage
    exit 0
fi

COMMAND="$1"
shift

case "${COMMAND}" in
    install)
        cmd_install
        ;;
    fetch-assets|fetch_assets|assets)
        cmd_fetch_assets "$@"
        ;;
    list-envs|list_envs)
        cmd_list_envs
        ;;
    random-agent|random_agent)
        cmd_random_agent "$@"
        ;;
    zero-agent|zero_agent)
        cmd_zero_agent "$@"
        ;;
    train)
        cmd_train "$@"
        ;;
    play|eval|evaluate)
        cmd_play "$@"
        ;;
    ros2-bridge|ros2_bridge|bridge)
        cmd_ros2_bridge "$@"
        ;;
    select-ckpt|select_ckpt|pick-ckpt|pick_ckpt)
        cmd_select_ckpt "$@"
        ;;
    check)
        check_prerequisites
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}[ERROR]${NC} Unknown command: ${COMMAND}"
        echo ""
        print_usage
        exit 1
        ;;
esac
