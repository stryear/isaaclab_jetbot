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
DEFAULT_ROBOT="jetbot"
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
    echo "  train              Train RL agent (PPO/AMP) using skrl"
    echo "  play               Evaluate a trained agent checkpoint"
    echo "  help               Show this help message"
    echo ""
    echo -e "${YELLOW}Options (for random-agent / zero-agent):${NC}"
    echo "  --num_envs N       Number of parallel environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --device DEVICE    Compute device (default: ${DEFAULT_DEVICE})"
    echo "  --robot ROBOT      Robot profile: jetbot | turtlebot3_burger (default: ${DEFAULT_ROBOT})"
    echo ""
    echo -e "${YELLOW}Options (for fetch-assets):${NC}"
    echo "  --robot ROBOT      Asset profile: jetbot | turtlebot3_burger | all (default: all)"
    echo "  --force            Re-download assets even if files already exist"
    echo ""
    echo -e "${YELLOW}Options (for train):${NC}"
    echo "  --algorithm ALG    RL algorithm: PPO, AMP, IPPO, MAPPO (default: ${DEFAULT_ALGORITHM})"
    echo "  --num_envs N       Number of parallel environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --max_iterations N Maximum training iterations"
    echo "  --seed N           Random seed"
    echo "  --checkpoint PATH  Resume training from checkpoint"
    echo "  --ml_framework FW  ML framework: torch, jax, jax-numpy (default: ${DEFAULT_ML_FRAMEWORK})"
    echo "  --distributed      Enable multi-GPU training"
    echo "  --video            Record training videos"
    echo "  --robot ROBOT      Robot profile: jetbot | turtlebot3_burger (default: ${DEFAULT_ROBOT})"
    echo ""
    echo -e "${YELLOW}Options (for play):${NC}"
    echo "  --checkpoint PATH  Path to trained model checkpoint"
    echo "  --algorithm ALG    Algorithm used during training (default: ${DEFAULT_ALGORITHM})"
    echo "  --num_envs N       Number of environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --real-time        Run evaluation in real-time"
    echo "  --video            Record evaluation video"
    echo "  --robot ROBOT      Robot profile: jetbot | turtlebot3_burger (default: ${DEFAULT_ROBOT})"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./launch.sh install"
    echo "  ./launch.sh fetch-assets --robot all"
    echo "  ./launch.sh train --algorithm PPO --num_envs 100"
    echo "  ./launch.sh train --algorithm PPO --num_envs 100 --robot turtlebot3_burger"
    echo "  ./launch.sh train --algorithm AMP --num_envs 50 --max_iterations 1000"
    echo "  ./launch.sh play --checkpoint logs/skrl/.../checkpoints/best_agent.pt"
    echo "  ./launch.sh random-agent --num_envs 64"
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
}

SELECTED_TASK="${TASK_NAME_JETBOT}"
FILTERED_ARGS=()

extract_robot_arg() {
    local robot="${DEFAULT_ROBOT}"
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
            *)
                FILTERED_ARGS+=("$1")
                shift
                ;;
        esac
    done

    SELECTED_TASK="$(resolve_task_name "${robot}")"
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
    python3 "${SCRIPT_DIR}/scripts/list_envs.py"
}

cmd_random_agent() {
    extract_robot_arg "$@"
    echo -e "${CYAN}[RUN]${NC} Running random action agent..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    python3 "${SCRIPT_DIR}/scripts/random_agent.py" \
        --task "${SELECTED_TASK}" \
        --num_envs "${DEFAULT_NUM_ENVS}" \
        "${FILTERED_ARGS[@]}"
}

cmd_zero_agent() {
    extract_robot_arg "$@"
    echo -e "${CYAN}[RUN]${NC} Running zero action agent..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    python3 "${SCRIPT_DIR}/scripts/zero_agent.py" \
        --task "${SELECTED_TASK}" \
        --num_envs "${DEFAULT_NUM_ENVS}" \
        "${FILTERED_ARGS[@]}"
}

cmd_train() {
    extract_robot_arg "$@"
    echo -e "${CYAN}[TRAIN]${NC} Starting RL training..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    python3 "${SCRIPT_DIR}/scripts/skrl/train.py" \
        --task "${SELECTED_TASK}" \
        "${FILTERED_ARGS[@]}"
}

cmd_play() {
    extract_robot_arg "$@"
    echo -e "${CYAN}[PLAY]${NC} Evaluating trained agent..."
    echo -e "  Task: ${YELLOW}${SELECTED_TASK}${NC}"
    python3 "${SCRIPT_DIR}/scripts/skrl/play.py" \
        --task "${SELECTED_TASK}" \
        "${FILTERED_ARGS[@]}"
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
