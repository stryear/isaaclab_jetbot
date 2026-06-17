#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${ROOT_DIR}/scripts/skrl/train.py"

ISAACLAB_SH="${ISAACLAB_SH:-$HOME/IsaacLab/isaaclab.sh}"
TASK="Isaac-Lab-Tutorial-LidarJunction-TurtleBot3-Direct-v0"
ALGORITHM="PPO"
NUM_ENVS=100
HEADLESS=1
DEBUG_PRINT_ENABLE=1
DEBUG_PRINT_INTERVAL_STEPS=200
CKPT_ROOT="${ROOT_DIR}/logs/skrl/lidar_junction_direct"
CURRICULUM_LOG_DIR="${ROOT_DIR}/logs/skrl/lidar_junction_curriculum"
STRICT_GATING=1
MAX_RETRIES_PER_PHASE=2
GATE_AVG_WINDOW=5
START_PHASE=1

print_usage() {
    cat <<'EOF'
Usage:
  scripts/skrl/train_lidar_junction_curriculum.sh [options]

Options:
  --num_envs N                Number of parallel envs (default: 100)
  --headless                  Enable headless mode (default)
  --no-headless               Disable headless mode
  --debug_print_enable 0|1    Enable debug prints (default: 1)
  --debug_interval N          Debug print interval steps (default: 200)
  --strict_gating 0|1         Enable metric-based phase gating (default: 1)
  --max_retries N             Max retries per phase when gating fails (default: 2)
  --start_phase N             Start curriculum from phase N in [1, 4] (default: 1)
  --isaaclab_sh PATH          Path to isaaclab.sh (default: ~/IsaacLab/isaaclab.sh)
  --task TASK_ID              Override task id
  --help                      Show this help

Notes:
  - By default, Phase1 starts from scratch and Phase2/3/4 auto-resume from latest checkpoint.
  - If --start_phase > 1, training starts at that phase and resumes from latest checkpoint.
  - Phases: 1(wide) -> 2(narrower) -> 3(3a: narrow, 1 obs) -> 4(3b: narrowest, 2 obs, X-junction)
  - Gating is based on latest [METRIC][LidarNav] and [DEBUG][LidarNav] lines from each phase log.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=1
            shift
            ;;
        --no-headless)
            HEADLESS=0
            shift
            ;;
        --debug_print_enable)
            DEBUG_PRINT_ENABLE="$2"
            shift 2
            ;;
        --debug_interval)
            DEBUG_PRINT_INTERVAL_STEPS="$2"
            shift 2
            ;;
        --strict_gating)
            STRICT_GATING="$2"
            shift 2
            ;;
        --max_retries)
            MAX_RETRIES_PER_PHASE="$2"
            shift 2
            ;;
        --start_phase)
            START_PHASE="$2"
            shift 2
            ;;
        --isaaclab_sh)
            ISAACLAB_SH="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] isaaclab.sh not executable: ${ISAACLAB_SH}" >&2
    exit 1
fi

if ! [[ "${START_PHASE}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --start_phase must be an integer in [1, 4]: ${START_PHASE}" >&2
    exit 1
fi
if [[ "${START_PHASE}" -lt 1 || "${START_PHASE}" -gt 4 ]]; then
    echo "[ERROR] --start_phase out of range [1, 4]: ${START_PHASE}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] train script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi
mkdir -p "${CURRICULUM_LOG_DIR}"

phase_params() {
    local phase="$1"
    case "${phase}" in
        1)
            PHASE_MAX_ITER=8000
            PHASE_JUNCTION_HALF_WIDTH=1.35
            PHASE_JUNCTION_HALF_LENGTH=6.0
            PHASE_JUNCTION_GOAL_FORWARD_MAX=1.9
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MIN=4.8
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MAX=5.4
            PHASE_JUNCTION_ROBOT_SPAWN_LATERAL_JITTER=0.03
            PHASE_JUNCTION_X_SAMPLING_PROB=0.0
            PHASE_NUM_OBSTACLES=1
            # METRIC thresholds
            T_SUCCESS_MIN=60.0
            T_COLLISION_MAX=18.0
            T_TIMEOUT_MAX=35.0
            T_GOAL_END_DIST_MAX=0.70
            T_MIN_LIDAR_MIN=0.65
            # DEBUG thresholds
            T_DEBUG_COLL_STEP_MAX=2.5
            T_DEBUG_SAT_STEP_MAX=35.0
            T_DEBUG_STALL_STEP_MAX=8.0
            ;;
        2)
            PHASE_MAX_ITER=6000
            PHASE_JUNCTION_HALF_WIDTH=1.30
            PHASE_JUNCTION_HALF_LENGTH=6.0
            PHASE_JUNCTION_GOAL_FORWARD_MAX=2.2
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MIN=4.8
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MAX=5.4
            PHASE_JUNCTION_ROBOT_SPAWN_LATERAL_JITTER=0.03
            PHASE_JUNCTION_X_SAMPLING_PROB=0.0
            PHASE_NUM_OBSTACLES=1
            T_SUCCESS_MIN=60.0
            T_COLLISION_MAX=10.0
            T_TIMEOUT_MAX=35.0
            T_GOAL_END_DIST_MAX=0.85
            T_MIN_LIDAR_MIN=0.75
            T_DEBUG_COLL_STEP_MAX=2.5
            T_DEBUG_SAT_STEP_MAX=30.0
            T_DEBUG_STALL_STEP_MAX=8.0
            ;;
        3)
            # Phase 3a: narrow corridor, 1 obstacle, no X-junction
            PHASE_MAX_ITER=8000
            PHASE_JUNCTION_HALF_WIDTH=1.25
            PHASE_JUNCTION_HALF_LENGTH=6.0
            PHASE_JUNCTION_GOAL_FORWARD_MAX=2.3
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MIN=4.8
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MAX=5.4
            PHASE_JUNCTION_ROBOT_SPAWN_LATERAL_JITTER=0.03
            PHASE_JUNCTION_X_SAMPLING_PROB=0.0
            PHASE_NUM_OBSTACLES=1
            T_SUCCESS_MIN=55.0
            T_COLLISION_MAX=12.0
            T_TIMEOUT_MAX=38.0
            T_GOAL_END_DIST_MAX=0.80
            T_MIN_LIDAR_MIN=0.70
            T_DEBUG_COLL_STEP_MAX=2.5
            T_DEBUG_SAT_STEP_MAX=32.0
            T_DEBUG_STALL_STEP_MAX=8.0
            ;;
        4)
            # Phase 3b: narrowest corridor, 2 obstacles, X-junction
            PHASE_MAX_ITER=12000
            PHASE_JUNCTION_HALF_WIDTH=1.20
            PHASE_JUNCTION_HALF_LENGTH=6.0
            PHASE_JUNCTION_GOAL_FORWARD_MAX=2.4
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MIN=4.6
            PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MAX=5.4
            PHASE_JUNCTION_ROBOT_SPAWN_LATERAL_JITTER=0.03
            PHASE_JUNCTION_X_SAMPLING_PROB=0.05
            PHASE_NUM_OBSTACLES=2
            T_SUCCESS_MIN=55.0
            T_COLLISION_MAX=12.0
            T_TIMEOUT_MAX=40.0
            T_GOAL_END_DIST_MAX=0.75
            T_MIN_LIDAR_MIN=0.75
            T_DEBUG_COLL_STEP_MAX=2.5
            T_DEBUG_SAT_STEP_MAX=32.0
            T_DEBUG_STALL_STEP_MAX=8.0
            ;;
        *)
            echo "[ERROR] Unsupported phase: ${phase}" >&2
            exit 1
            ;;
    esac
}

latest_ckpt() {
    local run_dir ckpt
    while IFS= read -r run_dir; do
        ckpt="${run_dir}/checkpoints/best_agent.pt"
        if [[ -f "${ckpt}" ]]; then
            echo "${ckpt}"
            return 0
        fi
    done < <(ls -1dt "${CKPT_ROOT}"/*_ppo_torch 2>/dev/null || true)

    return 1
}

phase_passed() {
    local log_file="$1"
    local n="${GATE_AVG_WINDOW}"
    local metric_lines debug_lines
    metric_lines="$(grep -E "\\[METRIC\\]\\[LidarNav\\]" "${log_file}" | tail -n "${n}")"
    debug_lines="$(grep -E "\\[DEBUG\\]\\[LidarNav\\]" "${log_file}" | tail -n "${n}")"

    if [[ -z "${metric_lines}" || -z "${debug_lines}" ]]; then
        echo "[GATE] missing METRIC/DEBUG lines in ${log_file}" >&2
        return 1
    fi

    # Helper: extract numeric value after "key=" (works with mawk)
    # Usage inside awk: extract($0, "success=") strips trailing %
    #
    # Average METRIC fields over last N lines and check thresholds in one awk pass
    local metric_result
    metric_result="$(echo "${metric_lines}" | awk \
        -v smin="${T_SUCCESS_MIN}" -v cmax="${T_COLLISION_MAX}" -v tmax="${T_TIMEOUT_MAX}" \
        -v gmax="${T_GOAL_END_DIST_MAX}" -v lmin="${T_MIN_LIDAR_MIN}" \
        'function extract(line, key,    p, rest, val) {
            p = index(line, key)
            if (p == 0) return 0
            rest = substr(line, p + length(key))
            val = rest + 0
            return val
        }
        {
            s += extract($0, "success=")
            c += extract($0, "collision=")
            t += extract($0, "timeout=")
            g += extract($0, "avg_goal_end_dist=")
            l += extract($0, "avg_min_lidar=")
            n++
        }
        END {
            if (n == 0) { print "PARSE_ERROR"; exit 1 }
            s /= n; c /= n; t /= n; g /= n; l /= n
            printf "[GATE] METRIC avg (last %d): success=%.1f%% collision=%.1f%% timeout=%.1f%% goal_dist=%.2f min_lidar=%.2f\n", n, s, c, t, g, l > "/dev/stderr"
            pass = (s >= smin) && (c <= cmax) && (t <= tmax) && (g <= gmax) && (l >= lmin)
            print (pass ? "PASS" : "FAIL")
        }'
    )"

    if [[ "${metric_result}" == "PARSE_ERROR" ]]; then
        echo "[GATE] failed to parse METRIC lines" >&2
        return 1
    fi

    # Average DEBUG fields over last N lines and check thresholds
    local debug_result
    debug_result="$(echo "${debug_lines}" | awk \
        -v dcmax="${T_DEBUG_COLL_STEP_MAX}" -v dsmax="${T_DEBUG_SAT_STEP_MAX}" -v dslmax="${T_DEBUG_STALL_STEP_MAX}" \
        'function extract(line, key,    p, rest, val) {
            p = index(line, key)
            if (p == 0) return 0
            rest = substr(line, p + length(key))
            val = rest + 0
            return val
        }
        {
            dc += extract($0, "coll_step=")
            ds += extract($0, "sat_step=")
            dsl += extract($0, "stall_step=")
            n++
        }
        END {
            if (n == 0) { print "PARSE_ERROR"; exit 1 }
            dc /= n; ds /= n; dsl /= n
            printf "[GATE] DEBUG avg (last %d): coll_step=%.1f%% sat_step=%.1f%% stall_step=%.1f%%\n", n, dc, ds, dsl > "/dev/stderr"
            pass = (dc <= dcmax) && (ds <= dsmax) && (dsl <= dslmax)
            print (pass ? "PASS" : "FAIL")
        }'
    )"

    if [[ "${debug_result}" == "PARSE_ERROR" ]]; then
        echo "[GATE] failed to parse DEBUG lines" >&2
        return 1
    fi

    [[ "${metric_result}" == "PASS" && "${debug_result}" == "PASS" ]]
}

run_phase() {
    local phase="$1"
    local attempt="$2"
    local ckpt_path="$3"
    phase_params "${phase}"

    local ts log_file
    ts="$(date +%Y-%m-%d_%H-%M-%S)"
    log_file="${CURRICULUM_LOG_DIR}/${ts}_phase${phase}_attempt${attempt}.log"

    echo "[RUN] phase=${phase} attempt=${attempt} log=${log_file}"
    echo "[RUN] params: iter=${PHASE_MAX_ITER} width=${PHASE_JUNCTION_HALF_WIDTH} len=${PHASE_JUNCTION_HALF_LENGTH} goal_fwd_max=${PHASE_JUNCTION_GOAL_FORWARD_MAX} spawn_off=[${PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MIN},${PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MAX}] jitter=${PHASE_JUNCTION_ROBOT_SPAWN_LATERAL_JITTER} x_prob=${PHASE_JUNCTION_X_SAMPLING_PROB} obstacles=${PHASE_NUM_OBSTACLES}"
    if [[ -n "${ckpt_path}" ]]; then
        echo "[RUN] resume checkpoint: ${ckpt_path}"
    else
        echo "[RUN] start from scratch"
    fi

    local cmd=(
        env PYTHONNOUSERSITE=1 PYTHONPATH=
        "${ISAACLAB_SH}" -p "${TRAIN_SCRIPT}"
        --task "${TASK}"
        --algorithm "${ALGORITHM}"
        --num_envs "${NUM_ENVS}"
        --max_iterations "${PHASE_MAX_ITER}"
        --num_obstacles "${PHASE_NUM_OBSTACLES}"
        --junction_half_width "${PHASE_JUNCTION_HALF_WIDTH}"
        --junction_half_length "${PHASE_JUNCTION_HALF_LENGTH}"
        --junction_goal_forward_max "${PHASE_JUNCTION_GOAL_FORWARD_MAX}"
        --junction_robot_spawn_offset_min "${PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MIN}"
        --junction_robot_spawn_offset_max "${PHASE_JUNCTION_ROBOT_SPAWN_OFFSET_MAX}"
        --junction_robot_spawn_lateral_jitter "${PHASE_JUNCTION_ROBOT_SPAWN_LATERAL_JITTER}"
        --junction_x_sampling_prob "${PHASE_JUNCTION_X_SAMPLING_PROB}"
        --debug_phase_tag "${phase}"
        --debug_print_enable "${DEBUG_PRINT_ENABLE}"
        --debug_print_interval_steps "${DEBUG_PRINT_INTERVAL_STEPS}"
    )
    if [[ "${HEADLESS}" == "1" ]]; then
        cmd+=(--headless)
    fi
    if [[ -n "${ckpt_path}" ]]; then
        cmd+=(--checkpoint "${ckpt_path}")
    fi

    (cd "${ROOT_DIR}" && "${cmd[@]}") 2>&1 | tee "${log_file}"
    RUN_LOG_FILE="${log_file}"
}

run_curriculum() {
    local phase attempt ckpt log_file
    for phase in $(seq "${START_PHASE}" 4); do
        attempt=1
        while true; do
            ckpt=""
            # Keep Phase1 retries from scratch to avoid repeatedly bootstrapping from a bad policy.
            # Phase2/3 continue to resume from the latest available checkpoint.
            if [[ "${phase}" -gt 1 ]]; then
                if ! ckpt="$(latest_ckpt)"; then
                    echo "[WARN] no checkpoint found under ${CKPT_ROOT}; phase ${phase} will start from scratch"
                    ckpt=""
                fi
            fi

            run_phase "${phase}" "${attempt}" "${ckpt}"
            log_file="${RUN_LOG_FILE}"

            if [[ "${STRICT_GATING}" != "1" ]]; then
                echo "[GATE] strict gating disabled; move to next phase"
                break
            fi

            if phase_passed "${log_file}"; then
                echo "[GATE] phase ${phase} PASSED"
                break
            fi

            echo "[GATE] phase ${phase} NOT passed"
            if [[ "${attempt}" -ge "${MAX_RETRIES_PER_PHASE}" ]]; then
                echo "[ERROR] phase ${phase} exceeded max retries (${MAX_RETRIES_PER_PHASE})" >&2
                exit 2
            fi
            attempt=$((attempt + 1))
            echo "[GATE] retry phase ${phase} (attempt ${attempt})"
        done
    done
    echo "[DONE] curriculum training completed"
}

run_curriculum
