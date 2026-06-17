#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/isaaclab_jetbot/logs/skrl/lidar_nav_direct"
VERBOSE="${VERBOSE:-0}"
STRICT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            [[ $# -lt 2 ]] && { echo "[ERROR] missing value for --root" >&2; exit 1; }
            ROOT_DIR="$2"
            shift 2
            ;;
        --root=*)
            ROOT_DIR="${1#*=}"
            shift
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        *)
            ROOT_DIR="$1"
            shift
            ;;
    esac
done

if [[ ! -d "${ROOT_DIR}" ]]; then
    echo "[ERROR] log root not found: ${ROOT_DIR}" >&2
    exit 1
fi

latest_strict=""
latest_soft=""
latest_any=""

yaml_val() {
    local file="$1"
    local key="$2"
    local line=""
    if command -v rg >/dev/null 2>&1; then
        line="$(rg -n "^${key}:" "${file}" | head -n1 || true)"
    else
        line="$(grep -nE "^${key}:" "${file}" | head -n1 || true)"
    fi
    [[ -z "${line}" ]] && { echo ""; return 0; }
    echo "${line}" | awk -F': ' '{print $2}' | tr -d '\r'
}

is_bad_signature() {
    local env_yaml="$1"
    local action_scale v_max omega_limit obstacle_slow danger_scale sat_scale sat_power
    action_scale="$(yaml_val "${env_yaml}" "action_scale")"
    v_max="$(yaml_val "${env_yaml}" "v_max")"
    omega_limit="$(yaml_val "${env_yaml}" "omega_limit")"
    obstacle_slow="$(yaml_val "${env_yaml}" "obstacle_slowdown_distance")"
    danger_scale="$(yaml_val "${env_yaml}" "danger_speed_penalty_scale")"
    sat_scale="$(yaml_val "${env_yaml}" "action_saturation_penalty_scale")"
    sat_power="$(yaml_val "${env_yaml}" "action_saturation_penalty_power")"

    # Known-bad signatures observed in degraded runs.
    if [[ "${sat_scale}" == "-1.0" ]]; then
        return 0
    fi
    if [[ "${action_scale}" == "8.0" && "${obstacle_slow}" == "0.9" && "${danger_scale}" == "-0.4" \
          && "${sat_scale}" == "-0.6" && "${sat_power}" == "2.0" ]]; then
        return 0
    fi
    if [[ "${v_max}" == "0.1" && "${omega_limit}" == "1.2" && "${obstacle_slow}" == "0.9" \
          && "${danger_scale}" == "-0.4" ]]; then
        return 0
    fi
    return 1
}

is_strict_healthy_signature() {
    local env_yaml="$1"
    local action_scale v_max omega_limit obstacle_slow danger_scale sat_scale sat_power
    action_scale="$(yaml_val "${env_yaml}" "action_scale")"
    v_max="$(yaml_val "${env_yaml}" "v_max")"
    omega_limit="$(yaml_val "${env_yaml}" "omega_limit")"
    obstacle_slow="$(yaml_val "${env_yaml}" "obstacle_slowdown_distance")"
    danger_scale="$(yaml_val "${env_yaml}" "danger_speed_penalty_scale")"
    sat_scale="$(yaml_val "${env_yaml}" "action_saturation_penalty_scale")"
    sat_power="$(yaml_val "${env_yaml}" "action_saturation_penalty_power")"

    # legacy stable profile
    if [[ "${action_scale}" == "6.0" && "${obstacle_slow}" == "0.75" && "${danger_scale}" == "-0.2" \
          && "${sat_scale}" == "-0.3" && "${sat_power}" == "1.5" ]]; then
        return 0
    fi

    # tuned profile (2026-03-02): less aggressive actions and softer saturation shaping
    if [[ "${action_scale}" == "5.0" && "${obstacle_slow}" == "0.85" && "${danger_scale}" == "-0.2" \
          && "${sat_scale}" == "-0.1" && "${sat_power}" == "1.5" ]]; then
        return 0
    fi

    # v-omega stable profile (2026-03-06)
    [[ "${v_max}" == "0.1" && "${omega_limit}" == "1.2" && "${obstacle_slow}" == "0.85" \
       && "${danger_scale}" == "-0.2" && "${sat_scale}" == "-0.1" && "${sat_power}" == "1.5" ]]
}

while IFS= read -r run_dir; do
    ckpt="${run_dir}/checkpoints/best_agent.pt"
    env_yaml="${run_dir}/params/env.yaml"
    [[ -f "${ckpt}" ]] || continue

    [[ -z "${latest_any}" ]] && latest_any="${ckpt}"

    if [[ ! -f "${env_yaml}" ]]; then
        [[ "${VERBOSE}" == "1" ]] && echo "[SKIP] $(basename "${run_dir}") no params/env.yaml" >&2
        continue
    fi

    if is_bad_signature "${env_yaml}"; then
        [[ "${VERBOSE}" == "1" ]] && echo "[SKIP] $(basename "${run_dir}") bad signature" >&2
        continue
    fi

    [[ -z "${latest_soft}" ]] && latest_soft="${ckpt}"

    if is_strict_healthy_signature "${env_yaml}"; then
        latest_strict="${ckpt}"
        break
    fi
done < <(ls -dt "${ROOT_DIR}"/*_ppo_torch 2>/dev/null || true)

if [[ -n "${latest_strict}" ]]; then
    echo "${latest_strict}"
    exit 0
fi

if [[ "${STRICT}" == "1" ]]; then
    echo "[ERROR] strict mode: no healthy-signature checkpoint found under ${ROOT_DIR}" >&2
    exit 2
fi

if [[ -n "${latest_soft}" ]]; then
    echo "${latest_soft}"
    exit 0
fi

if [[ -n "${latest_any}" ]]; then
    echo "[WARN] no healthy checkpoint found, fallback to latest available best_agent.pt" >&2
    echo "${latest_any}"
    exit 0
fi

echo "[ERROR] no checkpoint found under ${ROOT_DIR}" >&2
exit 1
