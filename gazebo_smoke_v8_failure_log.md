# Gazebo Smoke Test v8 - Failure Mode Collection

## Test Info
- Policy: policy_v8_best.onnx (v8 iter ~1350, collision 41-59%, success 17-29%, goal_d 1.25-1.34m)
- Date: 2026-03-15
- Environment: Gazebo turtlebot3_topo_drl
- Goal: Collect 10-20 failure cases, classify failure modes

## Failure Case Template

### Case #N: [Brief Description]
- **Timestamp**: [rosbag time or wall clock]
- **Failure Type**: [near-goal spinning / spiral into wall / goal cleared stop / fallback stuck / junction wrong turn]
- **Planner State**: [explore / fallback / junction / return]
- **Goal Distance**: [m]
- **Min LiDAR**: [m]
- **Speed Cap**: [m/s]
- **Observations**:
  - [What the robot did]
  - [What it should have done]
  - [Suspected root cause]

---

## Collected Cases

### Case #1:
- **Timestamp**:
- **Failure Type**:
- **Planner State**:
- **Goal Distance**:
- **Min LiDAR**:
- **Speed Cap**:
- **Observations**:
  -
  -
  -

### Case #2:
- **Timestamp**:
- **Failure Type**:
- **Planner State**:
- **Goal Distance**:
- **Min LiDAR**:
- **Speed Cap**:
- **Observations**:
  -
  -
  -

[Add more cases as needed]

---

## Summary Statistics (after collection)

| Failure Type | Count | % |
|--------------|-------|---|
| Near-goal spinning | | |
| Spiral into wall | | |
| Goal cleared stop | | |
| Fallback stuck | | |
| Junction wrong turn | | |
| Other | | |

## Key Observations

1. Most common failure mode:
2. Planner state distribution in failures:
3. Goal distance range in failures:
4. Suspected root causes:

## Next Steps

Based on failure analysis:
- [ ] Adjust training goal distribution
- [ ] Modify planner_state sampling
- [ ] Add specific failure scenarios to training
- [ ] Architectural changes (e.g., exclude fallback/return from DRL)
