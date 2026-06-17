#!/bin/bash
# Gazebo smoke test for v8 best_agent
# Purpose: Collect failure modes, NOT performance metrics

set -e

POLICY_PATH="/home/cs/isaaclab_jetbot/policy_v8_best.onnx"
BAG_DIR="/home/cs/isaaclab_jetbot/gazebo_smoke_v8_bags"
mkdir -p "$BAG_DIR"

echo "=== Gazebo Smoke Test v8 ==="
echo "Policy: $POLICY_PATH"
echo "Bag output: $BAG_DIR"
echo ""

# Launch Gazebo + turtlebot3_topo_drl in separate terminal
echo "Step 1: Launch Gazebo simulation (do this manually in another terminal):"
echo "  ros2 launch wpr_simulation2 turtlebot3_topo_drl.launch.py checkpoint:=$POLICY_PATH"
echo ""
echo "Step 2: Start rosbag recording (press Enter when Gazebo is ready)"
read -p "Press Enter to start recording..."

# Record critical topics
ros2 bag record \
  -o "$BAG_DIR/smoke_$(date +%Y%m%d_%H%M%S)" \
  /scan /odom /imu \
  /active_waypoint_goal /turn_cmd /speed_cap /planner_state /goal_id \
  /cmd_vel \
  /tf /tf_static &

ROSBAG_PID=$!
echo "Recording started (PID: $ROSBAG_PID)"
echo ""
echo "Step 3: Observe robot behavior and record failure cases:"
echo "  - Near-goal spinning"
echo "  - Spiral forward into wall"
echo "  - Stop after active_waypoint_goal cleared"
echo "  - Fallback/return stuck"
echo "  - Junction wrong turn / over-turn"
echo ""
echo "Press Ctrl+C to stop recording when done (target: 10-20 failure cases)"

# Wait for user interrupt
trap "kill $ROSBAG_PID 2>/dev/null; echo 'Recording stopped'; exit 0" INT
wait $ROSBAG_PID
