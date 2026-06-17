#!/bin/bash
# Verify Nav2 recovery chain when fallback occurs
# Run this after system restart

source /opt/ros/humble/setup.bash
source /home/cs/drl_planner/install/setup.bash

echo "=== /cmd_vel publishers (expect: only arbiter) ==="
ros2 topic info /cmd_vel -v 2>/dev/null | grep 'Node name' | sort | uniq -c

echo ""
echo "=== arbiter node count (expect: 1) ==="
ros2 node list 2>/dev/null | grep arbiter | wc -l

echo ""
echo "=== Monitoring for fallback (30s)..."
echo "Watch: planner_state | exec_source_hz | navigate_to_pose action"
echo ""

# Monitor planner_state and action goals simultaneously
ros2 topic echo /planner_state &
PLANNER_PID=$!
ros2 topic echo /exec_source_hz &
HZ_PID=$!

# Wait for fallback
sleep 30

kill $PLANNER_PID $HZ_PID 2>/dev/null

echo ""
echo "=== cmd_vel_nav2 message count in last 30s ==="
ros2 topic hz /cmd_vel_nav2 --window 10 &
HZ2_PID=$!
sleep 5
kill $HZ2_PID 2>/dev/null
