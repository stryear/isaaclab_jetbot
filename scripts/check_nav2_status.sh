#!/bin/bash
# Quick Nav2 status check - run while system is running

echo "=== cmd_vel topics ==="
ros2 topic list | grep cmd_vel

echo ""
echo "=== navigate_to_pose actions ==="
ros2 action list 2>/dev/null | grep navigate_to_pose || echo "(none found)"

echo ""
echo "=== Nav2 node lifecycle ==="
for node in /nav2/controller_server /nav2/bt_navigator /nav2/planner_server /controller_server /bt_navigator; do
  result=$(ros2 lifecycle get $node 2>/dev/null)
  if [ -n "$result" ]; then
    echo "  $node: $result"
  fi
done

echo ""
echo "=== cmd_vel topic publishers ==="
for topic in /cmd_vel /cmd_vel_nav /cmd_vel_drl /nav2/cmd_vel; do
  info=$(ros2 topic info $topic 2>/dev/null | grep -E 'Publisher|Type')
  if [ -n "$info" ]; then
    echo "  $topic:"
    echo "$info" | sed 's/^/    /'
  fi
done

echo ""
echo "=== Current planner_state ==="
ros2 topic echo /planner_state --once --timeout 2 2>/dev/null || echo "(no message)"

echo ""
echo "=== exec_source_hz latest ==="
ros2 topic echo /exec_source_hz --once --timeout 2 2>/dev/null || echo "(no message)"
