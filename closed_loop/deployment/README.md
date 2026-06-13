# closed_loop deployment nodes

ROS nodes that drive the robot from the `closed_loop` policy. They are kept here
(with the rest of our code) so they are versioned in this repo, but **on the
robot they must live inside the SimToolReal ROS package** so they can resolve
`rospy`, the SimToolReal message types, and the rest of the deployment stack.

## Files

| File | Purpose |
|------|---------|
| `goal_pose_node_closed_loop.py` | Closed-loop goal-pose node. Replans tool goals from live tool/object poses with the `closed_loop` policy and publishes `/robot_frame/goal_object_pose` (drop-in for SimToolReal's `goal_pose_node.py`). |
| `goal_pose_node_closed_loop_viz.py` | Debug twin of the above: same predict/replan loop, plus a viser scene rendering the full 15-waypoint trajectory (contact points, spline, normal/surface_dir arrows, tool-mesh ghosts). |

## Where these go in SimToolReal

On the deployment machine, copy both files into the SimToolReal deployment
package:

```
simtoolreal/deployment/goal_pose_node_closed_loop.py
simtoolreal/deployment/goal_pose_node_closed_loop_viz.py
```

They are the only additions SimToolReal needs to run the `closed_loop` policy;
nothing else in `simtoolreal/` is modified. (SimToolReal itself is an external
dependency and is not vendored into this repo — see the top-level `README.md`.)

## Install + run (on the robot's ROS python env)

```bash
pip install -e /path/to/closed_loop          # this package

# from the SimToolReal deployment dir, after copying the node in:
python deployment/goal_pose_node_closed_loop.py \
    --fixed-destination -0.365 -0.056 0.517 \
    --control-frame blue_brush
```

`rospy` and the IsaacGym/SimToolReal message types are provided by the robot's
ROS workspace, not by pip.
