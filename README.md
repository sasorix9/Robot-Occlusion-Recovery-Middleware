# Robot Occlusion Recovery Middleware

A small, synchronous middleware for recovering manipulation tasks when an
object is hidden. It uses an Intel RealSense D435i and a YOLOv8s-seg RKNN
model on RK3588 to track one `apple`, `orange`, `cup`, and `book`.

The system records whether each object is `visible`, `occluded`, or `missing`.
Only `cup` and `book` may occlude `apple` or `orange`.

## Files

- `vision_memory.py`: RGB-D capture, RKNN segmentation, memory aggregation,
  atomic persistence, and the `build`/`reobserve` commands.
- `task_manager.py`: validates memory and runs the fixed OpenVLA task flow.
- `book_pose_viewer.py`: read-only live diagnostic for book segmentation.
- `test_memory_logic.py`: offline `unittest` coverage; it never moves a robot.

## Requirements

The tested target is Ubuntu 20.04 aarch64 with Python 3.8, NumPy, OpenCV,
`pyrealsense2`, and RKNN Toolkit Lite2 1.6.0. The RKNN model and OpenVLA
runner are external artifacts and are not included.

Before deployment, check these installation-specific constants:

- `DEFAULT_MODEL_PATH` and `CAMERA_SERIAL` in `vision_memory.py`
- `OPENVLA_PATH` in `task_manager.py`

The OpenVLA executable must accept `--instruction`, `--max-steps`, and
`--enable-motion`, must handle `SIGINT`, and must not read stdin.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m compileall -q \
  vision_memory.py task_manager.py book_pose_viewer.py
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest -v test_memory_logic.py
```

## Build and Refresh Memory

Start `build` with an empty workspace. After the first observation, enter `s`,
place the objects, then enter `e`:

```bash
/usr/bin/python3 vision_memory.py build --memory-dir ./memory
```

Refresh the latest memory after any scene change:

```bash
/usr/bin/python3 vision_memory.py reobserve --memory-dir ./memory
```

`memory_initial.npz` preserves the initial experiment record.
`memory_current.npz` and `task_memory.json` are overwritten by every
successful re-observation; `commit.json` is written complete last.

## Run a Task

```bash
/usr/bin/python3 task_manager.py --memory-dir ./memory
```

The accepted instructions are exactly:

```text
pick up the orange and place it on the bowl
pick up the apple and place it on the bowl
```

Entering a complete instruction arms real motion because the task manager
passes `--enable-motion`. Verify the emergency stop and clear the robot's
workspace first. Run `reobserve` before a later task whenever the scene has
changed.

For a live, read-only book pose check:

```bash
DISPLAY=:0 /usr/bin/python3 book_pose_viewer.py
```

This project intentionally does not support multiple instances per class,
partial or multiple occluders, a moving camera, or concurrent tasks.
