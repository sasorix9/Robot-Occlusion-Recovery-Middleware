# Robot Occlusion Recovery Middleware

A small, synchronous middleware for recovering manipulation tasks when an
object is hidden. It uses an Intel RealSense D435i and a YOLOv8s-seg RKNN
model on RK3588 to track one `apple`, `orange`, `cup`, and `book`.

The system records whether each object is `visible`, `occluded`, or `missing`.
Only `cup` and `book` may occlude `apple` or `orange`.

```text
RealSense RGB-D -> visual memory -> task state
                                     |
                  OpenVLA/FR5 <------+
                         |
                         +-> reobserve after occluder removal
```

## Files

- `vision_memory.py`: RGB-D capture, RKNN segmentation, memory aggregation,
  atomic persistence, and the `build`/`reobserve` commands.
- `task_manager.py`: validates memory and runs the fixed OpenVLA task flow.
- `book_pose_viewer.py`: read-only live diagnostic for book segmentation.
- `test_memory_logic.py`: offline `unittest` coverage; it never moves a robot.
- `integration/run_fr5_openvla.cpp`: RealSense-to-OpenVLA-to-FR5 execution
  adapter used in the experiments.
- `CMakeLists.txt`: standalone build for the execution adapter.

## Requirements

The tested target is Ubuntu 20.04 aarch64 with Python 3.8, NumPy 1.17.4,
OpenCV 4.2.0, librealsense 2.54.2, RKNN Toolkit Lite2/runtime 1.6.0, and
RKNPU driver 0.9.8. The RKNN model, FAIRINO SDK, and fine-tuned OpenVLA
weights are external artifacts and are not included. The experiment used the
[RK3588 YOLOv8s-seg INT8 model](https://github.com/Qengineering/YoloV8-seg-NPU/tree/main/rk3588).

Before deployment, check these installation-specific constants:

- `DEFAULT_MODEL_PATH` and `CAMERA_SERIAL` in `vision_memory.py`
- server, robot, gripper, camera, and Tool/User constants at the top of
  `integration/run_fr5_openvla.cpp`

The OpenVLA executable must accept `--instruction`, `--max-steps`, and
`--enable-motion`, must handle `SIGINT`, and must not read stdin.

Build the included FR5 adapter against a locally installed FAIRINO SDK:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFAIRINO_SDK_DIR=/path/to/fairino/libfairino
cmake --build build --target run_fr5_openvla -j2
./build/run_fr5_openvla --self-test
export OPENVLA_PATH="$PWD/build/run_fr5_openvla"
```

The adapter also requires librealsense2, OpenCV, libcurl, nlohmann/json, and
POSIX threads. Dataset collection and RLDS conversion live in the companion
[Data_collector](https://github.com/sasorix9/Data_collector) repository.

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

## License and Attribution

Project code is released under the MIT License. Third-party models, SDKs, and
reference implementations retain their own licenses; see
`THIRD_PARTY_NOTICES.md`.
