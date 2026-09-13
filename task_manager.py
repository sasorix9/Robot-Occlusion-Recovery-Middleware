#!/usr/bin/env python3
import argparse
import json
import logging
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


LOGGER = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent / "memory"
OPENVLA_PATH = Path("/root/projects/gello_software/build/run_fr5_openvla")
VISION_MEMORY_PATH = Path(__file__).resolve().with_name("vision_memory.py")
MAX_STEPS = 1000
WAIT_STATUS_SECONDS = 5.0
POLL_SECONDS = 0.1

TASK_CLASSES = ("apple", "orange", "cup", "book")
TARGET_INSTRUCTIONS = {
    "pick up the orange and place it on the bowl": "orange",
    "pick up the apple and place it on the bowl": "apple",
}
OCCLUDER_INSTRUCTIONS = {
    "book": "pick up the book and place it on the plate",
    "cup": "pick up the cup and place it on the plate",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Run a visual-memory task.")
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def read_armed_instruction(input_fn=None):
    if input_fn is None:
        input_fn = input
    while True:
        instruction = input_fn("请输入任务指令：").strip()
        target = TARGET_INSTRUCTIONS.get(instruction)
        if target is not None:
            LOGGER.info("任务已武装：%s", instruction)
            return instruction, target
        LOGGER.warning("无效任务指令，请完整输入允许的指令。")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def validate_task_memory(memory):
    if not isinstance(memory, dict) or set(memory) != set(TASK_CLASSES):
        raise ValueError("task_memory: expected exactly four fixed classes")

    for class_name in TASK_CLASSES:
        item = memory[class_name]
        if not isinstance(item, dict) or set(item) != {
            "class",
            "state",
            "occluder",
        }:
            raise ValueError(
                "task_memory: invalid {} fields".format(class_name)
            )
        if item["class"] != class_name:
            raise ValueError(
                "task_memory: invalid {} class".format(class_name)
            )
        state = item["state"]
        occluder = item["occluder"]
        if not isinstance(state, str) or state not in (
            "visible",
            "occluded",
            "missing",
        ):
            raise ValueError(
                "task_memory: invalid {} state".format(class_name)
            )
        if not isinstance(occluder, str):
            raise ValueError(
                "task_memory: invalid {} occluder".format(class_name)
            )
        if state == "occluded":
            if class_name not in ("apple", "orange"):
                raise ValueError(
                    "task_memory: {} cannot be occluded".format(class_name)
                )
            if occluder not in OCCLUDER_INSTRUCTIONS:
                raise ValueError(
                    "task_memory: invalid {} occluder".format(class_name)
                )
        elif occluder != "NA":
            raise ValueError(
                "task_memory: non-occluded {} must use NA".format(class_name)
            )
    return memory


def load_task_memory(memory_dir):
    memory_dir = Path(memory_dir)
    try:
        commit = _read_json(memory_dir / "commit.json")
    except Exception as exc:
        raise ValueError("commit: {}".format(exc)) from exc
    if (
        not isinstance(commit, dict)
        or set(commit) != {"complete"}
        or type(commit["complete"]) is not bool
        or not commit["complete"]
    ):
        raise ValueError("commit: expected exactly complete=true")

    try:
        memory = _read_json(memory_dir / "task_memory.json")
    except Exception as exc:
        raise ValueError("task_memory: {}".format(exc)) from exc
    return validate_task_memory(memory)


def build_openvla_args(instruction):
    if (
        instruction not in TARGET_INSTRUCTIONS
        and instruction not in OCCLUDER_INSTRUCTIONS.values()
    ):
        raise ValueError("OpenVLA instruction is not allowed")
    return [
        str(OPENVLA_PATH),
        "--instruction",
        instruction,
        "--max-steps",
        str(MAX_STEPS),
        "--enable-motion",
    ]


def launch_openvla(instruction, stage):
    arguments = build_openvla_args(instruction)
    if not OPENVLA_PATH.is_file() or not os.access(str(OPENVLA_PATH), os.X_OK):
        raise RuntimeError("{}: OpenVLA is not executable".format(stage))
    try:
        return subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            shell=False,
        )
    except OSError as exc:
        raise RuntimeError(
            "{}: OpenVLA start failed: {}".format(stage, exc)
        ) from exc


def _raise_if_exited(process, stage, expected_key):
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            "{}: OpenVLA exited with status {} before key {}".format(
                stage, return_code, expected_key
            )
        )


def wait_for_control(process, expected_key, stage, stdin=None):
    if stdin is None:
        stdin = sys.stdin
    LOGGER.info("%s：等待输入 %s 后回车。", stage, expected_key)
    next_status = time.monotonic() + WAIT_STATUS_SECONDS
    while True:
        _raise_if_exited(process, stage, expected_key)
        now = time.monotonic()
        if now >= next_status:
            LOGGER.info("%s：仍在等待输入 %s。", stage, expected_key)
            next_status += WAIT_STATUS_SECONDS
            continue
        readable, _writable, _exceptional = select.select(
            [stdin],
            [],
            [],
            min(POLL_SECONDS, next_status - now),
        )
        _raise_if_exited(process, stage, expected_key)
        if not readable:
            continue
        line = stdin.readline()
        if line == "":
            raise EOFError("{}: stdin reached EOF".format(stage))
        if line.strip() == expected_key:
            return
        LOGGER.warning(
            "%s：输入无效，请输入 %s 后回车。", stage, expected_key
        )


def _wait_for_process_exit(process, stage):
    next_status = time.monotonic() + WAIT_STATUS_SECONDS
    while True:
        return_code = process.poll()
        if return_code is not None:
            return return_code
        now = time.monotonic()
        if now >= next_status:
            LOGGER.info("%s：SIGINT 已发送，仍在等待 OpenVLA 退出。", stage)
            next_status += WAIT_STATUS_SECONDS
            continue
        time.sleep(min(POLL_SECONDS, next_status - now))


def interrupt_and_wait(process, stage):
    if process.poll() is not None:
        raise RuntimeError("{}: OpenVLA exited before SIGINT".format(stage))
    try:
        process.send_signal(signal.SIGINT)
    except Exception as exc:
        raise RuntimeError("{}: SIGINT failed: {}".format(stage, exc)) from exc
    LOGGER.info("%s：已发送 SIGINT，等待 OpenVLA 退出。", stage)
    return _wait_for_process_exit(process, stage)


def _stop_after_abort(process, stage):
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        LOGGER.info("%s：异常中止，已发送 SIGINT。", stage)
        _wait_for_process_exit(process, stage)
    except Exception as exc:
        LOGGER.error("%s：异常中止时无法停止 OpenVLA：%s", stage, exc)


def run_motion(instruction, expected_key, stage, stdin=None):
    process = launch_openvla(instruction, stage)
    try:
        wait_for_control(process, expected_key, stage, stdin)
    except BaseException:
        _stop_after_abort(process, stage)
        raise
    interrupt_and_wait(process, stage)


def run_reobserve(memory_dir):
    if not VISION_MEMORY_PATH.is_file():
        raise RuntimeError("reobserve: vision_memory.py is missing")
    arguments = [
        sys.executable,
        str(VISION_MEMORY_PATH),
        "reobserve",
        "--memory-dir",
        str(Path(memory_dir)),
    ]
    try:
        subprocess.run(arguments, check=True, shell=False)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "reobserve failed with status {}".format(exc.returncode)
        ) from exc
    except OSError as exc:
        raise RuntimeError("reobserve start failed: {}".format(exc)) from exc


def execute_task(task_memory, instruction, target, memory_dir, stdin=None):
    validate_task_memory(task_memory)
    if TARGET_INSTRUCTIONS.get(instruction) != target:
        raise ValueError("target instruction does not match target")

    state = task_memory[target]["state"]
    if state == "missing":
        LOGGER.info("目标 %s 缺失，不启动 OpenVLA。", target)
        return
    if state == "visible":
        run_motion(instruction, "c", "执行目标任务", stdin)
        return

    occluder = task_memory[target]["occluder"]
    run_motion(
        OCCLUDER_INSTRUCTIONS[occluder],
        "q",
        "移除遮挡物 {}".format(occluder),
        stdin,
    )
    run_reobserve(memory_dir)
    try:
        updated = load_task_memory(memory_dir)
    except ValueError as exc:
        raise RuntimeError("reobserve result: {}".format(exc)) from exc
    updated_state = updated[target]["state"]
    if updated_state == "visible":
        run_motion(instruction, "q", "重新执行目标任务", stdin)
        return
    LOGGER.info(
        "重新观察后目标 %s 状态为 %s，不继续运动。",
        target,
        updated_state,
    )


def main(argv=None, input_fn=None, stdin=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        task_memory = load_task_memory(args.memory_dir)
        instruction, target = read_armed_instruction(input_fn)
        execute_task(
            task_memory,
            instruction,
            target,
            args.memory_dir,
            stdin,
        )
    except KeyboardInterrupt:
        LOGGER.error("taskmanager interrupted")
        return 1
    except EOFError as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
