#!/usr/bin/env python3
import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = Path(
    "/opt/occlusion/models/yolov8s_seg_coco_640_rk3588_int8.rknn"
)
DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent / "memory"
DEFAULT_CAPTURE_TIMEOUT = 30.0

CLASSES = ("apple", "orange", "cup", "book")
COCO_IDS = {"apple": 47, "orange": 49, "cup": 41, "book": 73}
TARGET_BY_COCO_ID = {class_id: name for name, class_id in COCO_IDS.items()}
FRAME_COUNT = 10
RELIABLE_FRAME_COUNT = 7
COVERAGE_THRESHOLD = 0.8
MIN_VALID_DEPTH_RATIO = 0.20
CONFIDENCE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.45
MASK_THRESHOLD = 0.50
LETTERBOX_FILL = 114
CAMERA_SERIAL = "348122071193"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 30
WARMUP_FRAMES = 30
MODEL_SIZE = 640
REG_MAX = 16
YOLO_OUTPUT_SHAPES = tuple(
    shape
    for grid_size in (80, 40, 20)
    for shape in (
        (1, 64, grid_size, grid_size),
        (1, 80, grid_size, grid_size),
        (1, 1, grid_size, grid_size),
        (1, 32, grid_size, grid_size),
    )
) + ((1, 32, 160, 160),)
MEMORY_KEYS = (
    "classes",
    "masks",
    "depth_cm",
    "has_reference",
    "states",
    "occluders",
)


def prepare_model_input(image_bgr):
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.dtype != np.uint8
    ):
        raise ValueError("image: expected uint8 HWC BGR data")

    original_height, original_width = image_bgr.shape[:2]
    if not original_height or not original_width:
        raise ValueError("image: dimensions must be positive")
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    scale = min(MODEL_SIZE / original_width, MODEL_SIZE / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    if (resized_width, resized_height) != (original_width, original_height):
        rgb = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    left = (MODEL_SIZE - resized_width) // 2
    top = (MODEL_SIZE - resized_height) // 2
    prepared = np.full(
        (MODEL_SIZE, MODEL_SIZE, 3), LETTERBOX_FILL, dtype=np.uint8
    )
    prepared[top : top + resized_height, left : left + resized_width] = rgb
    metadata = {
        "original_height": original_height,
        "original_width": original_width,
        "scale": scale,
        "left": left,
        "top": top,
        "resized_height": resized_height,
        "resized_width": resized_width,
    }
    return np.ascontiguousarray(prepared), metadata


def _validate_yolo_outputs(outputs):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
        YOLO_OUTPUT_SHAPES
    ):
        count = len(outputs) if isinstance(outputs, (list, tuple)) else "invalid"
        raise ValueError(
            "outputs: expected {} tensors, got {}".format(
                len(YOLO_OUTPUT_SHAPES), count
            )
        )
    validated = []
    for index, (output, shape) in enumerate(zip(outputs, YOLO_OUTPUT_SHAPES)):
        if not isinstance(output, np.ndarray):
            raise ValueError("output {}: expected NumPy array".format(index))
        if output.shape != shape:
            raise ValueError(
                "output {}: shape {} != {}".format(index, output.shape, shape)
            )
        if output.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError(
                "output {}: expected float16 or float32".format(index)
            )
        validated.append(output)
    return validated


def _flatten_nchw(tensor):
    return tensor.transpose(0, 2, 3, 1).reshape(-1, tensor.shape[1])


def _dfl(position):
    logits = position.astype(np.float32).reshape(
        1, 4, REG_MAX, position.shape[2], position.shape[3]
    )
    logits -= np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX, 1, 1)
    return np.sum(probabilities * bins, axis=2)


def _decode_boxes(position):
    grid_height, grid_width = position.shape[2:]
    columns, rows = np.meshgrid(
        np.arange(grid_width, dtype=np.float32),
        np.arange(grid_height, dtype=np.float32),
    )
    grid = np.stack((columns, rows), axis=0)[None]
    stride = np.array(
        [MODEL_SIZE / grid_width, MODEL_SIZE / grid_height], dtype=np.float32
    ).reshape(1, 2, 1, 1)
    distances = _dfl(position)
    top_left = (grid + 0.5 - distances[:, :2]) * stride
    bottom_right = (grid + 0.5 + distances[:, 2:]) * stride
    return np.concatenate((top_left, bottom_right), axis=1)


def _box_iou(first, second):
    intersection_width = max(
        0.0, min(first[2], second[2]) - max(first[0], second[0])
    )
    intersection_height = max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _nms_indices(boxes, scores, class_ids):
    kept = []
    for index in np.argsort(-scores, kind="mergesort"):
        if any(
            class_ids[index] == class_ids[other]
            and _box_iou(boxes[index], boxes[other]) > NMS_THRESHOLD
            for other in kept
        ):
            continue
        kept.append(int(index))
    return kept


def _restore_box(box, metadata):
    restored = np.array(box, dtype=np.float32, copy=True)
    restored[[0, 2]] = (restored[[0, 2]] - metadata["left"]) / metadata[
        "scale"
    ]
    restored[[1, 3]] = (restored[[1, 3]] - metadata["top"]) / metadata[
        "scale"
    ]
    restored[[0, 2]] = np.clip(
        restored[[0, 2]], 0, metadata["original_width"]
    )
    restored[[1, 3]] = np.clip(
        restored[[1, 3]], 0, metadata["original_height"]
    )
    return restored


def _restore_mask(probability, model_box, restored_box, metadata):
    probability = cv2.resize(
        probability,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    left = max(0, int(np.ceil(model_box[0])))
    top = max(0, int(np.ceil(model_box[1])))
    right = min(MODEL_SIZE, int(np.ceil(model_box[2])))
    bottom = min(MODEL_SIZE, int(np.ceil(model_box[3])))
    cropped = np.zeros_like(probability, dtype=np.float32)
    if right > left and bottom > top:
        cropped[top:bottom, left:right] = probability[top:bottom, left:right]

    content_top = metadata["top"]
    content_left = metadata["left"]
    content = cropped[
        content_top : content_top + metadata["resized_height"],
        content_left : content_left + metadata["resized_width"],
    ]
    original_size = (metadata["original_width"], metadata["original_height"])
    if content.shape[::-1] != original_size:
        content = cv2.resize(content, original_size, interpolation=cv2.INTER_LINEAR)
    mask = (content >= MASK_THRESHOLD).astype(np.uint8)

    left = max(0, int(np.floor(restored_box[0])))
    top = max(0, int(np.floor(restored_box[1])))
    right = min(metadata["original_width"], int(np.ceil(restored_box[2])))
    bottom = min(metadata["original_height"], int(np.ceil(restored_box[3])))
    bounded = np.zeros_like(mask)
    if right > left and bottom > top:
        bounded[top:bottom, left:right] = mask[top:bottom, left:right]
    return bounded


def postprocess_yolov8_seg(outputs, metadata):
    outputs = _validate_yolo_outputs(outputs)
    empty = (
        np.empty((0, 4), dtype=np.float32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(
            (0, metadata["original_height"], metadata["original_width"]),
            dtype=np.uint8,
        ),
    )
    boxes, class_probabilities, coefficients = [], [], []
    for branch in range(3):
        offset = branch * 4
        boxes.append(_flatten_nchw(_decode_boxes(outputs[offset])))
        class_probabilities.append(_flatten_nchw(outputs[offset + 1]))
        coefficients.append(_flatten_nchw(outputs[offset + 3]))

    boxes = np.concatenate(boxes).astype(np.float32)
    class_probabilities = np.concatenate(class_probabilities).astype(np.float32)
    coefficients = np.concatenate(coefficients).astype(np.float32)
    class_ids = np.argmax(class_probabilities, axis=1).astype(np.int64)
    scores = np.max(class_probabilities, axis=1).astype(np.float32)
    selected = (scores >= CONFIDENCE_THRESHOLD) & np.isin(
        class_ids, tuple(TARGET_BY_COCO_ID)
    )
    if not np.any(selected):
        return empty

    boxes = boxes[selected]
    class_ids = class_ids[selected]
    scores = scores[selected]
    coefficients = coefficients[selected]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, MODEL_SIZE)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, MODEL_SIZE)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    class_ids = class_ids[valid]
    scores = scores[valid]
    coefficients = coefficients[valid]
    if not len(boxes):
        return empty

    kept = _nms_indices(boxes, scores, class_ids)
    one_per_class = []
    seen = set()
    for index in kept:
        class_id = int(class_ids[index])
        if class_id not in seen:
            one_per_class.append(index)
            seen.add(class_id)
    boxes = boxes[one_per_class]
    class_ids = class_ids[one_per_class]
    scores = scores[one_per_class]
    coefficients = coefficients[one_per_class]

    prototype = outputs[12][0].astype(np.float32).reshape(32, -1)
    logits = np.matmul(coefficients, prototype)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    probabilities = probabilities.reshape(-1, 160, 160)
    restored_boxes = np.asarray(
        [_restore_box(box, metadata) for box in boxes], dtype=np.float32
    )
    masks = np.stack(
        [
            _restore_mask(probability, model_box, restored_box, metadata)
            for probability, model_box, restored_box in zip(
                probabilities, boxes, restored_boxes
            )
        ]
    )
    return restored_boxes, class_ids, scores, masks


def _create_rknn_runtime():
    from rknnlite.api import RKNNLite

    return RKNNLite()


class RknnSegmenter:
    def __init__(self, model_path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError("model not found: {}".format(model_path))
        self._runtime = _create_rknn_runtime()
        try:
            if self._runtime.load_rknn(str(model_path)) != 0:
                raise RuntimeError("load_rknn failed: {}".format(model_path))
            if self._runtime.init_runtime() != 0:
                raise RuntimeError("init_runtime failed")
        except Exception:
            self.close()
            raise

    def infer(self, image_bgr):
        if self._runtime is None:
            raise RuntimeError("RKNN runtime is closed")
        prepared, metadata = prepare_model_input(image_bgr)
        try:
            outputs = self._runtime.inference(
                inputs=[prepared[None]], data_format="nhwc"
            )
        except Exception as exc:
            raise RuntimeError("RKNN inference failed: {}".format(exc)) from exc
        if outputs is None:
            raise RuntimeError("RKNN inference failed: runtime returned no outputs")
        return postprocess_yolov8_seg(outputs, metadata)

    def close(self):
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.release()


def _remaining_wait_ms(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("capture timeout")
    return max(1, min(1000, int(remaining * 1000)))


def _capture_timeout(captured, last_error=None):
    message = "capture timeout: got {}/{} valid frames".format(
        captured, FRAME_COUNT
    )
    if last_error is not None:
        message += "; last frame error: {}".format(last_error)
    return TimeoutError(message)


def capture_raw_frames(segmenter, capture_timeout):
    try:
        serials = {
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        }
    except Exception as exc:
        raise RuntimeError("camera enumeration failed: {}".format(exc)) from exc
    if CAMERA_SERIAL not in serials:
        raise RuntimeError(
            "camera start failed: serial {} not found".format(CAMERA_SERIAL)
        )
    pipeline = rs.pipeline()
    configuration = rs.config()
    configuration.enable_device(CAMERA_SERIAL)
    configuration.enable_stream(
        rs.stream.color,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        rs.format.bgr8,
        CAMERA_FPS,
    )
    configuration.enable_stream(
        rs.stream.depth,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        rs.format.z16,
        CAMERA_FPS,
    )

    started = False
    try:
        try:
            profile = pipeline.start(configuration)
        except Exception as exc:
            raise RuntimeError("camera start failed: {}".format(exc)) from exc
        started = True
        deadline = time.monotonic() + float(capture_timeout)
        depth_scale_m = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
            raise RuntimeError("camera depth scale is invalid")
        align = rs.align(rs.stream.color)

        last_error = None
        warmed = 0
        while warmed < WARMUP_FRAMES:
            try:
                frames = pipeline.wait_for_frames(_remaining_wait_ms(deadline))
                if not frames:
                    raise RuntimeError("warmup frameset is missing")
                warmed += 1
            except TimeoutError:
                raise _capture_timeout(0, last_error)
            except Exception as exc:
                last_error = exc

        raw_frames = []
        while len(raw_frames) < FRAME_COUNT:
            try:
                frames = pipeline.wait_for_frames(_remaining_wait_ms(deadline))
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError("aligned color/depth frame is missing")
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy()
                if color.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                    raise RuntimeError("aligned color frame has wrong shape")
                if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                    raise RuntimeError("aligned depth frame has wrong shape")
            except TimeoutError:
                raise _capture_timeout(len(raw_frames), last_error)
            except Exception as exc:
                last_error = exc
                continue

            _boxes, class_ids, scores, masks = segmenter.infer(color)
            if time.monotonic() >= deadline:
                raise _capture_timeout(len(raw_frames), last_error)
            raw_frames.append((class_ids, scores, masks, depth, depth_scale_m))
        return raw_frames
    finally:
        if started:
            pipeline.stop()


def observe_memory(segmenter, history, capture_timeout):
    return update_memory(
        history,
        capture_raw_frames(segmenter, capture_timeout),
    )


def normalize_frame_observations(
    class_ids, scores, masks, depth_image, depth_scale_m
):
    if not (len(class_ids) == len(scores) == len(masks)):
        raise ValueError("detections: class, score, and mask counts differ")

    depth_image = np.asarray(depth_image)
    if depth_image.ndim != 2:
        raise ValueError("depth_image: expected a two-dimensional array")
    depth_scale_m = float(depth_scale_m)
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError("depth_scale_m: expected a positive finite value")

    selected = {}
    for class_id, score, mask in zip(class_ids, scores, masks):
        class_name = TARGET_BY_COCO_ID.get(class_id)
        score = float(score)
        if (
            class_name is not None
            and np.isfinite(score)
            and score >= CONFIDENCE_THRESHOLD
            and (class_name not in selected or score > selected[class_name][0])
        ):
            selected[class_name] = (score, mask)

    observations = {}
    for class_name in CLASSES:
        if class_name not in selected:
            continue
        score, mask = selected[class_name]
        mask = np.asarray(mask)
        if mask.shape != depth_image.shape:
            raise ValueError("mask: {} shape differs from depth".format(class_name))
        if not np.logical_or(mask == 0, mask == 1).all():
            raise ValueError("mask: {} is not binary".format(class_name))
        mask = mask.astype(np.bool_, copy=False)
        mask_pixels = int(np.count_nonzero(mask))
        if not mask_pixels:
            continue

        masked_depth = depth_image[mask]
        valid_depth = masked_depth[np.isfinite(masked_depth)]
        valid_depth = valid_depth[valid_depth > 0]
        if valid_depth.size / mask_pixels < MIN_VALID_DEPTH_RATIO:
            continue
        observations[class_name] = {
            "mask": mask.astype(np.uint8),
            "depth_cm": float(np.median(valid_depth) * depth_scale_m * 100.0),
            "confidence": score,
        }
    return observations


def _matching_masks(first, second):
    first = np.asarray(first, dtype=np.bool_)
    second = np.asarray(second, dtype=np.bool_)
    if first.shape != second.shape:
        raise ValueError("shape: masks differ")
    return first, second


def mask_iou(first, second):
    first, second = _matching_masks(first, second)
    union = int(np.count_nonzero(np.logical_or(first, second)))
    if not union:
        raise ValueError("union: masks are empty")
    intersection = int(np.count_nonzero(np.logical_and(first, second)))
    return intersection / union


def mask_coverage(target, candidate):
    target, candidate = _matching_masks(target, candidate)
    target_pixels = int(np.count_nonzero(target))
    if not target_pixels:
        raise ValueError("target: mask is empty")
    intersection = int(np.count_nonzero(np.logical_and(target, candidate)))
    return intersection / target_pixels


def aggregate_observations(frames):
    if len(frames) != FRAME_COUNT:
        raise ValueError("frames: expected exactly {}".format(FRAME_COUNT))

    grouped = {class_name: [] for class_name in CLASSES}
    for frame in frames:
        if not isinstance(frame, dict) or not set(frame).issubset(CLASSES):
            raise ValueError("frames: invalid class mapping")
        for class_name, observation in frame.items():
            if not isinstance(observation, dict) or set(observation) != {
                "mask",
                "depth_cm",
                "confidence",
            }:
                raise ValueError("frames: invalid {} observation".format(class_name))
            mask = np.asarray(observation["mask"])
            if mask.ndim != 2 or not mask.any():
                raise ValueError("frames: {} mask is empty".format(class_name))
            if not np.logical_or(mask == 0, mask == 1).all():
                raise ValueError("frames: {} mask is not binary".format(class_name))
            depth = float(observation["depth_cm"])
            if not np.isfinite(depth) or depth <= 0:
                raise ValueError("frames: {} depth is invalid".format(class_name))
            grouped[class_name].append((mask.astype(np.uint8), depth))

    reliable = {}
    for class_name in CLASSES:
        observations = grouped[class_name]
        if len(observations) < RELIABLE_FRAME_COUNT:
            continue
        masks = [observation[0] for observation in observations]
        medians = [
            float(
                np.median(
                    [
                        mask_iou(mask, other)
                        for other_index, other in enumerate(masks)
                        if other_index != index
                    ]
                )
            )
            for index, mask in enumerate(masks)
        ]
        representative_index = max(range(len(masks)), key=medians.__getitem__)
        reliable[class_name] = {
            "mask": masks[representative_index].copy(),
            "depth_cm": float(
                np.median([observation[1] for observation in observations])
            ),
        }
    return reliable


def _validate_reliable_observations(reliable):
    if not isinstance(reliable, dict) or not set(reliable).issubset(CLASSES):
        raise ValueError("reliable: invalid class mapping")
    normalized = {}
    for class_name in CLASSES:
        if class_name not in reliable:
            continue
        observation = reliable[class_name]
        if not isinstance(observation, dict) or set(observation) != {
            "mask",
            "depth_cm",
        }:
            raise ValueError("reliable: invalid {} observation".format(class_name))
        mask = np.asarray(observation["mask"])
        if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError("reliable: invalid {} mask shape".format(class_name))
        if not mask.any() or not np.logical_or(mask == 0, mask == 1).all():
            raise ValueError("reliable: invalid {} mask".format(class_name))
        depth = float(observation["depth_cm"])
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError("reliable: invalid {} depth".format(class_name))
        normalized[class_name] = {
            "mask": mask.astype(np.uint8),
            "depth_cm": depth,
        }
    return normalized


def update_visible_missing(history, reliable):
    validate_memory(history)
    reliable = _validate_reliable_observations(reliable)
    current = {key: value.copy() for key, value in history.items()}

    for index, class_name in enumerate(CLASSES):
        current["occluders"][index] = "NA"
        if class_name in reliable:
            current["masks"][index] = reliable[class_name]["mask"]
            current["depth_cm"][index] = reliable[class_name]["depth_cm"]
            current["has_reference"][index] = True
            current["states"][index] = "visible"
        else:
            current["states"][index] = "missing"
    return validate_memory(current)


def apply_occlusion_states(memory):
    validate_memory(memory)
    current = {key: value.copy() for key, value in memory.items()}

    for target_name in ("apple", "orange"):
        target_index = CLASSES.index(target_name)
        if (
            current["states"][target_index] != "missing"
            or not current["has_reference"][target_index]
        ):
            continue

        best_name = None
        best_coverage = COVERAGE_THRESHOLD
        for candidate_name in ("cup", "book"):
            candidate_index = CLASSES.index(candidate_name)
            if current["states"][candidate_index] != "visible":
                continue
            coverage = mask_coverage(
                current["masks"][target_index],
                current["masks"][candidate_index],
            )
            if (
                coverage > best_coverage
                and current["depth_cm"][candidate_index]
                < current["depth_cm"][target_index]
            ):
                best_name = candidate_name
                best_coverage = coverage

        if best_name is not None:
            current["states"][target_index] = "occluded"
            current["occluders"][target_index] = best_name
    return validate_memory(current)


def update_memory(history, raw_frames):
    validate_memory(history)
    if len(raw_frames) != FRAME_COUNT:
        raise ValueError("frames: expected exactly {}".format(FRAME_COUNT))
    observations = []
    for frame in raw_frames:
        if not isinstance(frame, (tuple, list)) or len(frame) != 5:
            raise ValueError("frames: expected five values per frame")
        observations.append(normalize_frame_observations(*frame))
    reliable = aggregate_observations(observations)
    return apply_occlusion_states(update_visible_missing(history, reliable))


def create_blank_memory():
    return {
        "classes": np.array(CLASSES, dtype="<U6"),
        "masks": np.zeros((len(CLASSES), IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8),
        "depth_cm": np.full(len(CLASSES), np.nan, dtype=np.float32),
        "has_reference": np.zeros(len(CLASSES), dtype=np.bool_),
        "states": np.full(len(CLASSES), "missing", dtype="<U8"),
        "occluders": np.full(len(CLASSES), "NA", dtype="<U4"),
    }


def validate_memory(memory):
    if not isinstance(memory, dict) or set(memory) != set(MEMORY_KEYS):
        raise ValueError("keys: memory must contain exactly {}".format(MEMORY_KEYS))

    expected = {
        "classes": ((len(CLASSES),), "unicode"),
        "masks": ((len(CLASSES), IMAGE_HEIGHT, IMAGE_WIDTH), np.dtype(np.uint8)),
        "depth_cm": ((len(CLASSES),), np.dtype(np.float32)),
        "has_reference": ((len(CLASSES),), np.dtype(np.bool_)),
        "states": ((len(CLASSES),), "unicode"),
        "occluders": ((len(CLASSES),), "unicode"),
    }
    for key, (shape, dtype) in expected.items():
        value = memory[key]
        if not isinstance(value, np.ndarray):
            raise ValueError("{}: expected numpy array".format(key))
        if value.shape != shape:
            raise ValueError(
                "{}: expected shape {}, got {}".format(key, shape, value.shape)
            )
        if dtype == "unicode":
            if value.dtype.kind != "U":
                raise ValueError("{}: expected Unicode dtype".format(key))
        elif value.dtype != dtype:
            raise ValueError(
                "{}: expected dtype {}, got {}".format(key, dtype, value.dtype)
            )

    if memory["classes"].tolist() != list(CLASSES):
        raise ValueError("classes: expected fixed class order {}".format(CLASSES))
    if not np.logical_or(memory["masks"] == 0, memory["masks"] == 1).all():
        raise ValueError("masks: values must be binary")
    if not set(memory["states"].tolist()).issubset(
        {"visible", "occluded", "missing"}
    ):
        raise ValueError("states: invalid state")
    if not set(memory["occluders"].tolist()).issubset({"NA", "cup", "book"}):
        raise ValueError("occluders: invalid occluder")

    for index, class_name in enumerate(CLASSES):
        has_reference = bool(memory["has_reference"][index])
        mask_has_pixels = bool(memory["masks"][index].any())
        depth = float(memory["depth_cm"][index])
        state = str(memory["states"][index])
        occluder = str(memory["occluders"][index])

        if has_reference:
            if not mask_has_pixels:
                raise ValueError("masks: {} reference mask is empty".format(class_name))
            if not np.isfinite(depth) or depth <= 0:
                raise ValueError(
                    "depth_cm: {} reference depth is invalid".format(class_name)
                )
        else:
            if mask_has_pixels:
                raise ValueError(
                    "masks: {} has mask without reference".format(class_name)
                )
            if not np.isnan(depth):
                raise ValueError(
                    "depth_cm: {} has depth without reference".format(class_name)
                )

        if state == "visible" and not has_reference:
            raise ValueError(
                "has_reference: visible {} needs a reference".format(class_name)
            )
        if state == "occluded":
            if class_name not in ("apple", "orange") or not has_reference:
                raise ValueError(
                    "states: invalid occluded state for {}".format(class_name)
                )
            if occluder not in ("cup", "book"):
                raise ValueError(
                    "occluders: occluded {} needs cup or book".format(class_name)
                )
        elif occluder != "NA":
            raise ValueError(
                "occluders: non-occluded {} must use NA".format(class_name)
            )

    return memory


def save_memory_npz(path, memory):
    path = Path(path)
    validate_memory(memory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez_compressed(output, **memory)


def load_memory_npz(path):
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            memory = {key: archive[key] for key in archive.files}
        return validate_memory(memory)
    except Exception as exc:
        raise ValueError("npz: {}".format(exc)) from exc


def project_task_memory(memory):
    validate_memory(memory)
    return {
        class_name: {
            "class": class_name,
            "state": str(memory["states"][index]),
            "occluder": str(memory["occluders"][index]),
        }
        for index, class_name in enumerate(CLASSES)
    }


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def _stage_file(directory, target_name, writer, validator):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(target_name), suffix=".tmp", dir=str(directory)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        validator(temporary)
        return temporary
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _stage_npz(directory, target_name, memory):
    return _stage_file(
        directory,
        target_name,
        lambda path: save_memory_npz(path, memory),
        load_memory_npz,
    )


def _stage_json(directory, target_name, value):
    def validate(path):
        if _read_json(path) != value:
            raise ValueError("json: staged value changed")

    return _stage_file(
        directory,
        target_name,
        lambda path: _write_json(path, value),
        validate,
    )


def _write_commit_marker(directory, complete):
    target = Path(directory) / "commit.json"
    temporary = _stage_json(directory, target.name, {"complete": bool(complete)})
    try:
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()


def read_commit_marker(directory):
    try:
        value = _read_json(Path(directory) / "commit.json")
    except Exception as exc:
        raise ValueError("commit: {}".format(exc)) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"complete"}
        or type(value["complete"]) is not bool
    ):
        raise ValueError("commit: expected exactly one boolean complete field")
    return value["complete"]


def commit_memory_files(
    directory, initial_memory=None, current_memory=None, finalize=True
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if initial_memory is None and current_memory is None:
        raise ValueError("commit: no data files supplied")
    if finalize and current_memory is None:
        raise ValueError("commit: final submission needs current memory")

    staged = []
    try:
        if initial_memory is not None:
            staged.append(
                (
                    _stage_npz(directory, "memory_initial.npz", initial_memory),
                    directory / "memory_initial.npz",
                )
            )
        if current_memory is not None:
            staged.append(
                (
                    _stage_npz(directory, "memory_current.npz", current_memory),
                    directory / "memory_current.npz",
                )
            )
            task_memory = project_task_memory(current_memory)
            staged.append(
                (
                    _stage_json(directory, "task_memory.json", task_memory),
                    directory / "task_memory.json",
                )
            )

        _write_commit_marker(directory, False)
        for temporary, target in staged:
            os.replace(str(temporary), str(target))
        if finalize:
            _write_commit_marker(directory, True)
    finally:
        for temporary, _target in staged:
            if temporary.exists():
                temporary.unlink()


def _report_initial_memory(memory):
    visible = [
        class_name
        for index, class_name in enumerate(CLASSES)
        if memory["states"][index] == "visible"
    ]
    for class_name in visible:
        LOGGER.info(
            "检测到物体 %s，已采集 mask、深度值。",
            class_name,
        )
    if not visible:
        LOGGER.info("未检测到可靠物体。")
    LOGGER.info("可靠物体总数：%d", len(visible))


def build_initial(segmenter, memory_dir, capture_timeout):
    initial = observe_memory(
        segmenter,
        create_blank_memory(),
        capture_timeout,
    )
    commit_memory_files(
        memory_dir,
        initial_memory=initial,
        finalize=False,
    )
    _report_initial_memory(initial)
    return initial


def _wait_for_token(expected, prompt, input_fn):
    while True:
        if input_fn(prompt).strip() == expected:
            return
        LOGGER.warning("输入无效，请输入 %s 后回车。", expected)


def wait_for_manual_move(input_fn=input):
    _wait_for_token("s", "请输入 s 后回车，开始移动物体：", input_fn)
    _wait_for_token("e", "移动完成后请输入 e 并回车：", input_fn)


def build_current(segmenter, initial, memory_dir, capture_timeout):
    current = observe_memory(segmenter, initial, capture_timeout)
    commit_memory_files(
        memory_dir,
        current_memory=current,
        finalize=True,
    )
    LOGGER.info("记忆槽更新完毕，目录：%s/", Path(memory_dir).resolve())
    return current


def run_build(segmenter, memory_dir, capture_timeout, input_fn=input):
    initial = build_initial(segmenter, memory_dir, capture_timeout)
    wait_for_manual_move(input_fn)
    return build_current(segmenter, initial, memory_dir, capture_timeout)


def load_reobserve_history(memory_dir):
    if not read_commit_marker(memory_dir):
        raise ValueError("commit: complete must be true")
    return load_memory_npz(Path(memory_dir) / "memory_current.npz")


def run_reobserve(segmenter, history, memory_dir, capture_timeout):
    current = observe_memory(segmenter, history, capture_timeout)
    commit_memory_files(
        memory_dir,
        current_memory=current,
        finalize=True,
    )
    LOGGER.info("重新观察完成，目录：%s/", Path(memory_dir).resolve())
    return current


def _positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="Build or refresh visual memory.")
    parser.add_argument("mode", choices=("build", "reobserve"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument(
        "--capture-timeout",
        type=_positive_float,
        default=DEFAULT_CAPTURE_TIMEOUT,
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    segmenter = None
    status = 0
    try:
        history = None
        if args.mode == "reobserve":
            history = load_reobserve_history(args.memory_dir)
        segmenter = RknnSegmenter(args.model)
        if args.mode == "build":
            run_build(segmenter, args.memory_dir, args.capture_timeout)
        else:
            run_reobserve(
                segmenter,
                history,
                args.memory_dir,
                args.capture_timeout,
            )
    except KeyboardInterrupt:
        LOGGER.error("操作被中断。")
        status = 1
    except Exception as exc:
        LOGGER.error("%s", exc)
        status = 1
    finally:
        if segmenter is not None:
            try:
                segmenter.close()
            except Exception as exc:
                LOGGER.error("RKNN release failed: %s", exc)
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
