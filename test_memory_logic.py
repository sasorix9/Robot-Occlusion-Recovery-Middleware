import contextlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import task_manager as tm
import vision_memory as vm


class CliTests(unittest.TestCase):
    def test_modes_defaults_and_fixed_constants(self):
        for mode in ("build", "reobserve"):
            args = vm.parse_args([mode])
            self.assertEqual(args.mode, mode)
            self.assertEqual(args.model, vm.DEFAULT_MODEL_PATH)
            self.assertEqual(args.memory_dir, vm.DEFAULT_MEMORY_DIR)
            self.assertEqual(args.capture_timeout, 30.0)

        self.assertEqual(vm.CLASSES, ("apple", "orange", "cup", "book"))
        self.assertEqual(
            vm.COCO_IDS, {"apple": 47, "orange": 49, "cup": 41, "book": 73}
        )
        self.assertEqual((vm.FRAME_COUNT, vm.RELIABLE_FRAME_COUNT), (10, 7))
        self.assertEqual(
            vm.DEFAULT_MODEL_PATH,
            Path(
                "/opt/occlusion/models/"
                "yolov8s_seg_coco_640_rk3588_int8.rknn"
            ),
        )
        self.assertEqual(vm.COVERAGE_THRESHOLD, 0.8)
        self.assertEqual(vm.MIN_VALID_DEPTH_RATIO, 0.20)
        self.assertEqual(
            (vm.CONFIDENCE_THRESHOLD, vm.NMS_THRESHOLD, vm.MASK_THRESHOLD),
            (0.20, 0.45, 0.50),
        )
        self.assertEqual(vm.LETTERBOX_FILL, 114)
        self.assertEqual(vm.CAMERA_SERIAL, "348122071193")
        self.assertEqual(
            (vm.IMAGE_WIDTH, vm.IMAGE_HEIGHT, vm.CAMERA_FPS), (640, 480, 30)
        )
        self.assertEqual(vm.WARMUP_FRAMES, 30)

    def test_cli_overrides_and_rejects_invalid_values(self):
        args = vm.parse_args(
            [
                "build",
                "--model",
                "/tmp/model.rknn",
                "--memory-dir",
                "/tmp/memory",
                "--capture-timeout",
                "2.5",
            ]
        )
        self.assertEqual(args.model, Path("/tmp/model.rknn"))
        self.assertEqual(args.memory_dir, Path("/tmp/memory"))
        self.assertEqual(args.capture_timeout, 2.5)

        for argv in (("unknown",), ("build", "--capture-timeout", "0")):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    vm.parse_args(list(argv))


class RknnSegmentationTests(unittest.TestCase):
    @staticmethod
    def outputs():
        return [
            np.zeros(shape, dtype=np.float32)
            for shape in vm.YOLO_OUTPUT_SHAPES
        ]

    @staticmethod
    def set_box(box_tensor, row, column, bin_index=2):
        box_tensor.fill(-20.0)
        for edge in range(4):
            box_tensor[0, edge * vm.REG_MAX + bin_index, row, column] = 20.0

    def test_preprocess_converts_bgr_and_uses_114_letterbox(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[0, 0] = (1, 2, 3)

        prepared, metadata = vm.prepare_model_input(image)

        self.assertEqual(prepared.shape, (640, 640, 3))
        self.assertEqual(prepared.dtype, np.uint8)
        self.assertTrue(prepared.flags.c_contiguous)
        np.testing.assert_array_equal(prepared[0, 0], (114, 114, 114))
        np.testing.assert_array_equal(prepared[80, 0], (3, 2, 1))
        self.assertEqual(
            metadata,
            {
                "original_height": 480,
                "original_width": 640,
                "scale": 1.0,
                "left": 0,
                "top": 80,
                "resized_height": 480,
                "resized_width": 640,
            },
        )

    def test_fixed_outputs_decode_cup_box_and_binary_camera_mask(self):
        outputs = self.outputs()
        self.set_box(outputs[0], row=20, column=30)
        outputs[1][0, vm.COCO_IDS["cup"], 20, 30] = 0.80
        outputs[3][0, 0, 20, 30] = 1.0
        outputs[12][0, 0] = 10.0
        _, metadata = vm.prepare_model_input(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )

        boxes, class_ids, scores, masks = vm.postprocess_yolov8_seg(
            outputs, metadata
        )

        self.assertEqual(class_ids.tolist(), [vm.COCO_IDS["cup"]])
        self.assertAlmostEqual(float(scores[0]), 0.80, places=6)
        self.assertEqual(boxes.shape, (1, 4))
        self.assertEqual(masks.shape, (1, 480, 640))
        self.assertEqual(masks.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(masks[0])), 0)
        self.assertTrue(np.logical_or(masks == 0, masks == 1).all())
        left, top, right, bottom = boxes[0].astype(int)
        self.assertFalse(masks[0, :top].any())
        self.assertFalse(masks[0, bottom:].any())
        self.assertFalse(masks[0, :, :left].any())
        self.assertFalse(masks[0, :, right:].any())

    def test_rejects_wrong_output_contract(self):
        outputs = self.outputs()
        outputs[3] = np.zeros((1, 31, 80, 80), dtype=np.float32)
        _, metadata = vm.prepare_model_input(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        with self.assertRaisesRegex(ValueError, "output 3"):
            vm.postprocess_yolov8_seg(outputs, metadata)

    def test_runtime_failures_release_and_missing_model_stops_first(self):
        class Runtime:
            def __init__(self, load_result=0, init_result=0):
                self.load_result = load_result
                self.init_result = init_result
                self.released = 0

            def load_rknn(self, _path):
                return self.load_result

            def init_runtime(self):
                return self.init_result

            def release(self):
                self.released += 1

        with tempfile.TemporaryDirectory() as directory_text:
            model_path = Path(directory_text) / "model.rknn"
            model_path.write_bytes(b"model")

            factory = mock.Mock()
            with mock.patch.object(vm, "_create_rknn_runtime", factory):
                with self.assertRaisesRegex(FileNotFoundError, "model"):
                    vm.RknnSegmenter(model_path.with_name("missing.rknn"))
            factory.assert_not_called()

            for load_result, init_result, message in (
                (1, 0, "load_rknn"),
                (0, 1, "init_runtime"),
            ):
                with self.subTest(message=message):
                    runtime = Runtime(load_result, init_result)
                    with mock.patch.object(
                        vm, "_create_rknn_runtime", return_value=runtime
                    ):
                        with self.assertRaisesRegex(RuntimeError, message):
                            vm.RknnSegmenter(model_path)
                    self.assertEqual(runtime.released, 1)

    def test_runtime_receives_batched_nhwc_input_and_closes(self):
        outputs = self.outputs()

        class Runtime:
            def __init__(self):
                self.call = None
                self.released = 0

            def load_rknn(self, _path):
                return 0

            def init_runtime(self):
                return 0

            def inference(self, inputs, data_format):
                self.call = (inputs, data_format)
                return outputs

            def release(self):
                self.released += 1

        with tempfile.TemporaryDirectory() as directory_text:
            model_path = Path(directory_text) / "model.rknn"
            model_path.write_bytes(b"model")
            runtime = Runtime()
            with mock.patch.object(
                vm, "_create_rknn_runtime", return_value=runtime
            ):
                segmenter = vm.RknnSegmenter(model_path)
            try:
                segmenter.infer(
                    np.zeros((vm.IMAGE_HEIGHT, vm.IMAGE_WIDTH, 3), dtype=np.uint8)
                )
            finally:
                segmenter.close()

        self.assertEqual(runtime.call[0][0].shape, (1, 640, 640, 3))
        self.assertEqual(runtime.call[1], "nhwc")
        self.assertEqual(runtime.released, 1)


class MemoryStructureTests(unittest.TestCase):
    def test_blank_memory_schema_and_values(self):
        memory = vm.create_blank_memory()

        self.assertEqual(set(memory), set(vm.MEMORY_KEYS))
        np.testing.assert_array_equal(memory["classes"], vm.CLASSES)
        self.assertEqual(memory["masks"].shape, (4, 480, 640))
        self.assertEqual(memory["masks"].dtype, np.uint8)
        self.assertFalse(memory["masks"].any())
        self.assertEqual(memory["depth_cm"].dtype, np.float32)
        self.assertTrue(np.isnan(memory["depth_cm"]).all())
        self.assertEqual(memory["has_reference"].dtype, np.bool_)
        self.assertFalse(memory["has_reference"].any())
        self.assertEqual(memory["states"].tolist(), ["missing"] * 4)
        self.assertEqual(memory["occluders"].tolist(), ["NA"] * 4)

    def test_memory_validation_accepts_valid_memory(self):
        memory = vm.create_blank_memory()
        memory["masks"][0, 10:20, 20:30] = 1
        memory["depth_cm"][0] = 42.0
        memory["has_reference"][0] = True
        memory["states"][0] = "visible"
        self.assertIs(vm.validate_memory(memory), memory)

    def test_memory_validation_rejects_schema_and_consistency_errors(self):
        cases = []

        memory = vm.create_blank_memory()
        del memory["masks"]
        cases.append(("keys", memory))

        memory = vm.create_blank_memory()
        memory["masks"] = memory["masks"][:, :-1]
        cases.append(("masks", memory))

        memory = vm.create_blank_memory()
        memory["depth_cm"] = memory["depth_cm"].astype(np.float64)
        cases.append(("depth_cm", memory))

        memory = vm.create_blank_memory()
        memory["classes"][[0, 1]] = memory["classes"][[1, 0]]
        cases.append(("classes", memory))

        memory = vm.create_blank_memory()
        memory["states"][0] = "invalid"
        cases.append(("states", memory))

        memory = vm.create_blank_memory()
        memory["occluders"][0] = "apple"
        cases.append(("occluders", memory))

        memory = vm.create_blank_memory()
        memory["depth_cm"][0] = 42.0
        cases.append(("depth_cm", memory))

        memory = vm.create_blank_memory()
        memory["has_reference"][0] = True
        memory["depth_cm"][0] = 42.0
        cases.append(("masks", memory))

        for field, invalid in cases:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                vm.validate_memory(invalid)


class NpzTests(unittest.TestCase):
    @staticmethod
    def mixed_memory():
        memory = vm.create_blank_memory()
        for index, state in enumerate(("visible", "occluded", "missing", "visible")):
            memory["masks"][index, index : index + 2, index : index + 3] = 1
            memory["depth_cm"][index] = 30.0 + index
            memory["has_reference"][index] = True
            memory["states"][index] = state
        memory["occluders"][1] = "book"
        return memory

    def test_npz_round_trip_without_object_arrays(self):
        expected = self.mixed_memory()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            vm.save_memory_npz(path, expected)
            actual = vm.load_memory_npz(path)

            for key in vm.MEMORY_KEYS:
                np.testing.assert_array_equal(actual[key], expected[key])
            with np.load(str(path), allow_pickle=False) as archive:
                self.assertTrue(
                    all(archive[key].dtype.kind != "O" for key in archive.files)
                )

    def test_npz_rejects_corrupt_and_object_data(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.npz"
            corrupt.write_bytes(b"not an npz")
            with self.assertRaisesRegex(ValueError, "npz"):
                vm.load_memory_npz(corrupt)

            object_path = Path(directory) / "object.npz"
            np.savez(str(object_path), classes=np.array(vm.CLASSES, dtype=object))
            with self.assertRaisesRegex(ValueError, "npz"):
                vm.load_memory_npz(object_path)


class TaskProjectionTests(unittest.TestCase):
    def test_projection_has_only_fixed_task_fields(self):
        memory = NpzTests.mixed_memory()
        projection = vm.project_task_memory(memory)

        self.assertEqual(list(projection), list(vm.CLASSES))
        for class_name in vm.CLASSES:
            self.assertEqual(
                set(projection[class_name]), {"class", "state", "occluder"}
            )
            self.assertEqual(projection[class_name]["class"], class_name)
        self.assertEqual(projection["orange"]["state"], "occluded")
        self.assertEqual(projection["orange"]["occluder"], "book")
        self.assertEqual(projection["apple"]["occluder"], "NA")
        self.assertEqual(json.loads(json.dumps(projection)), projection)


class CommitTests(unittest.TestCase):
    @staticmethod
    def seed_complete(directory):
        memory = NpzTests.mixed_memory()
        vm.commit_memory_files(directory, initial_memory=memory, finalize=False)
        vm.commit_memory_files(directory, current_memory=memory, finalize=True)
        return memory

    def test_successful_two_phase_commit(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            memory = NpzTests.mixed_memory()

            vm.commit_memory_files(directory, initial_memory=memory, finalize=False)
            self.assertFalse(vm.read_commit_marker(directory))
            self.assertTrue((directory / "memory_initial.npz").is_file())
            self.assertFalse((directory / "memory_current.npz").exists())

            vm.commit_memory_files(directory, current_memory=memory, finalize=True)
            self.assertTrue(vm.read_commit_marker(directory))
            self.assertEqual(
                json.loads((directory / "commit.json").read_text()),
                {"complete": True},
            )
            vm.load_memory_npz(directory / "memory_initial.npz")
            vm.load_memory_npz(directory / "memory_current.npz")
            self.assertEqual(
                json.loads((directory / "task_memory.json").read_text()),
                vm.project_task_memory(memory),
            )
            self.assertFalse(
                any(path.name.startswith(".") for path in directory.iterdir())
            )

    def test_staging_failure_preserves_previous_complete_commit(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            memory = self.seed_complete(directory)
            previous_initial = (directory / "memory_initial.npz").read_bytes()

            with mock.patch.object(
                vm.np, "savez_compressed", side_effect=OSError("full")
            ):
                with self.assertRaises(OSError):
                    vm.commit_memory_files(
                        directory, initial_memory=memory, finalize=False
                    )

            self.assertTrue(vm.read_commit_marker(directory))
            self.assertEqual(
                (directory / "memory_initial.npz").read_bytes(), previous_initial
            )
            self.assertFalse(
                any(path.name.startswith(".") for path in directory.iterdir())
            )

    def test_replacement_failures_leave_expected_commit_state(self):
        scenarios = (
            ("false marker", 1, True),
            ("current npz", 2, False),
            ("task json", 3, False),
            ("true marker", 4, False),
        )
        for name, fail_call, expected_complete in scenarios:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory_text:
                    directory = Path(directory_text)
                    memory = self.seed_complete(directory)
                    real_replace = vm.os.replace
                    call_count = 0

                    def failing_replace(source, destination):
                        nonlocal call_count
                        call_count += 1
                        if call_count == fail_call:
                            raise OSError(name)
                        return real_replace(source, destination)

                    with mock.patch.object(
                        vm.os, "replace", side_effect=failing_replace
                    ):
                        with self.assertRaises(OSError):
                            vm.commit_memory_files(
                                directory, current_memory=memory, finalize=True
                            )

                    self.assertEqual(
                        vm.read_commit_marker(directory), expected_complete
                    )
                    self.assertFalse(
                        any(path.name.startswith(".") for path in directory.iterdir())
                    )

    def test_initial_replacement_failure_leaves_false_marker(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            memory = self.seed_complete(directory)
            real_replace = vm.os.replace
            call_count = 0

            def fail_initial(source, destination):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("initial")
                return real_replace(source, destination)

            with mock.patch.object(vm.os, "replace", side_effect=fail_initial):
                with self.assertRaises(OSError):
                    vm.commit_memory_files(
                        directory, initial_memory=memory, finalize=False
                    )
            self.assertFalse(vm.read_commit_marker(directory))
            self.assertFalse(
                any(path.name.startswith(".") for path in directory.iterdir())
            )

    def test_commit_marker_rejects_extra_fields(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            (directory / "commit.json").write_text(
                json.dumps({"complete": True, "generation": 1})
            )
            with self.assertRaisesRegex(ValueError, "commit"):
                vm.read_commit_marker(directory)


class FrameObservationTests(unittest.TestCase):
    def test_filters_targets_and_keeps_highest_confidence_instance(self):
        depth = np.full((2, 5), 1000.0, dtype=np.float32)
        low_mask = np.zeros(depth.shape, dtype=np.uint8)
        low_mask[0, 0] = 1
        high_mask = np.zeros(depth.shape, dtype=np.uint8)
        high_mask[1, 3:5] = 1

        observations = vm.normalize_frame_observations(
            [0, vm.COCO_IDS["cup"], vm.COCO_IDS["cup"]],
            [0.99, 0.60, 0.90],
            [np.ones(depth.shape), low_mask, high_mask],
            depth,
            0.001,
        )

        self.assertEqual(list(observations), ["cup"])
        self.assertEqual(observations["cup"]["confidence"], 0.90)
        np.testing.assert_array_equal(observations["cup"]["mask"], high_mask)
        self.assertEqual(observations["cup"]["depth_cm"], 100.0)

    def test_rejects_empty_and_insufficient_depth_masks(self):
        empty = np.zeros((2, 5), dtype=np.uint8)
        full = np.ones((2, 5), dtype=np.uint8)
        finite_below = np.full((2, 5), np.nan, dtype=np.float32)
        finite_below[0, 0] = 1000.0
        finite_below[0, 1] = 0.0

        cases = (
            ("empty", empty, np.full((2, 5), 1000.0)),
            ("zero", full, np.zeros((2, 5))),
            ("nan", full, np.full((2, 5), np.nan)),
            ("below", np.ones((2, 3)), finite_below[:, :3]),
        )
        for name, mask, depth in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    vm.normalize_frame_observations(
                        [vm.COCO_IDS["apple"]], [0.90], [mask], depth, 0.001
                    ),
                    {},
                )

    def test_accepts_exactly_and_above_twenty_percent_without_rounding(self):
        mask = np.ones((2, 5), dtype=np.uint8)
        for valid_count in (2, 3):
            with self.subTest(valid_count=valid_count):
                depth = np.zeros(mask.shape, dtype=np.float32)
                depth.flat[:valid_count] = np.arange(1, valid_count + 1) * 100.0
                observations = vm.normalize_frame_observations(
                    [vm.COCO_IDS["orange"]], [0.75], [mask], depth, 0.001
                )
                self.assertIn("orange", observations)

        exact_depth = np.zeros(mask.shape, dtype=np.float32)
        exact_depth.flat[:2] = (100.0, 300.0)
        exact = vm.normalize_frame_observations(
            [vm.COCO_IDS["orange"]], [0.75], [mask], exact_depth, 0.001
        )["orange"]
        self.assertEqual(exact["depth_cm"], 20.0)


class MaskSimilarityTests(unittest.TestCase):
    def test_iou_for_identical_disjoint_and_partial_masks(self):
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        identical = target.copy()
        disjoint = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        partial = np.array([[0, 1], [1, 1]], dtype=np.uint8)

        self.assertEqual(vm.mask_iou(target, identical), 1.0)
        self.assertEqual(vm.mask_iou(target, disjoint), 0.0)
        self.assertAlmostEqual(vm.mask_iou(target, partial), 1.0 / 4.0)

    def test_coverage_uses_historical_target_as_denominator(self):
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        candidate = np.array([[0, 1], [1, 1]], dtype=np.uint8)

        self.assertEqual(vm.mask_coverage(target, target), 1.0)
        self.assertEqual(vm.mask_coverage(target, np.zeros((2, 2))), 0.0)
        self.assertEqual(vm.mask_coverage(target, candidate), 0.5)
        self.assertAlmostEqual(vm.mask_coverage(candidate, target), 1.0 / 3.0)

    def test_rejects_shape_mismatch_and_empty_denominators(self):
        empty = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "shape"):
            vm.mask_iou(empty, np.zeros((1, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "union"):
            vm.mask_iou(empty, empty)
        with self.assertRaisesRegex(ValueError, "target"):
            vm.mask_coverage(empty, np.ones((2, 2), dtype=np.uint8))


class AggregationTests(unittest.TestCase):
    @staticmethod
    def frames(class_name, masks, depths=None, confidences=None):
        depths = depths or [float(index + 1) for index in range(len(masks))]
        confidences = confidences or [0.90] * len(masks)
        frames = [{} for _ in range(vm.FRAME_COUNT)]
        for index, (mask, depth, confidence) in enumerate(
            zip(masks, depths, confidences)
        ):
            frames[index][class_name] = {
                "mask": np.array(mask, dtype=np.uint8),
                "depth_cm": float(depth),
                "confidence": float(confidence),
            }
        return frames

    def test_six_frames_are_unreliable_and_seven_are_reliable(self):
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        self.assertEqual(
            vm.aggregate_observations(self.frames("apple", [mask] * 6)), {}
        )
        reliable = vm.aggregate_observations(self.frames("apple", [mask] * 7))
        self.assertEqual(list(reliable), ["apple"])
        np.testing.assert_array_equal(reliable["apple"]["mask"], mask)

    def test_depth_median_handles_odd_and_even_hit_counts(self):
        mask = np.ones((1, 1), dtype=np.uint8)
        cases = (
            ([70, 10, 60, 20, 50, 30, 40], 40.0),
            ([8, 1, 7, 2, 6, 3, 5, 4], 4.5),
        )
        for depths, expected in cases:
            with self.subTest(depths=depths):
                result = vm.aggregate_observations(
                    self.frames("orange", [mask] * len(depths), depths)
                )
                self.assertEqual(result["orange"]["depth_cm"], expected)

    def test_medoid_uses_iou_then_earliest_frame_not_confidence(self):
        representative = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        outliers = (
            np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint8),
            np.array([[0, 0, 0], [1, 1, 0]], dtype=np.uint8),
            np.array([[0, 0, 0], [0, 1, 1]], dtype=np.uint8),
        )
        masks = [representative] * 4 + list(outliers)
        confidences = [0.50, 0.51, 0.52, 0.53, 0.99, 0.98, 0.97]
        result = vm.aggregate_observations(
            self.frames("cup", masks, confidences=confidences)
        )
        np.testing.assert_array_equal(result["cup"]["mask"], representative)

        tied = []
        for column in range(7):
            mask = np.zeros((1, 7), dtype=np.uint8)
            mask[0, column] = 1
            tied.append(mask)
        first = vm.aggregate_observations(self.frames("book", tied))
        reversed_result = vm.aggregate_observations(
            self.frames("book", list(reversed(tied)))
        )
        np.testing.assert_array_equal(first["book"]["mask"], tied[0])
        np.testing.assert_array_equal(reversed_result["book"]["mask"], tied[-1])


class VisibleMissingTests(unittest.TestCase):
    @staticmethod
    def mask_at(row, column):
        mask = np.zeros((vm.IMAGE_HEIGHT, vm.IMAGE_WIDTH), dtype=np.uint8)
        mask[row, column] = 1
        return mask

    @staticmethod
    def set_reference(memory, class_name, mask, depth, state="visible"):
        index = vm.CLASSES.index(class_name)
        memory["masks"][index] = mask
        memory["depth_cm"][index] = depth
        memory["has_reference"][index] = True
        memory["states"][index] = state

    def test_visible_updates_and_missing_preserves_history(self):
        history = vm.create_blank_memory()
        apple_old = self.mask_at(1, 1)
        orange_old = self.mask_at(2, 2)
        book_old = self.mask_at(3, 3)
        self.set_reference(history, "apple", apple_old, 10.0)
        self.set_reference(history, "orange", orange_old, 20.0, "occluded")
        history["occluders"][1] = "book"
        self.set_reference(history, "book", book_old, 40.0)

        apple_current = self.mask_at(1, 1)
        orange_moved = self.mask_at(20, 20)
        cup_new = self.mask_at(30, 30)
        reliable = {
            "apple": {"mask": apple_current, "depth_cm": 11.0},
            "orange": {"mask": orange_moved, "depth_cm": 22.0},
            "cup": {"mask": cup_new, "depth_cm": 33.0},
        }
        result = vm.update_visible_missing(history, reliable)

        self.assertEqual(
            result["states"].tolist(),
            ["visible", "visible", "visible", "missing"],
        )
        self.assertEqual(result["occluders"].tolist(), ["NA"] * 4)
        self.assertTrue(result["has_reference"].all())
        for class_name, mask, depth in (
            ("apple", apple_current, 11.0),
            ("orange", orange_moved, 22.0),
            ("cup", cup_new, 33.0),
            ("book", book_old, 40.0),
        ):
            index = vm.CLASSES.index(class_name)
            np.testing.assert_array_equal(result["masks"][index], mask)
            self.assertEqual(result["depth_cm"][index], depth)

        self.assertEqual(history["states"][3], "visible")
        np.testing.assert_array_equal(history["masks"][1], orange_old)

    def test_never_seen_object_stays_missing_without_reference(self):
        result = vm.update_visible_missing(vm.create_blank_memory(), {})
        self.assertEqual(result["states"].tolist(), ["missing"] * 4)
        self.assertEqual(result["occluders"].tolist(), ["NA"] * 4)
        self.assertFalse(result["has_reference"].any())
        self.assertFalse(result["masks"].any())
        self.assertTrue(np.isnan(result["depth_cm"]).all())


class OcclusionTests(unittest.TestCase):
    @staticmethod
    def mask(columns):
        mask = np.zeros((vm.IMAGE_HEIGHT, vm.IMAGE_WIDTH), dtype=np.uint8)
        mask[0, list(columns)] = 1
        return mask

    @staticmethod
    def set_object(memory, class_name, mask, depth, state):
        index = vm.CLASSES.index(class_name)
        memory["masks"][index] = mask
        memory["depth_cm"][index] = depth
        memory["has_reference"][index] = True
        memory["states"][index] = state

    def memory_for(self, target="apple", target_mask=None, target_depth=100.0):
        memory = vm.create_blank_memory()
        target_mask = target_mask if target_mask is not None else self.mask(range(10))
        self.set_object(memory, target, target_mask, target_depth, "missing")
        return memory

    def test_coverage_and_depth_conditions_are_strict(self):
        cases = (
            ("exact coverage", "apple", self.mask(range(8)), 90.0, "missing"),
            ("above coverage", "orange", self.mask(range(9)), 90.0, "occluded"),
            ("equal depth", "apple", self.mask(range(9)), 100.0, "missing"),
            ("farther", "apple", self.mask(range(9)), 110.0, "missing"),
        )
        for name, target, blocker_mask, blocker_depth, expected in cases:
            with self.subTest(name=name):
                memory = self.memory_for(target)
                self.set_object(memory, "cup", blocker_mask, blocker_depth, "visible")
                result = vm.apply_occlusion_states(memory)
                target_index = vm.CLASSES.index(target)
                self.assertEqual(result["states"][target_index], expected)
                self.assertEqual(
                    result["occluders"][target_index],
                    "cup" if expected == "occluded" else "NA",
                )

    def test_ignores_illegal_blocker_and_target_without_reference(self):
        target_mask = self.mask(range(10))
        memory = self.memory_for(target_mask=target_mask)
        self.set_object(memory, "orange", target_mask, 90.0, "visible")
        result = vm.apply_occlusion_states(memory)
        self.assertEqual(result["states"][0], "missing")

        no_reference = vm.create_blank_memory()
        self.set_object(no_reference, "cup", target_mask, 90.0, "visible")
        result = vm.apply_occlusion_states(no_reference)
        self.assertEqual(result["states"][0], "missing")
        self.assertFalse(result["has_reference"][0])

    def test_chooses_maximum_coverage_then_fixed_candidate_order(self):
        target_mask = self.mask(range(10))
        memory = self.memory_for(target_mask=target_mask)
        self.set_object(memory, "cup", self.mask(range(9)), 90.0, "visible")
        self.set_object(memory, "book", target_mask, 80.0, "visible")
        result = vm.apply_occlusion_states(memory)
        self.assertEqual(result["occluders"][0], "book")
        np.testing.assert_array_equal(result["masks"][0], target_mask)
        self.assertEqual(result["depth_cm"][0], 100.0)

        tied = self.memory_for(target_mask=target_mask)
        self.set_object(tied, "cup", target_mask, 90.0, "visible")
        self.set_object(tied, "book", target_mask, 80.0, "visible")
        self.assertEqual(vm.apply_occlusion_states(tied)["occluders"][0], "cup")


class CoreRegressionTests(unittest.TestCase):
    DEPTH_SCALE_M = 0.001

    @staticmethod
    def mask(pixel_count, offset=0):
        mask = np.zeros((vm.IMAGE_HEIGHT, vm.IMAGE_WIDTH), dtype=np.uint8)
        mask.flat[offset : offset + pixel_count] = 1
        return mask

    @classmethod
    def raw_frame(cls, detections=()):
        depth = np.zeros((vm.IMAGE_HEIGHT, vm.IMAGE_WIDTH), dtype=np.float32)
        class_ids, scores, masks = [], [], []
        for class_name, mask, depth_cm, score in detections:
            class_ids.append(vm.COCO_IDS[class_name])
            scores.append(score)
            masks.append(mask)
            depth[mask.astype(bool)] = depth_cm / (cls.DEPTH_SCALE_M * 100.0)
        return class_ids, scores, masks, depth, cls.DEPTH_SCALE_M

    @classmethod
    def frames_for(cls, class_name, mask, depth_cm, hits=7, score=0.90):
        detected = cls.raw_frame(((class_name, mask, depth_cm, score),))
        empty = cls.raw_frame()
        return [detected] * hits + [empty] * (vm.FRAME_COUNT - hits)

    @staticmethod
    def set_reference(
        memory, class_name, mask, depth_cm, state="visible", occluder="NA"
    ):
        index = vm.CLASSES.index(class_name)
        memory["masks"][index] = mask
        memory["depth_cm"][index] = depth_cm
        memory["has_reference"][index] = True
        memory["states"][index] = state
        memory["occluders"][index] = occluder

    def test_v01_v02_reliability_boundary(self):
        mask = self.mask(20)
        visible = vm.update_memory(
            vm.create_blank_memory(), self.frames_for("apple", mask, 42.0, hits=7)
        )
        self.assertEqual(visible["states"][0], "visible")
        self.assertTrue(visible["has_reference"][0])
        np.testing.assert_array_equal(visible["masks"][0], mask)
        self.assertAlmostEqual(visible["depth_cm"][0], 42.0)

        missing = vm.update_memory(
            vm.create_blank_memory(), self.frames_for("apple", mask, 42.0, hits=6)
        )
        self.assertEqual(missing["states"][0], "missing")
        self.assertFalse(missing["has_reference"][0])

    def test_v03_duplicate_instance_and_v04_incomplete_capture(self):
        low_mask = self.mask(10)
        high_mask = self.mask(10, offset=20)
        duplicate = self.raw_frame(
            (
                ("cup", low_mask, 20.0, 0.60),
                ("cup", high_mask, 30.0, 0.90),
            )
        )
        high_only = self.raw_frame((("cup", high_mask, 30.0, 0.90),))
        frames = [duplicate] + [high_only] * 6 + [self.raw_frame()] * 3
        result = vm.update_memory(vm.create_blank_memory(), frames)
        np.testing.assert_array_equal(result["masks"][2], high_mask)
        self.assertAlmostEqual(result["depth_cm"][2], 30.0)

        history = result
        snapshot = {key: value.copy() for key, value in history.items()}
        with self.assertRaisesRegex(ValueError, "frames"):
            vm.update_memory(history, [self.raw_frame()] * 9)
        for key in vm.MEMORY_KEYS:
            np.testing.assert_array_equal(history[key], snapshot[key])

    def test_v05_v06_moved_and_new_objects_update_references(self):
        history = vm.create_blank_memory()
        old_mask = self.mask(10)
        moved_mask = self.mask(10, offset=100)
        self.set_reference(history, "apple", old_mask, 50.0)
        moved = vm.update_memory(history, self.frames_for("apple", moved_mask, 60.0))
        self.assertEqual(moved["states"][0], "visible")
        np.testing.assert_array_equal(moved["masks"][0], moved_mask)
        self.assertAlmostEqual(moved["depth_cm"][0], 60.0)

        cup_mask = self.mask(10, offset=200)
        new_cup = vm.update_memory(
            vm.create_blank_memory(), self.frames_for("cup", cup_mask, 40.0)
        )
        self.assertEqual(new_cup["states"][2], "visible")
        self.assertTrue(new_cup["has_reference"][2])
        np.testing.assert_array_equal(new_cup["masks"][2], cup_mask)

    def test_v07_v08_v09_occlusion_boundaries(self):
        target_mask = self.mask(100)
        cases = (
            ("V7", 81, 90.0, "occluded"),
            ("V8", 80, 90.0, "missing"),
            ("V9 equal", 81, 100.0, "missing"),
            ("V9 farther", 81, 110.0, "missing"),
        )
        for name, blocker_pixels, blocker_depth, expected in cases:
            with self.subTest(name=name):
                history = vm.create_blank_memory()
                self.set_reference(history, "apple", target_mask, 100.0)
                blocker_mask = self.mask(blocker_pixels)
                result = vm.update_memory(
                    history,
                    self.frames_for("book", blocker_mask, blocker_depth),
                )
                self.assertEqual(result["states"][0], expected)
                self.assertEqual(
                    result["occluders"][0],
                    "book" if expected == "occluded" else "NA",
                )

    def test_v10_book_is_missing_not_occluded(self):
        target_mask = self.mask(100)
        history = vm.create_blank_memory()
        self.set_reference(history, "book", target_mask, 100.0)
        result = vm.update_memory(
            history, self.frames_for("cup", target_mask, 90.0)
        )
        self.assertEqual(result["states"][3], "missing")
        self.assertEqual(result["occluders"][3], "NA")
        np.testing.assert_array_equal(result["masks"][3], target_mask)

    def test_v11_reobserved_apple_becomes_visible_and_updates(self):
        history = vm.create_blank_memory()
        old_mask = self.mask(100)
        new_mask = self.mask(50, offset=200)
        self.set_reference(history, "apple", old_mask, 100.0, "occluded", "book")
        result = vm.update_memory(history, self.frames_for("apple", new_mask, 70.0))
        self.assertEqual(result["states"][0], "visible")
        self.assertEqual(result["occluders"][0], "NA")
        np.testing.assert_array_equal(result["masks"][0], new_mask)
        self.assertAlmostEqual(result["depth_cm"][0], 70.0)

    def test_v12_v13_reinfer_occlusion_without_inventing_history(self):
        target_mask = self.mask(100)
        blocker_mask = self.mask(81)
        history = vm.create_blank_memory()
        self.set_reference(history, "apple", target_mask, 100.0, "occluded", "book")
        still_occluded = vm.update_memory(
            history, self.frames_for("book", blocker_mask, 90.0)
        )
        self.assertEqual(still_occluded["states"][0], "occluded")
        self.assertEqual(still_occluded["occluders"][0], "book")
        np.testing.assert_array_equal(still_occluded["masks"][0], target_mask)
        self.assertEqual(still_occluded["depth_cm"][0], 100.0)

        never_seen = vm.update_memory(
            vm.create_blank_memory(), self.frames_for("book", blocker_mask, 90.0)
        )
        self.assertEqual(never_seen["states"][0], "missing")
        self.assertFalse(never_seen["has_reference"][0])
        self.assertEqual(never_seen["occluders"][0], "NA")


class VisionWorkflowTests(unittest.TestCase):
    @staticmethod
    def visible_memory(class_name="cup", depth_cm=50.0):
        memory = vm.create_blank_memory()
        index = vm.CLASSES.index(class_name)
        memory["masks"][index, 10:20, 10:20] = 1
        memory["depth_cm"][index] = depth_cm
        memory["has_reference"][index] = True
        memory["states"][index] = "visible"
        return memory

    def test_build_initial_commits_false_reports_visible_and_empty(self):
        cases = (
            (self.visible_memory(), "检测到物体 cup", "可靠物体总数：1"),
            (
                vm.create_blank_memory(),
                "未检测到可靠物体",
                "可靠物体总数：0",
            ),
        )
        for memory, expected_line, expected_total in cases:
            with self.subTest(expected=expected_total):
                with tempfile.TemporaryDirectory() as directory_text:
                    directory = Path(directory_text)
                    with mock.patch.object(
                        vm, "observe_memory", return_value=memory
                    ):
                        with self.assertLogs("vision_memory", level="INFO") as logs:
                            result = vm.build_initial(None, directory, 30.0)
                    self.assertIs(result, memory)
                    self.assertFalse(vm.read_commit_marker(directory))
                    saved = vm.load_memory_npz(directory / "memory_initial.npz")
                    np.testing.assert_array_equal(saved["states"], memory["states"])
                    joined = "\n".join(logs.output)
                    self.assertIn(expected_line, joined)
                    self.assertIn(expected_total, joined)

    def test_build_initial_observation_failure_preserves_old_commit(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            old = self.visible_memory("book")
            vm.commit_memory_files(directory, initial_memory=old, finalize=False)
            vm.commit_memory_files(directory, current_memory=old, finalize=True)
            old_initial = (directory / "memory_initial.npz").read_bytes()

            with mock.patch.object(
                vm, "observe_memory", side_effect=RuntimeError("camera")
            ):
                with self.assertRaisesRegex(RuntimeError, "camera"):
                    vm.build_initial(None, directory, 30.0)

            self.assertTrue(vm.read_commit_marker(directory))
            self.assertEqual(
                (directory / "memory_initial.npz").read_bytes(), old_initial
            )

    def test_manual_move_waits_for_exact_s_then_e(self):
        answers = iter(("x", "s", "wrong", "e"))
        prompts = []

        def input_fn(prompt):
            prompts.append(prompt)
            return next(answers)

        vm.wait_for_manual_move(input_fn)

        self.assertEqual(len(prompts), 4)
        self.assertIn("s", prompts[0])
        self.assertIn("s", prompts[1])
        self.assertIn("e", prompts[2])
        self.assertIn("e", prompts[3])

    def test_build_current_commits_current_json_and_true_last(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            initial = self.visible_memory("apple", 60.0)
            current = self.visible_memory("cup", 40.0)
            vm.commit_memory_files(
                directory, initial_memory=initial, finalize=False
            )
            initial_bytes = (directory / "memory_initial.npz").read_bytes()

            with mock.patch.object(vm, "observe_memory", return_value=current):
                result = vm.build_current(None, initial, directory, 30.0)

            self.assertIs(result, current)
            self.assertTrue(vm.read_commit_marker(directory))
            self.assertEqual(
                (directory / "memory_initial.npz").read_bytes(), initial_bytes
            )
            saved = vm.load_memory_npz(directory / "memory_current.npz")
            np.testing.assert_array_equal(saved["states"], current["states"])
            self.assertEqual(
                json.loads((directory / "task_memory.json").read_text()),
                vm.project_task_memory(current),
            )

    def test_reobserve_three_states_return_zero_and_keep_only_fixed_files(self):
        outputs = []
        visible = self.visible_memory("apple", 40.0)
        outputs.append(visible)

        occluded = self.visible_memory("apple", 50.0)
        occluded["states"][0] = "occluded"
        occluded["occluders"][0] = "book"
        outputs.append(occluded)

        missing = self.visible_memory("apple", 60.0)
        missing["states"][0] = "missing"
        outputs.append(missing)

        class Segmenter:
            def close(self):
                pass

        for current in outputs:
            expected_state = str(current["states"][0])
            with self.subTest(state=expected_state):
                with tempfile.TemporaryDirectory() as directory_text:
                    directory = Path(directory_text)
                    initial = self.visible_memory("book")
                    history = self.visible_memory("apple")
                    vm.commit_memory_files(
                        directory, initial_memory=initial, finalize=False
                    )
                    vm.commit_memory_files(
                        directory, current_memory=history, finalize=True
                    )
                    initial_bytes = (directory / "memory_initial.npz").read_bytes()

                    with mock.patch.object(
                        vm, "RknnSegmenter", return_value=Segmenter()
                    ), mock.patch.object(
                        vm, "observe_memory", return_value=current
                    ):
                        status = vm.main(
                            [
                                "reobserve",
                                "--model",
                                "/unused/model.rknn",
                                "--memory-dir",
                                str(directory),
                            ]
                        )

                    self.assertEqual(status, 0)
                    self.assertTrue(vm.read_commit_marker(directory))
                    self.assertEqual(
                        (directory / "memory_initial.npz").read_bytes(),
                        initial_bytes,
                    )
                    self.assertEqual(
                        set(path.name for path in directory.iterdir()),
                        {
                            "memory_initial.npz",
                            "memory_current.npz",
                            "task_memory.json",
                            "commit.json",
                        },
                    )
                    task = json.loads(
                        (directory / "task_memory.json").read_text()
                    )
                    self.assertEqual(task["apple"]["state"], expected_state)


class VisionFailureTests(unittest.TestCase):
    def test_reobserve_rejects_commit_and_current_before_model_init(self):
        memory = VisionWorkflowTests.visible_memory()
        with tempfile.TemporaryDirectory() as missing_text:
            missing = Path(missing_text)
            with mock.patch.object(vm, "RknnSegmenter") as constructor:
                self.assertEqual(
                    vm.main(["reobserve", "--memory-dir", str(missing)]), 1
                )
            constructor.assert_not_called()

        with tempfile.TemporaryDirectory() as false_text:
            false = Path(false_text)
            vm.commit_memory_files(false, initial_memory=memory, finalize=False)
            with mock.patch.object(vm, "RknnSegmenter") as constructor:
                self.assertEqual(
                    vm.main(["reobserve", "--memory-dir", str(false)]), 1
                )
            constructor.assert_not_called()

        with tempfile.TemporaryDirectory() as corrupt_text:
            corrupt = Path(corrupt_text)
            vm.commit_memory_files(corrupt, initial_memory=memory, finalize=False)
            vm.commit_memory_files(corrupt, current_memory=memory, finalize=True)
            (corrupt / "memory_current.npz").write_bytes(b"broken")
            with mock.patch.object(vm, "RknnSegmenter") as constructor:
                self.assertEqual(
                    vm.main(["reobserve", "--memory-dir", str(corrupt)]), 1
                )
            constructor.assert_not_called()

    def test_model_failure_returns_nonzero_without_observation(self):
        with mock.patch.object(
            vm, "RknnSegmenter", side_effect=RuntimeError("load_rknn failed")
        ), mock.patch.object(vm, "observe_memory") as observe:
            status = vm.main(["build", "--model", "/invalid/model.rknn"])

        self.assertEqual(status, 1)
        observe.assert_not_called()

    def test_observation_failure_releases_runtime_and_preserves_old_commit(self):
        class Segmenter:
            def __init__(self):
                self.closed = 0

            def close(self):
                self.closed += 1

        segmenter = Segmenter()
        memory = VisionWorkflowTests.visible_memory()
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            vm.commit_memory_files(
                directory, initial_memory=memory, finalize=False
            )
            vm.commit_memory_files(
                directory, current_memory=memory, finalize=True
            )
            before = {
                path.name: path.read_bytes()
                for path in directory.iterdir()
            }
            with mock.patch.object(
                vm, "RknnSegmenter", return_value=segmenter
            ), mock.patch.object(
                vm, "observe_memory", side_effect=RuntimeError("alignment failed")
            ):
                status = vm.main(
                    ["build", "--memory-dir", str(directory)]
                )

            self.assertEqual(status, 1)
            self.assertEqual(segmenter.closed, 1)
            self.assertTrue(vm.read_commit_marker(directory))
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in directory.iterdir()
                },
                before,
            )

    def test_second_observation_failure_leaves_incomplete_build(self):
        initial = VisionWorkflowTests.visible_memory("cup")
        answers = iter(("s", "e"))
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            with mock.patch.object(
                vm,
                "observe_memory",
                side_effect=(initial, TimeoutError("capture timeout")),
            ):
                with self.assertRaisesRegex(TimeoutError, "capture timeout"):
                    vm.run_build(
                        None,
                        directory,
                        30.0,
                        input_fn=lambda _prompt: next(answers),
                    )

            self.assertFalse(vm.read_commit_marker(directory))
            self.assertTrue((directory / "memory_initial.npz").is_file())
            self.assertFalse((directory / "memory_current.npz").exists())
            self.assertFalse((directory / "task_memory.json").exists())


class TaskManagerFixtures:
    @staticmethod
    def task_memory(target=None, state="missing", occluder="NA"):
        memory = {
            class_name: {
                "class": class_name,
                "state": "missing",
                "occluder": "NA",
            }
            for class_name in tm.TASK_CLASSES
        }
        if target is not None:
            memory[target]["state"] = state
            memory[target]["occluder"] = occluder
        return memory

    @staticmethod
    def write_task_files(directory, memory, commit=None):
        if commit is None:
            commit = {"complete": True}
        (directory / "commit.json").write_text(json.dumps(commit))
        (directory / "task_memory.json").write_text(json.dumps(memory))


class TaskManagerContractTests(TaskManagerFixtures, unittest.TestCase):

    def test_cli_and_exact_instruction_mapping_arm_once(self):
        self.assertEqual(tm.parse_args([]).memory_dir, tm.DEFAULT_MEMORY_DIR)
        self.assertEqual(
            tm.parse_args(["--memory-dir", "/tmp/test-memory"]).memory_dir,
            Path("/tmp/test-memory"),
        )

        for instruction, target in tm.TARGET_INSTRUCTIONS.items():
            with self.subTest(target=target):
                result = tm.read_armed_instruction(
                    lambda _prompt, value="  {}  ".format(instruction): value
                )
                self.assertEqual(result, (instruction, target))

        valid = next(iter(tm.TARGET_INSTRUCTIONS))
        answers = iter(
            (
                "",
                valid.upper(),
                valid.replace("pick", "take"),
                valid + " now",
                "  {}  ".format(valid),
            )
        )
        with self.assertLogs("task_manager", level="INFO") as logs:
            instruction, target = tm.read_armed_instruction(
                lambda _prompt: next(answers)
            )
        self.assertEqual(instruction, valid)
        self.assertEqual(target, tm.TARGET_INSTRUCTIONS[valid])
        self.assertEqual(
            sum("任务已武装" in line for line in logs.output),
            1,
        )
        source = Path(tm.__file__).read_text()
        self.assertNotIn("Type RUN", source)
        self.assertNotIn("ConfirmMotion", source)

    def test_task_memory_commit_and_schema_validation(self):
        valid = self.task_memory("apple", "occluded", "book")
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            self.write_task_files(directory, valid)
            self.assertEqual(tm.load_task_memory(directory), valid)

        commit_cases = (
            {"complete": False},
            {"complete": True, "generation": 1},
            {"complete": 1},
        )
        for commit in commit_cases:
            with self.subTest(commit=commit):
                with tempfile.TemporaryDirectory() as directory_text:
                    directory = Path(directory_text)
                    self.write_task_files(directory, valid, commit)
                    with self.assertRaisesRegex(ValueError, "commit"):
                        tm.load_task_memory(directory)

        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            with self.assertRaisesRegex(ValueError, "commit"):
                tm.load_task_memory(directory)
            (directory / "commit.json").write_text('{"complete": true}')
            with self.assertRaisesRegex(ValueError, "task_memory"):
                tm.load_task_memory(directory)
            (directory / "task_memory.json").write_text("broken")
            with self.assertRaisesRegex(ValueError, "task_memory"):
                tm.load_task_memory(directory)

        invalid_memories = []
        invalid = self.task_memory()
        invalid.pop("book")
        invalid_memories.append(invalid)
        invalid = self.task_memory()
        invalid["apple"].pop("state")
        invalid_memories.append(invalid)
        invalid = self.task_memory()
        invalid["apple"]["extra"] = True
        invalid_memories.append(invalid)
        invalid = self.task_memory()
        invalid["apple"]["class"] = "orange"
        invalid_memories.append(invalid)
        invalid = self.task_memory()
        invalid["apple"]["state"] = "unknown"
        invalid_memories.append(invalid)
        invalid = self.task_memory("apple", "occluded", "orange")
        invalid_memories.append(invalid)
        invalid = self.task_memory("book", "occluded", "cup")
        invalid_memories.append(invalid)
        invalid = self.task_memory("apple", "visible", "book")
        invalid_memories.append(invalid)
        invalid = self.task_memory("apple", "occluded", "book")
        invalid["apple"]["occluder"] = []
        invalid_memories.append(invalid)
        for invalid in invalid_memories:
            with self.subTest(memory=invalid):
                with self.assertRaises(ValueError):
                    tm.validate_task_memory(invalid)

    def test_invalid_commit_blocks_input_and_process_creation(self):
        with tempfile.TemporaryDirectory() as directory_text:
            input_fn = mock.Mock(side_effect=AssertionError("input called"))
            with mock.patch.object(tm.subprocess, "Popen") as popen:
                status = tm.main(
                    ["--memory-dir", directory_text], input_fn=input_fn
                )
            self.assertEqual(status, 1)
            input_fn.assert_not_called()
            popen.assert_not_called()

    def test_openvla_arguments_are_fixed_lists(self):
        instructions = dict(tm.TARGET_INSTRUCTIONS)
        instructions.update(
            {
                instruction: occluder
                for occluder, instruction in tm.OCCLUDER_INSTRUCTIONS.items()
            }
        )
        for instruction in instructions:
            with self.subTest(instruction=instruction):
                self.assertEqual(
                    tm.build_openvla_args(instruction),
                    [
                        str(tm.OPENVLA_PATH),
                        "--instruction",
                        instruction,
                        "--max-steps",
                        "1000",
                        "--enable-motion",
                    ],
                )
        for instruction in ('"; touch /tmp/bad', "pick up the banana"):
            with self.subTest(instruction=instruction):
                with self.assertRaisesRegex(ValueError, "not allowed"):
                    tm.build_openvla_args(instruction)


class TaskManagerProcessTests(unittest.TestCase):
    class FakeProcess:
        def __init__(self, returncode=None, signal_error=None):
            self.returncode = returncode
            self.signal_error = signal_error
            self.signals = []

        def poll(self):
            return self.returncode

        def send_signal(self, value):
            self.signals.append(value)
            if self.signal_error is not None:
                raise self.signal_error
            self.returncode = -value

    def test_control_wait_rejects_wrong_key_and_detects_early_exit(self):
        process = self.FakeProcess()
        stream = io.StringIO("x\nc\n")
        with mock.patch.object(
            tm.select,
            "select",
            side_effect=lambda readers, _w, _x, _timeout: (
                readers,
                [],
                [],
            ),
        ):
            with self.assertLogs("task_manager", level="INFO") as logs:
                tm.wait_for_control(process, "c", "visible", stream)
        self.assertTrue(any("输入无效" in line for line in logs.output))

        exited = self.FakeProcess(returncode=7)
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            tm.wait_for_control(exited, "q", "occluder", io.StringIO())

    def test_normal_eof_and_keyboard_interrupt_signal_once(self):
        instruction = next(iter(tm.TARGET_INSTRUCTIONS))

        normal = self.FakeProcess()
        with mock.patch.object(
            tm, "launch_openvla", return_value=normal
        ), mock.patch.object(tm, "wait_for_control"):
            tm.run_motion(instruction, "c", "normal")
        self.assertEqual(normal.signals, [signal.SIGINT])

        eof = self.FakeProcess()
        with mock.patch.object(
            tm, "launch_openvla", return_value=eof
        ), mock.patch.object(
            tm.select,
            "select",
            side_effect=lambda readers, _w, _x, _timeout: (
                readers,
                [],
                [],
            ),
        ):
            with self.assertRaises(EOFError):
                tm.run_motion(instruction, "c", "eof", io.StringIO())
        self.assertEqual(eof.signals, [signal.SIGINT])

        interrupted = self.FakeProcess()
        with mock.patch.object(
            tm, "launch_openvla", return_value=interrupted
        ), mock.patch.object(
            tm, "wait_for_control", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                tm.run_motion(instruction, "c", "interrupt")
        self.assertEqual(interrupted.signals, [signal.SIGINT])

    def test_main_abort_returns_nonzero_after_signaling_child(self):
        instruction = "pick up the apple and place it on the bowl"
        memory = TaskManagerFixtures.task_memory("apple", "visible")
        for error in (EOFError("input ended"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                process = self.FakeProcess()
                with mock.patch.object(
                    tm, "load_task_memory", return_value=memory
                ), mock.patch.object(
                    tm,
                    "read_armed_instruction",
                    return_value=(instruction, "apple"),
                ), mock.patch.object(
                    tm, "launch_openvla", return_value=process
                ), mock.patch.object(
                    tm, "wait_for_control", side_effect=error
                ):
                    status = tm.main([])
                self.assertEqual(status, 1)
                self.assertEqual(process.signals, [signal.SIGINT])

    def test_sigint_wait_reports_twice_without_forcing_child(self):
        child_code = (
            "import pathlib,signal,sys,time;"
            "count=[0];"
            "signal.signal(signal.SIGINT,lambda *_:count.__setitem__(0,count[0]+1));"
            "pathlib.Path(sys.argv[1]).write_text('ready');"
            "end=time.monotonic()+10.4;"
            "exec(\"while time.monotonic()<end:\\n time.sleep(0.1)\");"
            "pathlib.Path(sys.argv[2]).write_text(str(count[0]))"
        )
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            ready = directory / "ready"
            count = directory / "count"
            process = subprocess.Popen(
                [sys.executable, "-c", child_code, str(ready), str(count)]
            )
            deadline = time.monotonic() + 2.0
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            with self.assertLogs("task_manager", level="INFO") as logs:
                return_code = tm.interrupt_and_wait(process, "slow child")

            self.assertEqual(return_code, 0)
            self.assertEqual(count.read_text(), "1")
            self.assertGreaterEqual(
                sum("仍在等待" in line for line in logs.output),
                2,
            )


class TaskManagerWorkflowTests(TaskManagerFixtures, unittest.TestCase):
    def test_visible_and_missing_branches(self):
        instruction = "pick up the apple and place it on the bowl"
        stream = object()
        with mock.patch.object(tm, "run_motion") as motion, mock.patch.object(
            tm, "run_reobserve"
        ) as reobserve:
            tm.execute_task(
                self.task_memory("apple", "visible"),
                instruction,
                "apple",
                "/tmp/memory",
                stream,
            )
            motion.assert_called_once_with(
                instruction, "c", "执行目标任务", stream
            )
            reobserve.assert_not_called()

        with mock.patch.object(tm, "run_motion") as motion, mock.patch.object(
            tm, "run_reobserve"
        ) as reobserve:
            tm.execute_task(
                self.task_memory("apple", "missing"),
                instruction,
                "apple",
                "/tmp/memory",
            )
            motion.assert_not_called()
            reobserve.assert_not_called()

    def test_occluder_commands_and_reobserve_arguments(self):
        instruction = "pick up the apple and place it on the bowl"
        for occluder in ("book", "cup"):
            with self.subTest(occluder=occluder):
                updated = self.task_memory("apple", "missing")
                with mock.patch.object(tm, "run_motion") as motion, mock.patch.object(
                    tm, "run_reobserve"
                ) as reobserve, mock.patch.object(
                    tm, "load_task_memory", return_value=updated
                ):
                    tm.execute_task(
                        self.task_memory("apple", "occluded", occluder),
                        instruction,
                        "apple",
                        Path("/tmp/memory"),
                    )
                motion.assert_called_once_with(
                    tm.OCCLUDER_INSTRUCTIONS[occluder],
                    "q",
                    "移除遮挡物 {}".format(occluder),
                    None,
                )
                reobserve.assert_called_once_with(Path("/tmp/memory"))

        with mock.patch.object(tm.subprocess, "run") as run:
            tm.run_reobserve(Path("/tmp/memory"))
        run.assert_called_once_with(
            [
                sys.executable,
                str(tm.VISION_MEMORY_PATH),
                "reobserve",
                "--memory-dir",
                "/tmp/memory",
            ],
            check=True,
            shell=False,
        )

    def test_reobserve_visible_occluded_and_missing_results(self):
        instruction = "pick up the orange and place it on the bowl"
        initial = self.task_memory("orange", "occluded", "book")
        for state in ("visible", "occluded", "missing"):
            with self.subTest(state=state):
                occluder = "cup" if state == "occluded" else "NA"
                updated = self.task_memory("orange", state, occluder)
                with mock.patch.object(tm, "run_motion") as motion, mock.patch.object(
                    tm, "run_reobserve"
                ) as reobserve, mock.patch.object(
                    tm, "load_task_memory", return_value=updated
                ):
                    tm.execute_task(
                        initial,
                        instruction,
                        "orange",
                        "/tmp/memory",
                    )
                reobserve.assert_called_once_with("/tmp/memory")
                expected = [
                    mock.call(
                        tm.OCCLUDER_INSTRUCTIONS["book"],
                        "q",
                        "移除遮挡物 book",
                        None,
                    )
                ]
                if state == "visible":
                    expected.append(
                        mock.call(
                            instruction,
                            "q",
                            "重新执行目标任务",
                            None,
                        )
                    )
                self.assertEqual(motion.call_args_list, expected)

    def test_main_missing_is_success_without_process(self):
        memory = self.task_memory("orange", "missing")
        instruction = "pick up the orange and place it on the bowl"
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            self.write_task_files(directory, memory)
            with mock.patch.object(tm, "run_motion") as motion:
                status = tm.main(
                    ["--memory-dir", directory_text],
                    input_fn=lambda _prompt: instruction,
                )
        self.assertEqual(status, 0)
        motion.assert_not_called()


class TaskManagerFailureTests(TaskManagerFixtures, unittest.TestCase):
    def test_launch_missing_nonexecutable_and_start_failure(self):
        instruction = next(iter(tm.TARGET_INSTRUCTIONS))
        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "openvla"
            with mock.patch.object(tm, "OPENVLA_PATH", path):
                with self.assertRaisesRegex(RuntimeError, "not executable"):
                    tm.launch_openvla(instruction, "launch")

            path.write_text("placeholder")
            path.chmod(0o755)
            with mock.patch.object(tm, "OPENVLA_PATH", path), mock.patch.object(
                tm.subprocess,
                "Popen",
                side_effect=OSError("start error"),
            ):
                with self.assertRaisesRegex(RuntimeError, "launch.*start"):
                    tm.launch_openvla(instruction, "launch")

    def test_early_exit_and_sigint_failure_are_errors(self):
        instruction = next(iter(tm.TARGET_INSTRUCTIONS))
        exited = TaskManagerProcessTests.FakeProcess(returncode=3)
        with mock.patch.object(
            tm, "launch_openvla", return_value=exited
        ):
            with self.assertRaisesRegex(RuntimeError, "status 3"):
                tm.run_motion(instruction, "c", "early", io.StringIO())
        self.assertEqual(exited.signals, [])

        failed = TaskManagerProcessTests.FakeProcess(
            signal_error=OSError("signal error")
        )
        with mock.patch.object(
            tm, "launch_openvla", return_value=failed
        ), mock.patch.object(tm, "wait_for_control"):
            with self.assertRaisesRegex(RuntimeError, "SIGINT failed"):
                tm.run_motion(instruction, "c", "signal")
        self.assertEqual(failed.signals, [signal.SIGINT])

    def test_reobserve_failure_or_bad_update_stops_second_motion(self):
        instruction = "pick up the apple and place it on the bowl"
        initial = self.task_memory("apple", "occluded", "book")
        with mock.patch.object(tm, "run_motion") as motion, mock.patch.object(
            tm, "run_reobserve", side_effect=RuntimeError("reobserve failed")
        ), mock.patch.object(tm, "load_task_memory") as load:
            with self.assertRaisesRegex(RuntimeError, "reobserve"):
                tm.execute_task(
                    initial,
                    instruction,
                    "apple",
                    "/tmp/memory",
                )
        self.assertEqual(motion.call_count, 1)
        load.assert_not_called()

        with mock.patch.object(tm, "run_motion") as motion, mock.patch.object(
            tm, "run_reobserve"
        ), mock.patch.object(
            tm,
            "load_task_memory",
            side_effect=ValueError("task_memory: broken"),
        ):
            with self.assertRaisesRegex(RuntimeError, "reobserve result"):
                tm.execute_task(
                    initial,
                    instruction,
                    "apple",
                    "/tmp/memory",
                )
        self.assertEqual(motion.call_count, 1)

    def test_reobserve_nonzero_is_reported(self):
        error = subprocess.CalledProcessError(9, ["vision_memory.py"])
        with mock.patch.object(tm.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "status 9"):
                tm.run_reobserve("/tmp/memory")


if __name__ == "__main__":
    unittest.main()
