#!/usr/bin/env python3
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs

import vision_memory as vm


WINDOW = "Book pose finder | s: save | q/ESC: quit"


def main():
    serials = {
        device.get_info(rs.camera_info.serial_number)
        for device in rs.context().query_devices()
    }
    if vm.CAMERA_SERIAL not in serials:
        raise RuntimeError("camera {} not found".format(vm.CAMERA_SERIAL))

    segmenter = vm.RknnSegmenter(vm.DEFAULT_MODEL_PATH)
    pipeline = rs.pipeline()
    started = False
    rolling = deque(maxlen=vm.FRAME_COUNT)
    last_status = None
    try:
        config = rs.config()
        config.enable_device(vm.CAMERA_SERIAL)
        config.enable_stream(
            rs.stream.color,
            vm.IMAGE_WIDTH,
            vm.IMAGE_HEIGHT,
            rs.format.bgr8,
            vm.CAMERA_FPS,
        )
        config.enable_stream(
            rs.stream.depth,
            vm.IMAGE_WIDTH,
            vm.IMAGE_HEIGHT,
            rs.format.z16,
            vm.CAMERA_FPS,
        )
        pipeline.start(config)
        started = True
        align = rs.align(rs.stream.color)
        for _ in range(vm.WARMUP_FRAMES):
            pipeline.wait_for_frames(1000)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 480)
        print(
            "实时窗口已启动：绿色 STABLE 表示"
            "最近 10 帧至少 7 帧有效。"
        )

        while True:
            frames = align.process(pipeline.wait_for_frames(1000))
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            image = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            boxes, class_ids, scores, masks = segmenter.infer(image)

            book = None
            for box, class_id, score, mask in zip(
                boxes, class_ids, scores, masks
            ):
                if int(class_id) == vm.COCO_IDS["book"]:
                    book = box, float(score), mask
                    break

            accepted = False
            confidence = 0.0
            depth_ratio = 0.0
            book_mask = np.zeros(
                (vm.IMAGE_HEIGHT, vm.IMAGE_WIDTH), dtype=np.uint8
            )
            annotated = image.copy()
            if book is not None:
                box, confidence, book_mask = book
                selected = book_mask.astype(bool)
                pixels = int(selected.sum())
                valid = int(
                    np.count_nonzero(
                        np.isfinite(depth[selected]) & (depth[selected] > 0)
                    )
                )
                depth_ratio = valid / pixels if pixels else 0.0
                accepted = (
                    pixels > 0 and depth_ratio >= vm.MIN_VALID_DEPTH_RATIO
                )
                mask_color = np.zeros_like(image)
                mask_color[selected] = (20, 210, 20)
                annotated = cv2.addWeighted(
                    mask_color, 0.38, annotated, 0.62, 0
                )
                x1, y1, x2, y2 = [int(round(value)) for value in box]
                outline = (20, 230, 20) if accepted else (0, 220, 255)
                contours = cv2.findContours(
                    book_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0]
                cv2.drawContours(annotated, contours, -1, outline, 2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), outline, 2)

            rolling.append(accepted)
            hits = sum(rolling)
            if len(rolling) < vm.FRAME_COUNT:
                status = "COLLECTING {}/10".format(len(rolling))
                status_color = (0, 220, 255)
            elif hits >= vm.RELIABLE_FRAME_COUNT:
                status = "STABLE {}/10".format(hits)
                status_color = (20, 230, 20)
            else:
                status = "UNSTABLE {}/10".format(hits)
                status_color = (20, 20, 235)

            cv2.rectangle(annotated, (0, 0), (640, 92), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                "BOOK {}".format(status),
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                2,
                cv2.LINE_AA,
            )
            detail = (
                "conf={:.3f} depth={:.1%}".format(confidence, depth_ratio)
                if book is not None
                else "not detected | threshold=0.20"
            )
            cv2.putText(
                annotated,
                detail,
                (14, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.67,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            mask_panel = np.zeros_like(image)
            mask_panel[book_mask.astype(bool)] = status_color
            cv2.putText(
                mask_panel,
                "BOOK MASK",
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask_panel,
                "s: save   q/ESC: quit",
                (14, 468),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            canvas = np.hstack((annotated, mask_panel))
            cv2.imshow(WINDOW, canvas)

            if status != last_status:
                print(
                    "{} conf={:.3f} depth={:.1%}".format(
                        status, confidence, depth_ratio
                    ),
                    flush=True,
                )
                last_status = status

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord("s"):
                path = Path(
                    "/tmp/book-pose-{}.png".format(
                        int(time.time() * 1000)
                    )
                )
                if not cv2.imwrite(str(path), canvas):
                    raise RuntimeError("failed to save {}".format(path))
                print("已保存 {}".format(path), flush=True)
    finally:
        if started:
            pipeline.stop()
        segmenter.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
