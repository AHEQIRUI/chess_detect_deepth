from abc import ABC, abstractmethod
import numpy as np
import cv2


class CameraBase(ABC):
    @abstractmethod
    def open(self) -> bool: ...
    @abstractmethod
    def capture(self) -> tuple[np.ndarray, np.ndarray]: ...
    @abstractmethod
    def close(self) -> None: ...


class MockCamera(CameraBase):
    """Mock camera for testing. Generates a synthetic chessboard image."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._opened = False

    def open(self) -> bool:
        self._opened = True
        return True

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._opened:
            raise RuntimeError("Camera not opened")
        bgr = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        depth = np.full((self.height, self.width), 500.0, dtype=np.float32)
        return bgr, depth

    def close(self) -> None:
        self._opened = False


class OrbbecCamera(CameraBase):
    """Real camera using Orbbec SDK for ASTRA Pro Plus.

    Requires the Orbbec SDK with Python bindings (pyorbbecsdk) installed.
    Applies camera calibration (undistortion) using factory intrinsics.
    """

    def __init__(self, calib_file: str | None = None):
        self._pipeline = None
        self._opened = False
        self._camera_matrix = None
        self._dist_coeffs = None
        self._calib_file = calib_file

    def open(self) -> bool:
        self._load_calibration()
        return self._start_pipeline()

    def _load_calibration(self):
        """Parse factory calibration file and compute undistortion maps."""
        import os
        path = self._calib_file
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..",
                                "CameraParameters_Astra Pro Plus.txt")
        try:
            params = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("Astra"):
                        continue
                    key, val = line.split(" = ")
                    params[key] = float(val)

            self._camera_matrix = np.array([
                [params["RGB fx"], 0, params["RGB cx"]],
                [0, params["RGB fy"], params["RGB cy"]],
                [0, 0, 1],
            ], dtype=np.float32)

            self._dist_coeffs = np.array([
                params["k1"], params["k2"], params["p1"], params["p2"], params["k3"],
            ], dtype=np.float32)
        except Exception as e:
            print(f"Warning: Could not load calibration ({e}), skipping undistortion")
            self._camera_matrix = None
            self._dist_coeffs = None

    def _start_pipeline(self) -> bool:
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

            config = Config()
            pipeline = Pipeline()

            color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            config.enable_stream(color_profile)

            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)

            config.set_align_mode(OBAlignMode.HW_MODE)
            pipeline.start(config)

            self._pipeline = pipeline
            self._opened = True
            return True
        except Exception as e:
            print(f"Failed to open Orbbec camera: {e}")
            self.close()
            return False

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._opened:
            raise RuntimeError("Camera not opened")

        for attempt in range(30):
            frames = self._pipeline.wait_for_frames(200)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is not None and depth_frame is not None:
                bgr = self._color_frame_to_bgr(color_frame)
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                scale = depth_frame.get_depth_scale()
                depth_mm = depth.astype(np.float32) * scale

                if self._camera_matrix is not None:
                    bgr = cv2.undistort(bgr, self._camera_matrix, self._dist_coeffs)

                return bgr, depth_mm

        raise RuntimeError("Failed to get frames from camera")

    @staticmethod
    def _color_frame_to_bgr(frame) -> np.ndarray:
        from pyorbbecsdk import OBFormat

        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asarray(frame.get_data())

        if color_format == OBFormat.RGB:
            image = data.reshape((height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            return data.reshape((height, width, 3))
        elif color_format == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
        elif color_format == OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            raise RuntimeError(f"Unsupported color format: {color_format}")

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._opened = False
