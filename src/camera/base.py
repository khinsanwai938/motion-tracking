from abc import ABC, abstractmethod
import numpy as np


class CameraSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    @abstractmethod
    def get_focal_length_px(self) -> float: ...

    @abstractmethod
    def release(self) -> None: ...