"""Trajectory 的 JSONL 追加存储和恢复。"""

import json
from pathlib import Path
from threading import Lock
from typing import Iterator, Union

from .trajectory import Trajectory, TrajectoryEvent


class TrajectoryStore:
    """以一行一个完整轨迹的方式保存长时间采集结果。"""

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._lock = Lock()

    def ensure_file(self) -> None:
        """确保 JSONL 文件和父目录存在，即使当前没有任何记录。"""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.touch(exist_ok=True)

    def append(self, trajectory: Trajectory) -> None:
        """追加一条轨迹，并在首次写入时创建父目录。"""
        with self._lock:
            # 采集任务通常边运行边落盘，因此采用 append 而不是一次性重写文件。
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n"
                )

    def iter_trajectories(self) -> Iterator[Trajectory]:
        """按文件顺序恢复轨迹；文件不存在或空行只表示当前没有记录。"""
        if not self._path.exists():
            return

        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                # 反序列化时显式重建 dataclass，确保下游仍拿到统一对象而非裸字典。
                events = [TrajectoryEvent(**event) for event in payload["events"]]
                yield Trajectory(
                    task_id=payload["task_id"],
                    seed=payload["seed"],
                    events=events,
                    terminal_state=payload["terminal_state"],
                    evaluation=payload["evaluation"],
                )
