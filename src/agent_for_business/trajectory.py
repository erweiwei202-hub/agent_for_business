"""统一记录一次 τ³ Retail 任务执行过程的数据契约。"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryEvent:
    """按真实消息顺序保存 user、assistant、tool call 和 tool result。"""

    kind: str
    content: Any = None
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None


@dataclass
class Trajectory:
    """一条可持久化的完整轨迹，连接运行时、校验器和训练数据构建。"""

    task_id: str
    seed: Optional[int]
    # events 保存过程；terminal_state/evaluation 保存过程结束后的两个观察面。
    events: List[TrajectoryEvent]
    terminal_state: Dict[str, Any]
    evaluation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 兼容的嵌套字典，供 JSONL store 写入。"""
        return asdict(self)


class TrajectoryRecorder:
    """按事件发生顺序收集消息，并在任务结束时生成轨迹快照。"""

    def __init__(self, task_id: str, seed: Optional[int] = None) -> None:
        self._task_id = task_id
        self._seed = seed
        self._events: List[TrajectoryEvent] = []

    def append_user(self, content: str) -> None:
        """记录 User Simulator 发出的消息。"""
        self._events.append(TrajectoryEvent(kind="user_message", content=content))

    def append_tool_call(
        self,
        *,
        name: str,
        arguments: Dict[str, Any],
        call_id: str,
    ) -> None:
        """记录 assistant 请求工具的动作及其调用 id。"""
        self._events.append(
            TrajectoryEvent(
                kind="tool_call",
                tool_name=name,
                arguments=arguments,
                tool_call_id=call_id,
            )
        )

    def append_tool_result(self, *, call_id: str, content: Any) -> None:
        """记录工具返回值，并用调用 id 与对应 tool call 关联。"""
        self._events.append(
            TrajectoryEvent(
                kind="tool_result",
                content=content,
                tool_call_id=call_id,
            )
        )

    def append_assistant(self, content: str) -> None:
        """记录 assistant 的文本消息，包括最终回复或操作确认摘要。"""
        self._events.append(
            TrajectoryEvent(kind="assistant_message", content=content)
        )

    def finish(
        self,
        *,
        terminal_state: Dict[str, Any],
        evaluation: Dict[str, Any],
    ) -> Trajectory:
        """封存当前事件列表；复制列表避免后续 recorder 变化轨迹内容。"""
        return Trajectory(
            task_id=self._task_id,
            seed=self._seed,
            events=list(self._events),
            terminal_state=terminal_state,
            evaluation=evaluation,
        )
