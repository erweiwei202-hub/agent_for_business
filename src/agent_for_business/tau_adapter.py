"""把 τ³ SimulationRun 的公开消息接口转换为项目统一轨迹。"""

from typing import Any, Dict

from .trajectory import Trajectory, TrajectoryRecorder


class SimulationTrajectoryAdapter:
    """隔离 τ³ 消息对象结构，避免其他模块依赖第三方消息类型。"""

    def from_simulation(
        self,
        simulation: Any,
        *,
        terminal_state: Dict[str, Any],
        evaluation: Dict[str, Any],
    ) -> Trajectory:
        """按消息顺序转换 simulation，并附加调用方提供的终局信息。"""
        recorder = TrajectoryRecorder(
            task_id=simulation.task_id,
            seed=getattr(simulation, "seed", None),
        )

        for message in simulation.get_messages():
            role = getattr(message, "role", None)
            tool_calls = getattr(message, "tool_calls", None) or []

            if role == "user":
                recorder.append_user(getattr(message, "content", None) or "")
            elif role == "assistant":
                # 一个 assistant 消息可能同时包含多个工具动作和一段文本；
                # 两者都要保留，Verifier 才能检查调用顺序和确认语义。
                for tool_call in tool_calls:
                    recorder.append_tool_call(
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        call_id=tool_call.id,
                    )
                content = getattr(message, "content", None)
                if content:
                    recorder.append_assistant(content)
            elif role == "tool":
                # τ³ 的 tool 消息 id 对应前一个 tool call 的 id，保留它供校验器配对。
                recorder.append_tool_result(
                    call_id=getattr(message, "id", ""),
                    content=getattr(message, "content", None),
                )

        return recorder.finish(
            terminal_state=terminal_state,
            evaluation=evaluation,
        )
