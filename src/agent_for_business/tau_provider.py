"""τ³ Retail 运行时的懒加载 Provider 边界。"""

from typing import Any, Callable, Dict, List, Optional


class Tau2RetailProvider:
    """按 task id 启动一次 τ³ Retail SimulationRun。"""

    def __init__(
        self,
        *,
        agent_llm: str,
        user_llm: str,
        task_split_name: str = "base",
        agent_llm_args: Optional[Dict[str, Any]] = None,
        user_llm_args: Optional[Dict[str, Any]] = None,
        max_steps: int = 100,
        max_errors: int = 5,
        config_factory: Optional[Callable[..., Any]] = None,
        task_loader: Optional[Callable[..., List[Any]]] = None,
        single_task_runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.agent_llm = agent_llm
        self.user_llm = user_llm
        self.task_split_name = task_split_name
        self.agent_llm_args = agent_llm_args or {}
        self.user_llm_args = user_llm_args or {}
        self.max_steps = max_steps
        self.max_errors = max_errors
        self._config_factory = config_factory or self._default_config_factory
        self._task_loader = task_loader or self._default_task_loader
        self._single_task_runner = (
            single_task_runner or self._default_single_task_runner
        )

    def run(self, *, task_id: str, seed: int) -> Any:
        """构造运行配置、确认唯一任务后执行单任务 simulation。"""
        config = self._config_factory(
            domain="retail",
            task_split_name=self.task_split_name,
            agent="llm_agent",
            llm_agent=self.agent_llm,
            llm_args_agent=self.agent_llm_args,
            user="user_simulator",
            llm_user=self.user_llm,
            llm_args_user=self.user_llm_args,
            num_trials=1,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
        )
        tasks = self._task_loader(
            "retail",
            task_split_name=self.task_split_name,
            task_ids=[task_id],
        )
        # task id 是实验划分的基本单位；多于/少于一条都意味着配置或 split 有误。
        if len(tasks) != 1:
            raise ValueError(
                "Expected exactly one Retail task for task_id=" + task_id
            )
        return self._single_task_runner(config, tasks[0], seed=seed)

    @staticmethod
    def _default_config_factory(**kwargs: Any) -> Any:
        """在真正运行时才导入 τ³，避免核心测试依赖完整上游环境。"""
        from tau2 import TextRunConfig

        return TextRunConfig(**kwargs)

    @staticmethod
    def _default_task_loader(*args: Any, **kwargs: Any) -> List[Any]:
        """延迟调用 τ³ 的官方 task loader。"""
        from tau2.runner import get_tasks

        return get_tasks(*args, **kwargs)

    @staticmethod
    def _default_single_task_runner(config: Any, task: Any, *, seed: int) -> Any:
        """延迟调用 τ³ 的单任务执行器。"""
        from tau2.runner import run_single_task

        return run_single_task(
            config,
            task,
            seed=seed,
            evaluation_llm=getattr(config, "llm_agent", None),
            evaluation_llm_args=getattr(config, "llm_args_agent", None),
        )
