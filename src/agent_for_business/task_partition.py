"""为 Retail benchmark 建立 task-level 的可复现实验划分。"""

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable, Tuple, Union


RETAIL_VALIDATION_IDS = (
    "93", "95", "96", "98", "99", "103", "104", "105",
    "106", "107", "109", "110", "112", "113",
)
# final_test 只用于最终报告，不能参与 SFT validation benchmark。
RETAIL_FINAL_TEST_IDS = (
    "5", "9", "12", "17", "18", "26", "27", "32", "33", "36",
    "38", "39", "40", "42", "45", "49", "51", "53", "55", "56",
    "60", "61", "62", "64", "65", "68", "70", "71", "74", "77",
)


@dataclass(frozen=True)
class TaskPartition:
    """记录 train、validation、final_test 和 reserve 各自拥有的 task id。"""

    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    final_test: Tuple[str, ...]
    reserve: Tuple[str, ...]


def build_task_partition(
    official_train: Iterable[str],
    official_test: Iterable[str],
    validation_ids: Iterable[str],
    final_test_ids: Iterable[str],
) -> TaskPartition:
    """校验并构建互不重叠的划分，同时保留官方 task 顺序。"""

    train_ids = tuple(official_train)
    test_ids = tuple(official_test)
    validation_set = set(validation_ids)
    final_test_set = set(final_test_ids)

    if validation_set & final_test_set:
        raise ValueError("task IDs overlap between validation and final_test")
    unknown_validation = validation_set - set(train_ids)
    if unknown_validation:
        raise ValueError(
            "validation task IDs not in official_train: "
            + ", ".join(sorted(unknown_validation))
        )
    unknown_final_test = final_test_set - set(test_ids)
    if unknown_final_test:
        raise ValueError(
            "final_test task IDs not in official_test: "
            + ", ".join(sorted(unknown_final_test))
        )

    # validation 从 official_train 中扣出；final_test 从 official_test 中选出，
    # 这样训练、validation 和最终评估天然使用不同的 task 集合。
    return TaskPartition(
        train=tuple(task_id for task_id in train_ids if task_id not in validation_set),
        validation=tuple(task_id for task_id in train_ids if task_id in validation_set),
        final_test=tuple(task_id for task_id in test_ids if task_id in final_test_set),
        reserve=tuple(task_id for task_id in test_ids if task_id not in final_test_set),
    )


def load_retail_task_partition(
    split_tasks_path: Union[str, PathLike],
) -> TaskPartition:
    """读取官方 Retail split，并应用项目固定的 validation/final_test manifest。"""

    split_tasks = json.loads(Path(split_tasks_path).read_text(encoding="utf-8"))
    return build_task_partition(
        official_train=split_tasks["train"],
        official_test=split_tasks["test"],
        validation_ids=RETAIL_VALIDATION_IDS,
        final_test_ids=RETAIL_FINAL_TEST_IDS,
    )
