import pytest

from agent_for_business.task_partition import (
    build_task_partition,
    load_retail_task_partition,
)


def test_builds_task_level_train_validation_test_and_reserve_sets():
    partition = build_task_partition(
        official_train=["0", "1", "2", "3"],
        official_test=["4", "5", "6"],
        validation_ids=["1"],
        final_test_ids=["4", "5"],
    )

    assert partition.train == ("0", "2", "3")
    assert partition.validation == ("1",)
    assert partition.final_test == ("4", "5")
    assert partition.reserve == ("6",)


def test_rejects_task_overlap_between_splits():
    with pytest.raises(ValueError, match="overlap"):
        build_task_partition(
            official_train=["0", "1"],
            official_test=["2", "3"],
            validation_ids=["1"],
            final_test_ids=["1"],
        )


def test_rejects_manifest_ids_outside_official_splits():
    with pytest.raises(ValueError, match="not in official_train"):
        build_task_partition(
            official_train=["0", "1"],
            official_test=["2", "3"],
            validation_ids=["9"],
            final_test_ids=["2"],
        )


def test_loads_retail_split_tasks_with_fixed_manifest():
    partition = load_retail_task_partition(
        "vendor/tau2-bench/data/tau2/domains/retail/split_tasks.json"
    )

    assert partition.train == (
        "0", "1", "2", "3", "4", "6", "7", "8", "10", "11",
        "13", "14", "15", "16", "19", "20", "21", "22", "23", "24",
        "25", "28", "29", "30", "31", "34", "35", "37", "41", "43",
        "44", "46", "47", "48", "50", "52", "54", "57", "58", "59",
        "63", "66", "67", "69", "72", "73", "75", "76", "78", "80",
        "81", "82", "83", "84", "85", "87", "88", "89", "91", "92",
    )
    assert len(partition.validation) == 14
    assert len(partition.final_test) == 30
    assert len(partition.reserve) == 10
