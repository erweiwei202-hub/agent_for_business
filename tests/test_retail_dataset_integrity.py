import json
from pathlib import Path


def test_retail_golden_product_actions_reference_existing_products():
    data_dir = (
        Path(__file__).parents[1]
        / "vendor"
        / "tau2-bench"
        / "data"
        / "tau2"
        / "domains"
        / "retail"
    )
    tasks = json.loads((data_dir / "tasks.json").read_text(encoding="utf-8"))
    database = json.loads((data_dir / "db.json").read_text(encoding="utf-8"))
    product_ids = set(database["products"])

    invalid_actions = []
    for task in tasks:
        for action in (task.get("evaluation_criteria") or {}).get("actions") or []:
            if action.get("name") != "get_product_details":
                continue
            product_id = (action.get("arguments") or {}).get("product_id")
            if product_id not in product_ids:
                invalid_actions.append((task["id"], action.get("action_id"), product_id))

    assert invalid_actions == []
