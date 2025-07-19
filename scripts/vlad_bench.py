"""Convert VLADBench dataset to format for qwen-vl-finetune"""

import argparse
from pathlib import Path
import json

COUNTRIES = {
    "Amercia": "America",
    "America": "America",
    "Canada": "Canada",
    "China": "China",
    "Germany": "Germany",
    "Japan": "Japan",
    "LingoQA": None,
    "SODA": "China",
}


def load_task(input_path: Path, task_key: str) -> list[dict]:
    """Load a single task from the input dataset."""
    task_path = input_path / task_key
    jobs_path = Path(f"{task_path}_E.json")
    if not jobs_path.exists():
        print(f"Missing task: {jobs_path}")
        return []
    jobs = json.load(jobs_path.open())
    annotations: list[dict] = []
    for job in jobs:
        country = job["country"]
        assert country in COUNTRIES, (task_key, job["id"], country)
        country = COUNTRIES[job["country"]]
        questions: list[str] = job["questions"]
        answers: list[str] = job["reference"]
        assert len(questions) == len(answers)
        for question, answer in zip(questions, answers):
            image_key, _, question = question.partition(";")
            if image_key.startswith("["):
                # Video not supported yet
                continue
            image_path = task_path / image_key
            assert image_path.exists(), (task_key, job["id"], image_path)
            if country is not None:
                question = f"The image is from {country}. {question}"
            annotations.append(
                dict(
                    image=f"{task_key}/{image_key}",
                    conversations=[
                        {
                            "from": "human",
                            "value": f"<image>\n{question}",
                        },
                        {
                            "from": "gpt",
                            "value": str(answer),
                        },
                    ],
                )
            )
    return annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input dataset directory.",
    )
    args = parser.parse_args()
    input_path: Path = args.input

    all_tasks: dict[str, dict[str, list[str]]] = json.load((input_path / "all_tasks.json").open())
    annotations: list[dict] = []
    for k1, t1 in all_tasks.items():
        for k2, t2 in t1.items():
            for k3 in t2:
                task_key = "/".join([k1, k2, k3])
                annotations.extend(load_task(input_path, task_key))
    json.dump(annotations, (input_path / "annotations.json").open("w"), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
