import wandb


def sync():
    api = wandb.Api()
    # Add all projects you want to check against
    project_paths = ["BayesRL/math_evals", "CodeShield/math_evals"]

    all_run_names = set()

    for path in project_paths:
        print(f"Fetching existing runs from {path}...")
        try:
            runs = api.runs(path)
            for run in runs:
                all_run_names.add(run.name)
        except Exception as e:
            print(f"Error accessing {path}: {e}")

    with open("completed_runs.txt", "w") as f:
        for name in sorted(list(all_run_names)):
            f.write(f"{name}\n")

    print(f"Sync complete. {len(all_run_names)} runs cached.")


if __name__ == "__main__":
    sync()
