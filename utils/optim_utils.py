from pathlib import Path


def log_execution_time(dataset_name, elapsed_seconds, path=None):

    if path is None:
        file_path = Path(__file__).resolve().parent.parent / 'outputs' / 'optim_exec_times.txt'
    else:
        file_path = path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a", encoding='utf-8') as f:
        f.write(f"{dataset_name} = {elapsed_seconds:.2f}\n")