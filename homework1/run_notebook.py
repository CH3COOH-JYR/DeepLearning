import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description="执行作业、验证所有单元并保存输出和图表。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="另存执行结果；默认更新原 Notebook。")
    parser.add_argument("--timeout", type=int, default=600, help="每个代码单元的超时秒数。")
    args = parser.parse_args()
    notebook_path = Path(__file__).resolve().with_name("homework.ipynb")
    output_path = args.output.resolve() if args.output else notebook_path
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    def report_cell(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"执行代码单元 {cell_index + 1}/{len(notebook.cells)}", flush=True)

    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="homework1-") as runtime_directory:
        environment = os.environ.copy()
        environment.update({
            "HOMEWORK_DEVICE": args.device,
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "MPLCONFIGDIR": str(Path(runtime_directory) / "matplotlib"),
            "IPYTHONDIR": str(Path(runtime_directory) / "ipython"),
        })
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(notebook_path.parent)}},
            on_cell_start=report_cell,
        )
        kernel_manager = client.create_kernel_manager()
        kernel_manager.connection_file = str(Path(runtime_directory) / "kernel.json")
        kernel_manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        client.execute(env=environment)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError("存在未执行的代码单元。")
    errors = [output for cell in code_cells for output in cell.outputs if output.output_type == "error"]
    if errors:
        raise RuntimeError(f"Notebook 包含 {len(errors)} 个执行错误。")
    figure_count = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells for output in cell.outputs
    )
    if figure_count < 6:
        raise RuntimeError(f"预期至少 6 个图表输出，实际 {figure_count} 个。")
    nbformat.validate(notebook)
    nbformat.write(notebook, output_path)
    elapsed = time.perf_counter() - started_at
    print(f"通过：{len(code_cells)} 个代码单元、{figure_count} 个图表，无执行错误；耗时 {elapsed:.1f} 秒。")
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()
