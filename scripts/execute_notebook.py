"""Execute the saved-results notebook; never performs paid API calls."""

import json
import sys
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

# Keep the kernel environment project-local and match this Python interpreter.
kernels = Path(".cache/jupyter/kernels/sms-spam")
kernels.mkdir(parents=True, exist_ok=True)
(kernels / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "SMS spam classifier",
            "language": "python",
        }
    )
)
manager = KernelSpecManager(kernel_dirs=[str(kernels.parent.resolve())])
from jupyter_client import KernelManager

km = KernelManager(kernel_name="sms-spam", kernel_spec_manager=manager)
path = Path("task_1_modelling/analysis.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook, km=km, timeout=180, resources={"metadata": {"path": str(Path.cwd())}}
).execute()
nbformat.write(notebook, path)
print("Executed", path)
