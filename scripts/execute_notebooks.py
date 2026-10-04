"""Execute existing ipynb files top-to-bottom; measurements remain in their cells."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    names = sys.argv[1:] or [p.name for p in sorted((ROOT / "notebook").glob("*.ipynb"))]
    for name in names:
        path = ROOT / "notebook" / name
        notebook = nbformat.read(path, as_version=4)
        print(f'Executing {name}', flush=True)
        client = NotebookClient(notebook, timeout=1800, kernel_name="decision-llm",
                                resources={"metadata": {"path": str(ROOT / "notebook")}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        print(f'Saved executed {name}', flush=True)
