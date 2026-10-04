"""Git clean filter: retain notebook sources while keeping outputs local.

Install the local filter with the git config commands in README.md.
Uses only the Python standard library; malformed notebook input fails closed.
"""

import json
import sys


def clean_notebook(notebook):
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    notebook.get("metadata", {}).pop("widgets", None)
    return notebook


if __name__ == "__main__":
    notebook = json.loads(sys.stdin.buffer.read().decode("utf-8-sig"))
    result = json.dumps(clean_notebook(notebook), ensure_ascii=False, indent=1) + "\n"
    sys.stdout.buffer.write(result.encode("utf-8"))
