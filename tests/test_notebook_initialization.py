"""Offline checks of notebook initialization against the real raw logger."""
import ast
import json
from pathlib import Path

from rag_baselines.data import read_jsonl, write_json
from rag_baselines.raw_logging import RawLog


ROOT = Path(__file__).resolve().parents[1]


def test_abcd_shared_tree_preparation_records_one_scope_without_api_calls(tmp_path):
    notebook = json.loads((ROOT/'notebook/05_measure_abcd.ipynb').read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'shared_by_methods' in ''.join(cell['source']))
    scopes = [node for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.With) and any(
                  isinstance(item.context_expr, ast.Call) and any(
                      keyword.arg == 'phase' and isinstance(keyword.value, ast.Constant)
                      and keyword.value.value == 'preparation'
                      for keyword in item.context_expr.keywords)
                  for item in node.items)]
    assert len(scopes) == 1
    tree_directory = tmp_path/'tree'
    for name in ('tree.json', 'manifest.json', 'checkpoint.json', 'checkpoint_hash.json'):
        write_json(tree_directory/name, {'fixture': name})
    log = RawLog(tmp_path, manifest={'fixture': True}, prices={})
    preparation = {'directory': str(tree_directory), 'cache_hit': True}
    block = ast.Module(body=scopes, type_ignores=[])
    exec(compile(block, '05_measure_abcd.ipynb:tree_preparation', 'exec'),
         {'RAW_LOG': log, 'Path': Path, 'preparation': preparation})

    index = read_jsonl(log.directory/'events.jsonl')
    reused = next(row for row in index if row['stage'] == 'tree' and row['kind'] == 'reused')
    events = read_jsonl(log.directory/reused['events_path'])
    tree_event = next(row for row in events if row['stage'] == 'tree' and row['kind'] == 'reused')
    assert tree_event['method'] == 'shared'
    assert tree_event['payload']['shared_by_methods'] == ['C', 'D']
    assert tree_event['payload']['actual_new_api_calls'] == 0
    assert sum(row['kind'] == 'started' for row in index) == 1
    assert any(row['kind'] == 'completed' for row in events)
    assert len(list((log.directory/'sources').iterdir())) == 4
    assert read_jsonl(log.directory/'api_calls.jsonl') == []
