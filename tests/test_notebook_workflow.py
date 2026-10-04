"""Exercise migrated execution switches with no models or paid calls."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rag_baselines.data import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def staged_cells():
    notebook = read_json(ROOT/'notebook/05_measure_abcd.ipynb')
    return ["".join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code'][1:5]


@pytest.mark.parametrize('failure', [False, True])
def test_staged_steps_run_in_order_and_publish_only_completed_evaluation(tmp_path, failure):
    config = read_json(ROOT/'config/validation_study.json')
    write_json(tmp_path/'config/validation_study.json', config)
    pointer = tmp_path/'results/current_staged_run.txt'
    pointer.parent.mkdir(parents=True)
    pointer.write_text('previous-complete-study', encoding='utf-8')
    events = []

    def replay(source, output):
        events.append('replay')
        write_json(output/'d_input_replay_summary.json', {'accepted': 2, 'decisions': 2, 'failures': []})

    def check(source, output, device):
        assert device == 'cpu'
        events.append('decisions')

    def c_development(root, output, study):
        events.append('C_dev')
        return [], {'selected_arm': 'C_top3'}

    def compare(root, output, study, *, split, include_c, d_candidate=False):
        events.append('eval' if split == 'eval' else ('D_candidate' if d_candidate else 'ABD_dev'))
        if failure and split == 'eval':
            raise RuntimeError('simulated failed evaluation')
        return []

    def freeze(root, output):
        events.append('freeze_D')
        return [], {'selected_arm': 'D_candidate'}

    import scripts.validate_laya_replay as replay_module
    import scripts.check_laya_decisions as checks_module
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(replay_module, 'validate_replay', replay)
        patch.setattr(checks_module, 'run_checks', check)
        namespace = dict(ROOT=tmp_path, read_json=read_json, write_json=write_json,
            os=SimpleNamespace(environ={'OPENAI_API_KEY': 'fixture', 'TYPESAFE_API_KEY': 'fixture'}),
            selected_questions=lambda root, split, count: [{'id': str(i)} for i in range(count*2)],
            datetime=datetime, timezone=timezone, uuid4=uuid4,
            notebook_source_sha256=lambda path: 'fixture-source-hash',
            display=lambda obj: None, run_c_development=c_development,
            run_comparison=compare, freeze_d_policy=freeze)
        for source in staged_cells():
            if failure and 'freeze_d_policy(ROOT' in source:
                with pytest.raises(RuntimeError, match='failed evaluation'):
                    exec(compile(source, '<staged-notebook>', 'exec'), namespace)
            else:
                exec(compile(source, '<staged-notebook>', 'exec'), namespace)
    assert events == ['replay', 'decisions', 'C_dev', 'ABD_dev', 'D_candidate', 'freeze_D', 'eval']
    expected = 'previous-complete-study' if failure else str(namespace['OUT'])
    assert pointer.read_text(encoding='utf-8') == expected


def test_disabled_staged_execution_does_not_create_results_or_read_api_keys(tmp_path):
    write_json(tmp_path/'config/validation_study.json', read_json(ROOT/'config/validation_study.json'))
    namespace = dict(ROOT=tmp_path, read_json=read_json)
    for source in staged_cells():
        source = source.replace('RUN_STUDY = True', 'RUN_STUDY = False')
        exec(compile(source, '<disabled-staged-notebook>', 'exec'), namespace)
    assert namespace['OUT'] is None
    assert not (tmp_path/'results').exists()
