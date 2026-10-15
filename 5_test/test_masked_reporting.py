"""Repeated refresh must read primary artifacts, never previous derived output."""
import importlib.util
import json
from pathlib import Path

def test_live_report_does_not_recursively_embed_itself(tmp_path):
    path=Path(__file__).resolve().parents[1]/'3_model_training/scripts/masked_edit_report.py'
    spec=importlib.util.spec_from_file_location('masked_reporting',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'live_report.json').write_text(json.dumps({'stage':'old_report','payload':{'stage':'previous'}}))
    folder=tmp_path/'control_validation'/'seed142';folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps({'phase':'complete','stage':'control_validation','seed':142}))
    first=module.refresh(tmp_path);size=(tmp_path/'live_report.json').stat().st_size
    second=module.refresh(tmp_path)
    assert first['summaries']==second['summaries']
    assert len(second['summaries'])==1
    assert all(r['path']!='live_report.json' for r in second['summaries'])
    assert (tmp_path/'live_report.json').stat().st_size<=size+100
