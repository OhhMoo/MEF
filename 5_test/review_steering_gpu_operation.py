"""Read-only independent audit of the allocated steering smoke run."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen,rdMolDescriptors
from emf.redesign_evaluation import load_committed_proposals
from emf.steering_data import SteeringDataset

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'4_experimental_result/result/evidence_steering_001'
OP=STUDY/'operation'
EVAL=OP/'evaluation_validation'
report=json.loads((OP/'task_complete.json').read_text())
assert report['step']==80 and report['frozen_representation_unchanged']
assert all(np.isfinite(value) for value in report['metrics'].values())
payload=torch.load(OP/'checkpoints/source_final.pt',map_location='cpu',weights_only=False)
initial=torch.load(ROOT/payload['config']['initializer'],map_location='cpu',weights_only=False)
frozen_names=[name for name in payload['model'] if name.startswith(('encoder.','normalizer.'))]
assert frozen_names
assert all(torch.equal(payload['model'][name],initial['model'][name]) for name in frozen_names)
protocol=json.loads((EVAL/'protocol.json').read_text())
assert hashlib.sha256((OP/'checkpoints/source_final.pt').read_bytes()).hexdigest()==protocol['generator_checkpoint_sha256']
dataset=SteeringDataset(ROOT/'2_data_storage/processed/evidence_steering_toy','validation')
private=json.loads((dataset.data_dir/'private/oracle.json').read_text())
metrics=list(csv.DictReader((EVAL/'metrics.csv').open()))
rows=0
truth_rows=0
invalid=0
paired_bytes=0
for case in protocol['cases']:
    campaign=case['campaign']
    name=dataset.campaign_ids[campaign]
    lead=Chem.MolFromSmiles(private['campaign_leads'][name])
    lead_features=np.array([Crippen.MolLogP(lead),rdMolDescriptors.CalcTPSA(lead)])
    for condition in ('correct','opposite'):
        path=EVAL/f"pair_{case['pair_number']:03d}"/f'campaign_{campaign:03d}'/'support_004'/condition
        ledger,ranking,commit=load_committed_proposals(path)
        assert len(ledger)==16
        assert commit['oracle_access_before_commit'] is False
        assert all(row['nfe']==8 and row['atom_count']==9 for row in ledger)
        assert [row['attempt_id'] for row in ledger]==list(range(16))
        nested=dataset.nested_support(campaign,case['lead_index'],
            seed=dataset.paired_support_seed(name),sizes=(4,16,64))
        assert len(set(nested['reserve_smiles']))==64
        assert np.array_equal(nested['lead_values'],np.zeros(2))
        for row in ledger:
            assert row['reserved_support_match']==bool(row['smiles'] in nested['reserve_smiles'])
        if condition=='opposite':
            partner=EVAL/f"pair_{case['pair_number']:03d}"/f"campaign_{case['opposite']:03d}"/'support_004'/'correct'
            assert (path/'attempts.jsonl').read_bytes()==(partner/'attempts.jsonl').read_bytes()
            assert (path/'ranking.json').read_bytes()==(partner/'ranking.json').read_bytes()
            paired_bytes+=1
        outcomes=[json.loads(line) for line in (path/'evaluation_outcomes.jsonl').read_text().splitlines()]
        for row in outcomes:
            mol=Chem.MolFromSmiles(row['smiles'])
            values=np.array([Crippen.MolLogP(mol),rdMolDescriptors.CalcTPSA(mol)])
            delta=(values-lead_features)/np.array(private['scales'])
            expected=np.array([private['coefficients'][name]*delta[0],-delta[1]])
            np.testing.assert_allclose(row['outcomes'],expected,atol=1e-12,rtol=1e-12)
            assert row['profile_success']==bool(expected[0]>=.25 and expected[1]>=-.1)
            truth_rows+=1
        truth={row['smiles']:row for row in outcomes}
        distinct_hit=sum(bool(row['chemically_valid'] and row['connected'] and row['supported_chemistry']
            and not row['duplicate'] and row['within_neighbourhood'] and not row['reserved_support_match']
            and not row['lead_match'] and truth.get(row['smiles'],{}).get('profile_success')) for row in ledger)
        measurement=next(row for row in metrics if row['campaign']==name and row['condition']==condition)
        assert float(measurement['local_profile_yield_per_attempt'])==distinct_hit/16
        assert int(measurement['physical_attempts'])==(16 if condition=='correct' else 0)
        rows+=len(ledger)
        invalid+=sum(not(row['chemically_valid'] and row['connected']) for row in ledger)
assert rows==256 and paired_bytes==8
assert sum(int(row['physical_attempts']) for row in metrics)==128
assert sum(int(row['physical_oracle_calls']) for row in metrics)==38
result={'reviewed_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
 'slurm_job_id':'375128','source_updates':80,'source_training_seconds':report['training_seconds'],
 'logical_attempts_audited':rows,'physical_generation_attempts':128,'invalid_logical_attempts_retained':invalid,
 'exact_replayed_batches':paired_bytes,'external_outcome_rows_independently_recomputed':truth_rows,
 'physical_oracle_calls':38,'frozen_tensors_identical_to_initializer':len(frozen_names),
 'checkpoint_hash_unchanged_after_evaluation':True,'maximum_reserved_support':64,
 'nfe':8,'actual_allocation_seconds':12,'allocation_gpu_hours':12/3600,
 'notes':['The smoke run checks operation, not trained steering performance.',
          'Withheld response diagnostics use the shared r8 scorer.']}
(ROOT/'5_test/results/steering_gpu_operation_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
