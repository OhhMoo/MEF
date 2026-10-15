# ELF data and encoder study

Study completed 10 October 2026 before EMF data implementation. Upstream source remains unchanged.

## Source versions and entry points

- Official JAX reference: `0_external_repo/ELF`, commit `5098bf28b5e9b52c329970a7e4e1cc28251c76e6`.
- Official GPU port: `0_external_repo/ELF_pytorch`, branch `pytorch_elf`, commit `b29d8833609e9ab7f67cd9da39435ac5cea04837`.
- Read both READMEs, both `utils/data_utils.py` and `utils/encoder_utils.py`, both T5 encoder modules, training entry points/steps, configuration examples, checkpoint utilities, and the PyTorch sampling/final-decoding path.

The JAX entry point is `src/train.py --config ...`; the PyTorch launch script calls the corresponding GPU entry point. The latter is the practical reference for this L40S cluster. Text model checkpoints and T5 weights are not molecular representations and will not be loaded into EMF.

## Actual data-to-loss path

1. `load_dataset` reads pretokenized Hugging Face data. Its loader expects exactly one split per configured dataset path and falls back to local Arrow `save_to_disk` storage. Raw evaluation JSONL is tokenized when loaded. No tokenization occurs in the normal training step.
2. `get_dataloader` truncates/pads to a fixed length. For conditional tasks it concatenates condition tokens and target tokens. Lengths, rather than token equality, establish valid positions. The collator returns token IDs, a pairwise encoder attention mask, a valid-position mask, and a condition-position mask.
3. `build_self_attn_cond_masks` allows condition tokens to attend only to conditions while target positions see all valid tokens. Label dropping modifies this encoder mask *before* embedding, preventing conditions from leaking into unconditional targets.
4. `encode_text` returns the last frozen T5 hidden state and applies `(latent - latent_mean) / latent_std`. The PyTorch function uses `torch.no_grad`, eval mode, and CUDA BF16 only when appropriate. The JAX function is JIT compiled and uses fixed separate encoder parameters.
5. `train_step` interpolates normalized embeddings and Gaussian noise. Targets and model output are converted to velocity with the same endpoint denominator. Only generated positions contribute to loss; PAD positions are excluded unless the task explicitly treats EOS-padded tails as targets.
6. Terminal categorical training uses a separate corrupted-latent branch at time one. The Transformer trunk is shared with transport; mode tokens signal decoder mode. The final category head is read during terminal decoding, not during continuous integration.
7. During generation the implementation integrates continuous latents and performs a separate final decoder-mode forward. Its requested integration count does not include this extra terminal forward, and optional guidance/self-conditioning add further forwards. EMF will report its total actual model evaluations explicitly.

### The important JAX/PyTorch difference

JAX chooses one decoder/denoiser branch for the entire device mini-batch with `jax.lax.cond`. PyTorch draws the branch independently for each example, evaluates a mixed batch, masks each loss by branch membership, and divides their combined sum by all live positions. Its expected objective is `decoder_prob * CE + (1 - decoder_prob) * L2`, not a sum of two independently normalized branch losses. Reported branch metrics are normalized separately.

The current EMF exploration plan explicitly uses `L_flow + L_end`, trains both branches each update, and keeps distinct graph encoder/flow/decoder modules. That is a deliberate graph adaptation, not a claim of numerical parity with ELF's branch mixture or shared trunk.

## Encoder/checkpoint details relevant to reproducibility

JAX builds its own T5 architecture and loads encoder weights from a separate trusted pickle; model checkpoints store flow/trunk weights, EMA, optimizer, step, epoch, and PRNG state. PyTorch calls `transformers.T5EncoderModel.from_pretrained`; its `encoder_checkpoint` configuration field is logged but not used to load separate encoder weights in the normal entry point. T5 is excluded from the trainable optimizer.

The PyTorch checkpoint stores parameters, EMA, optimizer, scheduler, step, epoch, a CPU generator, and optional accumulation buffers. Ordinary CUDA noise/time draws use the global CUDA RNG, which this checkpoint does not save. EMF checkpoints should save all RNG states needed for interrupted operational runs, as well as the molecular split manifest and frozen latent-normalization statistics. Resume errors should be visible rather than silently starting over.

## What carries over to molecular graphs

Retain frozen deterministic targets after representation warm-up, explicit masks, normalized continuous states, destination prediction, safe endpoint handling, terminal decoding, EMA/checkpoint patterns, and a separate sampling configuration.

Replace T5 tokenization/contextual embeddings with a molecular node-and-unordered-pair encoder. Replace vocabulary unembedding with atom and bond heads. Use active node/pair coordinates throughout Gaussian noise, normalization, transport, and losses. Padding is storage only: it must neither alter graph identity nor introduce an extra generated atom category. Pair ordering is fixed at `i < j`; mirror only when constructing adjacency. A single chemistry codec owns both training graph creation and terminal graph validation.

The first data implementation depends only on Python, NumPy, PyTorch, and RDKit. It does not need Hugging Face datasets, T5 weights, PyG, 3D coordinates, or quantum properties.

## QM9 acquisition and processing choice

Use the authors' [original Figshare collection](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904), not the unrelated revised QM9 release. The original XYZ files contain author-provided GDB9 SMILES; those provide the 2D graph without inferring bonds from geometry.

| Artifact | Official file URL | Size | Published MD5 |
|---|---|---:|---|
| Molecular archive | `https://ndownloader.figshare.com/files/3195389` | 86,144,227 bytes | `ad1ebd51ee7f5b3a6e32e974e5d54012` |
| Format README | `https://ndownloader.figshare.com/files/3195392` | 5,198 bytes | `c6581a03f673746528c57acfc6b79679` |
| Geometry exclusions | `https://ndownloader.figshare.com/files/3195404` | 486,752 bytes | `a361887bacb427b8a0ce7903d92a53b4` |

Metadata was verified using Figshare API articles `1057646`, `1057641`, and `1057644`. Historical article URLs swapped dataset titles; verified file ID and archive name remove that ambiguity. No molecular archive was downloaded during this study.

The README specifies atom count on line one, one-based molecular ID on line two, and GDB9/relaxed SMILES on line `n_atoms + 4`. Parse the first SMILES. Exclude the official 3,054 inconsistent IDs. Read tar members directly without extraction. Retain source IDs and observed content hashes in the preparation manifest.

Processing after build authorization:

1. Parse/sanitize with RDKit, remove explicit hydrogen atoms and stereochemical labels, retain connected neutral C/N/O/F molecules with one to nine heavy atoms, and reject radicals/unrepresentable chemistry.
2. Canonicalize the retained 2D identity, deduplicate identities before splitting, and Kekulize with aromatic flags cleared to obtain no-bond/single/double/triple labels. Choose atom order by canonical SMILES parsing.
3. Make a seeded identity-disjoint train/validation/test split; cap the initial training subset only after split creation. Save the exact selected identities and split recipe. This is an exploratory split, not a published QM9 benchmark split.
4. Save NumPy graph tensors and a metadata manifest under `2_data_storage/processed`; preserve raw archives under `2_data_storage/raw`. Estimate the atom-count distribution from the selected training split only.
5. Check a few graph/SMILES round trips, padding and pair masks, split disjointness, and count prior behavior. No reconstruction accuracy threshold or causal-control suite blocks first generation.

## Integration proposal

Public chemistry functions should convert SMILES to fixed-size categorical graph arrays and decode arrays into a structured outcome containing canonical identity, chemical validity, connectivity, and an explicit failure reason. Disconnected sanitized graphs are distinguishable from valence failures and are preserved in the attempt ledger.

Public data functions should prepare/load the processed split, collate one batch against the coordinator's `GraphBatch`, and expose a training count prior. Keep download/preparation separate from minibatch access. Randomness is explicit; workers do not rewrite shared manifests or install dependencies concurrently.

## Authorized implementation and first operational checks

After the coordinator read the three source studies and fixed `GraphBatch` /
`GraphMasks`, the authorized implementation was written to `emf/data.py` and
`emf/chem.py`. The original archive and both metadata files passed their published
integrity checks. Streaming preparation retained 130,165 standardized identities
from 130,831 characterized records: 580 records had charged atoms, and 86 repeated
an existing identity. The coordinator's 80/10/10 recipe produced a capped training
set of 20,000, validation set of 13,016, and test set of 13,017. Every retained
categorical graph matched its standardized identity when decoded.

Prepared splits and the manifest live in `2_data_storage/processed/qm9`. The
training count prior is saved in the manifest and covers counts one through nine;
83.455% of selected training molecules have nine heavy atoms. The dataset exposes
`.smiles`, `.ids`, and `.counts`; `collate_graphs` builds the shared tensor contract.

Seventeen focused CPU checks passed, covering aromatic and single-atom round
trips, restricted chemistry, explicit valence failures, disconnected identities,
padding, split deduplication, and the selected-training count prior. A real
128-molecule batch has atom labels `[128,9]`, packed pair labels `[128,36]`, correct
active masks, and 128 exact codec round trips. Reports are in
`5_test/results/data_builder.xml` and `data_preparation_check.json`. These checks
validate the data path; they do not impose a model reconstruction threshold.
