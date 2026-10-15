# ELF sampling and evaluation study

Read before implementing EMF, 10 October 2026. Reference revisions: JAX/main `5098bf28b5e9b52c329970a7e4e1cc28251c76e6`; official PyTorch branch `b29d8833609e9ab7f67cd9da39435ac5cea04837`. Upstream files remain unchanged.

## The complete upstream path

1. **Frozen contextual targets.** Both `src/utils/encoder_utils.py::encode_text` implementations return normalized T5 states. PyTorch applies `torch.no_grad()` and encoder evaluation mode; `src/train.py::run_training` explicitly freezes encoder parameters. Normalization uses configured scalar mean/std, rather than fitted molecular node/pair statistics.
2. **Continuous interpolation.** `src/utils/sampling_utils.py::add_noise` forms $z_t=tx+(1-t)s\varepsilon$, where $s$ is `denoiser_noise_scale`. Training `sample_timesteps` supports uniform and logit-normal draws on $(0,1)$. Conditioning positions are restored to their clean states.
3. **Destination prediction.** `src/modules/model.py::ELF.forward` returns the continuous destination prediction and, when requested, categorical logits. `src/utils/sampling_utils.py::net_out_to_v_x` discards logits during transport and computes

   $$
   v(z,t)=\frac{\widehat x-z}{\max(1-t,t_{\mathrm{eps}})}.
   $$

   Thus transport never chooses a category. Prefix time tokens, model-mode tokens, and optional self-conditioning guide a shared Transformer trunk. PyTorch computes the destination and decoder output heads in float32 even when the trunk uses bf16.
4. **Euler or stochastic rollout.** `_ode_step` advances $z\leftarrow z+(t_{next}-t)v$. `_sde_step` first perturbs the state backwards with $\alpha=\operatorname{clip}(1-\gamma\Delta t,0,1)$, $t_{back}=\alpha t$, $z_{back}=\alpha z+(1-\alpha)s\varepsilon$, then takes an Euler step to the next time. This particular stochastic sampler is optional, rather than required by continuous transport. `_generate_samples_single_batch` in each `generation_utils.py` integrates every interval and always uses ODE for the last interval.
5. **One final categorical choice.** `generation_utils.py::_dlm_decode_batch` sends terminal $z$ through the shared trunk in decoder mode at $t=1$, discards its continuous head, and argmaxes decoder logits. `mask_after_eos` converts text after the first EOS to padding. Conditional runs also shift away the preserved prompt. Neither operation maps directly to molecular graph handling.
6. **Checkpoint and evaluation.** JAX `generation_utils.py::_setup_generation` selects `ema_params1`; PyTorch `generation.py::_build_eval_model` makes an evaluation copy and loads EMA parameters. The `eval.py` entry points reconstruct model architecture and an optimizer/state template, restore the model/EMA state, set seed-specific generators, and generate. Training calls `run_generation` at configured epoch intervals and saves checkpoints independently.

The unconditional configuration uses 32/64-step SDE, logit-normal integration grids, and self-conditioning guidance scale 3. Conditional generation uses a 64-step ODE and classifier-free guidance. These are language settings to understand, rather than molecular defaults to inherit unchanged.

## Decoder training and branch differences

In JAX `src/train_step.py::train_step`, one Bernoulli choice selects an entire minibatch: decoder CE or velocity L2. In PyTorch the same function chooses modes per example, makes one mixed forward, and masks CE/L2 to their selected rows. The expected objective has weights `decoder_prob` and `1-decoder_prob`, rather than an unweighted sum.

Decoder training corrupts each token with a random logit-normal mixing coefficient and a separate noise scale, then labels its model time as $t=1$. This is not the EMF plan's additive $X+\sigma\xi$ corruption. Both upstream branches share the Transformer trunk between denoising and decoding. EMF's initial small model has distinct $F_\theta$ and $q_\phi$ functions so their roles remain explicit.

When $1-t<t_{\mathrm{eps}}$, upstream clamps both predicted and target velocity denominators. That no longer equals the constant straight-line derivative $X-s\varepsilon$. EMF avoids this discrepancy by training/integrating only to $T<1$ and using the exact denominator there.

## What transfers into the first EMF sampler

The implementation should adapt ELF's explicit separation between continuous rollout and terminal categorical decoding, its destination-to-velocity transformation, evaluation mode, explicit randomness, checkpoint/config association, and per-run sample ledger.

The initial EMF sampler follows the research equation directly:

$$
z_0\sim\mathcal N(0,I),\qquad
z_{j+1}=z_j+\Delta t\frac{F_\theta(z_j,t_j)-z_j}{1-t_j},\qquad
\widehat G=D_\phi(F_\theta(z(T),T)).
$$

Use $K-1$ uniformly spaced Euler intervals on $[0,T]$ and one endpoint prediction, for exactly $K$ calls to the flow model per attempted molecule. One decoder call then chooses atom and bond categories. Atom counts are drawn from the training prior and stay fixed. Noise, predictions, and updates remain zero outside active nodes and unordered pairs; chemistry conversion owns symmetric graph construction.

Upstream `num_sampling_steps=N` means $N$ integration calls **plus** a separate decoder-mode trunk call. Guidance may require additional trunk calls per interval. EMF uses an explicitly named total flow-evaluation budget and leaves guidance/SDE/self-conditioning for exploration after the basic path works.

## Timing and attempt accounting

JAX unconditional generation calls `block_until_ready()` around model timing. The PyTorch generation path uses `time.time()` without CUDA synchronization, and both reported neural timings omit host tokenizer conversion. EMF must synchronize CUDA at timing boundaries and measure the whole count/noise → flow → endpoint → decode → host chemistry path. File writing can be separately recorded. Batch timing amortized over its attempts is throughput accounting, not individual latency.

The upstream text evaluator uses GPT-2 perplexity, unigram entropy, BLEU, and ROUGE. Its PPL routine skips empty generated texts. EMF replaces those task-specific metrics with chemistry sanitization and connectivity, unique connected valid identities, training-set novelty, failure counts, and complete-pipeline throughput. Every requested attempt remains in the denominator and JSONL ledger; disconnected sanitized graphs remain visible separately. No retry, repair, or filtering produces replacement attempts.

The evaluator will write an attempt JSONL/CSV and a run summary JSON/CSV under `4_experimental_result/result/<run>/`. It records seed, atom count, flow evaluations, canonical identity or failure reason, validity, connectivity, duplicate/novel flags, timing, and run metadata. Novelty is a set comparison against standardized training identities. Training checkpoint provenance and fitted normalization/count prior must travel together.

## Practical checks after contracts are agreed

- A constant destination oracle yields its destination at the endpoint under the declared Euler schedule, with exactly $K$ flow calls and one decoder call.
- Inactive latent coordinates remain zero and pair ordering never changes during integration.
- A fixed generator produces repeatable count/noise draws and outputs without resetting process-wide random state.
- Mixed valid, duplicate, disconnected, and invalid attempts produce correct denominators and round-trip through the ledger files.
- Full integration uses a trained checkpoint, checks that every requested attempt is retained, and records synchronized device timing.

This study completes the sampling/evaluation read-through. Implementation waits for the coordinator's shared tensor and chemistry contracts.
