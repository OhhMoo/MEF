# ELF model and training study

Reviewed 10 October 2026 before EMF implementation. JAX reference commit: `5098bf28b5e9b52c329970a7e4e1cc28251c76e6`; official PyTorch branch commit: `b29d8833609e9ab7f67cd9da39435ac5cea04837`.

## What the upstream model actually learns

ELF uses a frozen contextual text encoder and a trainable Transformer with two output heads. The Transformer receives a noisy continuous embedding and predicts its **clean destination**, not velocity directly. A second head maps the same Transformer hidden states to token logits. Its denoising and decoding modes therefore share the trunk.

The main source references, relative to `0_external_repo/ELF_pytorch/`, are:

| Source | Behavior |
|---|---|
| `src/modules/model.py:18`, `ELFBlock` | Pre-RMSNorm attention, residual connection, pre-RMSNorm SwiGLU, residual connection. |
| `src/modules/model.py:52`, `ELF` | Bottleneck input projection, learned time-prefix tokens, optional self-conditioning and model-mode tokens, Transformer blocks, clean-embedding head and factored categorical head. |
| `src/modules/model.py:159`, `ELF.forward` | Optionally projects concatenated noisy/self-conditioned embeddings; prepends condition tokens; executes the shared trunk; removes prefix positions; returns `(clean_prediction, decoder_logits)`. |
| `src/modules/layers.py:99`, `RMSNorm` | Computes the mean square in float32 before rescaling in input precision. |
| `src/modules/layers.py:127`, `TimestepEmbedder` | Sinusoidal scalar embedding followed by two small normally initialized linear layers and SiLU. |
| `src/modules/layers.py:185`, `Attention` | Q/K RMS normalization and PyTorch scaled-dot-product attention; text RoPE is supplied externally. |
| `src/modules/layers.py:224`, `SwiGLUFFN` | Shared gated pointwise feed-forward operation. |
| `src/modules/layers.py:243`, `FinalLayer` | RMSNorm followed by a zero-initialized clean-embedding projection. |
| `src/modules/t5_encoder.py:83`, `get_encoder` | Loads pretrained Hugging Face T5; the training runner freezes it. |
| `src/utils/encoder_utils.py:6`, `encode_text` | No-gradient encoder evaluation followed by `(embedding-mean)/std`. |

The JAX equivalents are `ELFBlock` at `ELF/src/modules/model.py:12`, `ELF` at line 39, and the corresponding classes in `ELF/src/modules/layers.py`. Its T5 implementation contains explicit relative-position attention, RMS normalization, encoder blocks, and frozen-encoder loading through the runner. Neither text positions nor a T5 tokenizer belong in the molecular graph model.

## Noise, parameterization, and endpoint behavior

`src/utils/sampling_utils.py:11`, `add_noise`, constructs

$$
z_t=t x+(1-t)s\varepsilon.
$$

`src/utils/sampling_utils.py:115`, `net_out_to_v_x`, interprets the network output as $\hat x$ and converts it to

$$
v_\theta=\frac{\hat x-z_t}{\max(1-t,\delta)}.
$$

The default clamp is $\delta=0.05$. The training target uses the same denominator (`src/train_step.py:120`), and the L2 loss averages velocity error across latent channels (`src/train_step.py:236`). For $t\leq 1-\delta$, this is exactly clean-prediction error weighted by $(1-t)^{-2}$. For later times, the clamp changes that weighting and the target is no longer the unclamped interpolation derivative. This distinction should be explicit in EMF.

The OpenWebText configuration uses logit-normal training times with mean $-1.5$, standard deviation $0.8$, denoising noise scale $2$, normalized T5 latents with standard deviation $0.2$, and a $0.05$ denominator clamp. These text-specific settings are upstream facts, not molecular defaults.

Decoder examples use a different corruption (`src/train_step.py:111`): per-token $a=\operatorname{sigmoid}(\mathcal N(\mu_d,\sigma_d^2))$, input $a x+(1-a)s_d\xi$, and model time fixed at one. Categorical supervision operates on these decoder examples; it does not discretize denoising trajectories. The OpenWebText default uses decoder probability $0.2$, decoder corruption mean $0.8$, standard deviation $0.8$, and noise scale $5$.

EMF's agreed first exploration instead uses $t\sim U(0,T)$, $T=0.95$, unit Gaussian interpolation noise, and decoder corruption $X+\sigma\xi$. Restricting time below one makes the mathematical velocity exact throughout the training/sampling domain and avoids an endpoint singularity. Sampling must finish with one clean prediction at $T$ before graph decoding.

## The complete training process

`src/train.py:84`, `run_training`, seeds random generators, loads the tokenizer/data, loads and freezes T5, creates the ELF model, derives batch sizes and optimizer-step counts, constructs optimizer/scheduler/EMA state, optionally resumes a checkpoint, and executes `train_step` on minibatches. It saves checkpoints and periodically runs generation. CUDA execution uses bf16 autocast inside the training step while input/output projections and losses use float32. The current CUDA runner calls `torch.compile` unconditionally; the config's `use_compile` flag controls evaluation elsewhere rather than that training branch. Our small first model can start with eager execution.

There is a material JAX/PyTorch schedule difference:

- `ELF/src/train_step.py:89` draws one decoder/denoiser Bernoulli for the entire device minibatch and `jax.lax.cond` executes the selected branch.
- `ELF_pytorch/src/train_step.py:104` draws one Bernoulli per example. A single forward receives a mixture of decoder and denoiser inputs, both heads run, and CE/L2 terms are masked to their corresponding rows.
- The PyTorch combined denominator is the active-token count across both branches (`src/train_step.py:251`); its expected objective is $p_d\,\mathrm{CE}+(1-p_d)\,\mathrm{L2}$. This differs from EMF's equal-weight sum of flow and terminal-decoder losses on each minibatch.

Self-conditioning uses a detached preliminary clean prediction, retains it for approximately half the denoiser examples, and supplies zero self-conditioning in decoder mode. Optional self-conditioning guidance adds a detached velocity correction derived from two model evaluations (`src/train_step.py:175`). Conditional text generation additionally drops conditions before encoding and supports classifier-free guidance. These features add complexity and evaluations; the initial unconditional molecular generator will omit them.

`src/utils/train_utils.py` owns the optimizer, linear-warmup/constant-or-cosine learning rate, and EMA. Gradient accumulation divides minibatch loss before backward; clipping at norm one, optimizer/scheduler steps, and EMA updates happen only on actual optimizer updates. The upstream learning rate scales with effective batch size, including accumulation. Its default optimizer is Muon. `src/utils/muon_utils.py` implements matrix/other-parameter partitioning, a patched Nesterov-Adam auxiliary update, float32 Newton–Schulz iterations, layout-aware scaling, missing-gradient handling, and distributed padding. The first EMF run can use the upstream-supported AdamW choice and retain clipping; reproducing Muon is unnecessary to establish a small generator.

Checkpointing stores parameters, optimizer/scheduler, EMA, counters, and a dedicated CPU generator state (`src/utils/checkpoint_utils.py:49`). Global Torch/NumPy/CUDA random states are not all saved there. EMF's runner should save the full states it actually uses and avoid silently continuing from scratch when an explicitly requested resume fails.

## Reuse and molecular adaptation

Retain the upstream repositories unchanged. Reuse the actual official PyTorch implementations of `RMSNorm`, `TimestepEmbedder`, `Attention` without RoPE, `SwiGLUFFN`, and `FinalLayer` where they fit the graph model. Their external source should remain clear in imports or a small attribution-preserving adapter. If primitives are extracted into local files, include the MIT copyright/permission notice from `0_external_repo/ELF_pytorch/LICENSE`; do not present copied code as newly invented layers.

Use one small node-and-unordered-pair graph block rather than retaining the language-sized ELF model. A clear block has three operations: update each pair using its current state and symmetric endpoint information; aggregate incident pairs into each node; apply the reused position-free attention/feed-forward operations. All functions are shared across atoms and pairs, use active masks, and contain no atom-index or pair-index positional embeddings. Symmetric endpoint combinations such as $h_i+h_j$ and $|h_i-h_j|$ make pair updates independent of endpoint order. Tests should verify equivariance under an atom permutation with the induced pair permutation, not just tensor shapes.

The encoder must preserve explicit pair features. A node-only equivariant representation can lose adjacency in highly symmetric molecules; retaining pairs avoids forcing the decoder to infer information the representation discarded.

The planned $E,F_\theta,q_\phi$ split is an intentional adaptation of ELF, whose trunk is shared across denoising and decoding. During the first exploration:

1. Learn encoder and decoder through reconstruction.
2. Freeze the encoder and train-fitted normalization.
3. Train the flow with velocity matching and keep the decoder trainable using noisy clean latents.

At $\lambda=0$, flow predictions receive no categorical-loss gradient. The terminal decoder learns on its own corruption branch. This is the agreed exploratory model; a shared-trunk ELF variant or a frozen-common-decoder attribution study may be considered after we inspect real behavior.

For active latent coordinates, the implementation should state the entire objective directly:

$$
\mathcal L_{\mathrm{explore}}
=\mathbb E\!\left[\frac{\|F_\theta(Z_t,t)-X\|_2^2}{m(1-t)^2}\right]
+\mathbb E\!\left[\mathrm{CE}_A(q_\phi(X+\sigma\xi),G)
+\mathrm{CE}_B(q_\phi(X+\sigma\xi),G)\right].
$$

Here $m$ counts active node and unordered-pair channels, while atom and bond CE are averaged separately. No-bond remains a real active pair category; padding does not. For atom counts of one, the bond loss contributes zero. Compute the velocity denominators, reductions, and categorical losses in float32. Keep loss components visible in returned metrics and leave the training loop in the coordinator-owned runner.

## Operational checks for the build phase

The important checks are: exact interpolation/velocity algebra; explicit pair ordering; permutation equivariance and padding isolation; finite endpoint behavior at $T$; finite forward/backward on a real graph batch; encoder parameters unchanged after freezing; terminal loss reaching the decoder but not the flow at $\lambda=0$; a short reconstruction learning run; and complete noise-to-graph sampling with the endpoint evaluation counted. These are implementation checks for the first generator, not a controlled experiment suite.

After source review, the coordinator authorized implementation against `3_model_training/emf/contracts.py`. The small model is now in `models.py`, its explicit objective is in `objectives.py`, and attributed upstream primitives are in `elf_layers.py`. Input/time and output projections remain float32 while the graph trunk supports bf16. The default encoder, flow, and decoder contain 1,663,920, 1,724,848, and 1,664,088 parameters respectively (5,052,856 total). Operational/math tests live in `5_test/test_models.py`.

No upstream files, dependencies, allocations, or training jobs were changed by the model builder.
