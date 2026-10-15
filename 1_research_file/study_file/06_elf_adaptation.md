# ELF understanding and EMF adaptation

The coordinator and three workers completed the source study before EMF
implementation. Read the model, data, and sampling studies beside this file for
the function-by-function traces. Exact upstream revisions are recorded in
0_external_repo/provenance.json.

ELF freezes normalized T5 embeddings, predicts clean continuous destinations,
converts them into transport velocity, and uses the shared Transformer in a
terminal token-decoding mode. The JAX implementation draws a decoder/denoiser
branch per minibatch; the official PyTorch port draws one per example. Its
decoder corruption is logit-normal mixing, not our additive Gaussian corruption.
The source defaults include text RoPE, self-conditioning, guidance, Muon, and EMA.

For the first EMF generator we retain clean prediction, continuous interpolation,
velocity loss, terminal decoding, masked reductions, and suitable actual PyTorch
layer primitives. We replace T5/text positions with an equivariant node/pair graph
encoder; use separate E/F/q modules, additive decoder corruption, uniform times
below T=0.95, and the explicit sum of flow and decoder losses. These are deliberate
adaptations, not an exact reproduction of upstream ELF training. The endpoint
prediction is included in the total NFE. Initial execution uses eager PyTorch,
AdamW, and full checkpoint/RNG state for readable operation and resume.

The first experiment builds and inspects molecular generation. It does not need
a frozen common decoder, matched baselines, or near-perfect reconstruction before
the first run. Operational review checks that the implementation follows the
math and actually trains/samples; causal comparisons follow later.
