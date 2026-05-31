# ComfyUI Anima Flow Corrective Sampler

Custom ComfyUI sampler nodes for Anima / Cosmos-style rectified-flow image
models. It packages the official Anima Diffusers FlowMatch Euler schedule by
default, with optional UniPC and PC3 solver variants for A/B testing in ComfyUI.

This is an independent implementation, not an official NVIDIA or CircleStone
Labs release. The design is aligned with public Cosmos / Cosmos Predict2.5
pipeline ideas such as rectified-flow scheduling and UniPC-style
predictor-corrector sampling.

The default profile follows the public Anima Diffusers scheduler config:

```text
solver        = flow_euler
schedule      = flow_diffusers_linear_shift
flow_shift    = 3.0
steps         = 30
cfg           = 5.0
cfg_mode      = const
final_clean_pass = false
```

The goal is to improve prompt structure, spatial relationships, and detail
stability while keeping the node surface small enough for daily use.

## Update

May 31, 2026:

- The packaged default now uses 30 steps, CFG 5.0, constant CFG,
  `flow_euler`, `flow_diffusers_linear_shift`, `flow_shift 3.0`, and no final
  clean pass.
- Added `flow_diffusers_linear_shift`, mirroring the non-dynamic-shifting
  `FlowMatchEulerDiscreteScheduler` sigma grid used by
  `Anima-Base-v1.0-Diffusers`.
- Added two explicit Diffusers-grid experimental solvers:
  `flow_unipc2_diffusers_x0` and `flow_pc3_diffusers_damped`. Both keep the
  existing solver math but use conservative low-sigma tail handling on the
  Diffusers timestep grid.
- Kept the previous enhanced profile selectable:
  `flow_unipc2_x0 + flow_rf_linear_shift + flow_shift 5.0`.
- Simplified the released node surface to `Anima Flow Corrective Sampler` and
  `Anima Flow Settings`.

## Example Output

The files below were generated from the same prompt family and are included as
single-image examples instead of a four-way grid. Import
[`examples/workflows/anima_better_sampler.json`](examples/workflows/anima_better_sampler.json)
in ComfyUI to inspect the example workflow.

| UniPC, linear shift5, const CFG 7 | PC3, linear shift5, const CFG 7 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/KeithZ117/Comfyui-anima-sampler/main/examples/comparison/unipc_linear_shift_cfg7.jpg" alt="UniPC linear shift CFG 7 example" width="260"> | <img src="https://raw.githubusercontent.com/KeithZ117/Comfyui-anima-sampler/main/examples/comparison/pc3_linear_shift_cfg7.jpg" alt="PC3 linear shift CFG 7 example" width="260"> |

| er_sde + simple, CFG 4.5 | er_sde + simple, CFG 7 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/KeithZ117/Comfyui-anima-sampler/main/examples/comparison/er_sde_simple_cfg45.jpg" alt="er_sde simple CFG 4.5 example" width="260"> | <img src="https://raw.githubusercontent.com/KeithZ117/Comfyui-anima-sampler/main/examples/comparison/er_sde_simple_cfg7.jpg" alt="er_sde simple CFG 7 example" width="260"> |

## Nodes

- `Anima Flow Corrective Sampler`: the main sampler node.
- `Anima Flow Settings`: optional advanced controls for solver and CFG tuning.

The sampler works without connecting `Anima Flow Settings`; the tested defaults
are built in. Connect the settings node only when you want to tune advanced
parameters.

The sampler outputs both `LATENT` and `IMAGE`. Connect a `VAE` to the optional
`vae` input when you want the image output decoded directly from the sampler.
Leave `vae` disconnected when you only need the latent output.

`ramp cfg` starts guidance low and smoothly raises it to the selected `cfg`.
With the default `cfg=5`, it starts near `4.5`, keeps that low guidance through
the early high-noise phase, and reaches `5` before the tail/detail phase.

## Install

Install with ComfyUI-Manager when the node is available in the registry, or
install directly from GitHub.

GitHub install:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/KeithZ117/Comfyui-anima-sampler.git
```

Manual install:

Copy this repository into ComfyUI's `custom_nodes` directory:

```text
ComfyUI/custom_nodes/Comfyui-anima-sampler
```

Then restart ComfyUI.

This repository does not include model weights. Install Anima and its required
text encoder / VAE files according to the official Anima model card:

- https://huggingface.co/circlestone-labs/Anima

## Recommended Use

Use `Anima Flow Corrective Sampler` in place of a normal sampler node.

Everyday controls:

- `steps`: default `30`
- `cfg`: default `5.0`
- `cfg_mode`: recommended `const` or `ramp cfg`
- `flow_solver`: default `flow_euler`
- `flow_schedule`: default `flow_diffusers_linear_shift`
- `flow_shift`: default `3.0` and used by shift-aware schedules such as
  `flow_diffusers_linear_shift`, `flow_cosmos_rf_tail`, and
  `flow_rf_linear_shift`
- `denoise`
- `add_noise`
- optional `vae` for direct image output

`flow_cosmos` is now the pure Cosmos RFlow-shaped schedule. Use
`flow_cosmos_rf_tail` when you want the shifted RF-tail path controlled by
`flow_shift`.

`flow_cosmos_rho7` is the pure rho7/Karras-style Cosmos schedule. Connect
`Anima Flow Settings` and enable `flow_rho7_tail_auto` to add the previous
RF-tail-auto modification on top of rho7.

`flow_diffusers_linear_shift` mirrors the non-dynamic-shifting
`FlowMatchEulerDiscreteScheduler` path used by the Anima Diffusers release. The
packaged default uses `flow_shift 3.0` and no final clean pass.

`flow_rf_linear_shift` follows the newer Cosmos Predict2.5 normalized RF
linear inference grid with the same shift formula used by their FlowUniPC
scheduler. It uses `flow_shift` directly; `flow_shift 5.0` matches their
published default shape. When this schedule is selected without connecting
`Anima Flow Settings`, the sampler automatically disables `final_clean_pass`
to match Cosmos 2.5's default "walk to terminal zero" behavior.

`flow_rf_linear_s_tail_shift5` is a fixed shift-5 extension of
`flow_rf_linear_shift`. It keeps the early section close to linear shift5, then
uses an S-shaped sigmoid tail near the final 30-step region to enter low-noise
refinement more smoothly.
The external `flow_shift` value is ignored by this schedule because shift5 is
baked into the preset name.

## Current Default

The packaged default now follows the official Anima Diffusers-style path:
`flow_euler + flow_diffusers_linear_shift + flow_shift 3.0 + const cfg 5.0`,
with no final clean pass when the settings node is disconnected.

Official reference combinations:

- Cosmos2: `AB2 x0/denoised solver + Karras/rho7 sigmas`
  (`sigma_max 80.0`, `sigma_min 0.002`, `rho 7`) + constant CFG + final
  clean pass at the last non-zero sigma.
- Anima Diffusers: `FlowMatchEulerDiscreteScheduler`
  (`shift 3`, `use_dynamic_shifting false`, `stochastic_sampling false`) +
  Euler stepping, walking to terminal zero.
- Cosmos 2.5: `FlowUniPC order 2 + normalized RF linear shift schedule`
  (`shift 5` in the released configs) + constant CFG, walking to terminal
  zero without the old Cosmos2-style final clean pass.

The previous enhanced profile remains selectable as
`flow_unipc2_x0 + flow_rf_linear_shift + flow_shift 5.0 + const cfg 7.0`.

`flow_pc3_damped + flow_cosmos` and `flow_cosmos_rf_tail + flow_shift 5.0`
remain available as explicit alternatives from previous testing.

`flow_cosmos_rf_tail` preserves the old shifted RF-tail behavior.
`flow_cosmos_rho7` remains available as a quality/reference baseline.
`flow_rf_linear_shift` is available for testing the newer Cosmos 2.5 default
linear+shift schedule.
`flow_rf_linear_s_tail_shift5` is available as the fixed shift5 linear+S-tail
variant.
`simple` remains available for comparison with native ComfyUI-style schedules.

`flow_unipc2_x0` is available as an Anima-adapted FlowUniPC solver. It applies
the Cosmos 2.5 UniP/UniC BH predictor/corrector structure directly to
ComfyUI's denoised/x0 output instead of treating it as Cosmos 2.5 velocity
output. The settings node exposes the original-style UniPC controls for
solver order, `bh1`/`bh2`, lower-order final, disabled early correctors, and
dynamic thresholding.

`flow_unipc2_diffusers_x0` reuses the same UniPC math with a conservative
Diffusers-grid policy: order-2 body steps, first-order low-sigma tail steps, and
tail corrector skips where the official Diffusers grid has very large lambda
gaps.

`flow_pc3_diffusers_damped` reuses PC3 in the stable body of the Diffusers grid
and skips endpoint correction in the low-sigma tail. This keeps the old
`flow_pc3_damped` behavior unchanged while making the Diffusers-grid experiment
explicit.

`flow_ab2` is available as a one-model-call Adams-Bashforth 2 solver. It uses
the previous x0 prediction after an Euler warmup step, matching the residual
x0 AB2 idea in the Cosmos reference scheduler while using this sampler's
normalized RF time.

The final clean pass is optional. When enabled, the integration loop stops at
the last non-zero sigma instead of taking the appended terminal zero interval,
then asks the model for one more x0/denoised prediction at that same non-zero
sigma. This matches the explicit clean pass used in the Cosmos reference
pipelines more closely than cleaning a terminal-zero state. The disconnected
daily defaults for `flow_diffusers_linear_shift`, `flow_rf_linear_shift`, and
`flow_rf_linear_s_tail_shift5` leave it off so the linear RF presets walk to
terminal zero.

## Development

Run the local test suite from the repository root:

```text
python -m unittest discover -s tests
```

## References

- NVIDIA Cosmos Predict2: https://github.com/nvidia-cosmos/cosmos-predict2
- NVIDIA Cosmos Predict2.5: https://github.com/nvidia-cosmos/cosmos-predict2.5
- UniPC paper: https://arxiv.org/abs/2302.04867

## License

This sampler code is released under the MIT License. See `LICENSE`.

Model weights are not included and are not relicensed by this project. The
official Anima model card currently lists Anima under the CircleStone Labs
Non-Commercial License and notes that it is also subject to the NVIDIA Open
Model License Agreement for derivative Cosmos models.
