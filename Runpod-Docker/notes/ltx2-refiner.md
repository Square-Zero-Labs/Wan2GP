# LTX‑2 Self Refiner (Wan2GP)

## What It Is
The **Self Refiner** is a Wan2GP‑specific refinement pass that re‑denoises selected diffusion steps to improve video quality and stability. It runs *inside* the standard denoising loop and re‑evaluates noisy latents multiple times at specific steps.

## Where It Runs
- **LTX‑2 pipelines** (`distilled`, `two‑stage`, `one‑stage`) can use it.
- It is **disabled by default** (UI dropdown = Disabled).
- It runs **only if enabled** and (optionally) a plan is provided.

## How It Works (High‑Level)
1) During each denoising step, the refiner can run extra “micro‑steps”.
2) For those steps:
   - The current latent is **perturbed** with noise.
   - The model **re‑denoises** it.
   - The refiner measures **uncertainty** using a norm over the change in predicted x0.
   - A **certain/uncertain mask** blends refined vs previous predictions.
3) If enough tokens become “certain”, refinement stops early.

This can improve stability and reduce artifacts, at the cost of extra compute.

## Inputs / Controls
- **Self Refiner Setting**: Disabled / P1 norm / P2 norm
- **Plan (P&P Plan)**: `start-end:repeat_count` entries (comma‑separated). The **first numbers** are the **denoising steps** to refine (0‑indexed, inclusive), and the **number after `:`** is how many **refiner passes** are added per step. Example: `2-6:2` means steps 2 through 6 each get 2 extra refinement passes.
- **Important:** refinement runs only when `repeat_count > 1`. A value of `:1` is a no‑op.
- **Uncertainty Threshold**: how strict “certainty” is
- **Certain Percentage**: if enough tokens are “certain”, skip further refinement

## Default Behavior
If the refiner is **enabled** but the plan is empty, it falls back to the **code default** in `shared/utils/self_refiner.py`:
```
1-5:3,6-13:1
```

## Stage‑1 vs Stage‑2
- **Stage‑1** uses the first plan.
- **Stage‑2** runs only if a **second plan** is provided (separated with `;`).

Example:
- `2-8:3` → stage‑1 only
- `2-8:3;2-4:2` → stage‑1 and stage‑2

## P1 vs P2 Norm
The refiner uses a norm to measure uncertainty:
- **P1 norm** = L1 distance (sum of absolute differences). More robust/stable.
- **P2 norm** = L2 distance (Euclidean). More sensitive to outliers.

**Recommendation for stability issues:** start with **P1**.

## Recommended Plans (Distilled LTX‑2)
Default distilled uses **8 steps in stage‑1** and **3 steps in stage‑2**.

Start with a light plan:
- **Stage‑1 only:** `2-6:2`
- **Stage‑1 + Stage‑2:** `2-6:2;1-2:2`

If you want a little stronger:
- **Stage‑1 only:** `2-8:2`
- **Stage‑1 + Stage‑2:** `2-8:2;1-2:2`

## File Locations (Reference)
- Core logic: `shared/utils/self_refiner.py`
- LTX‑2 hookup: `models/ltx2/ltx_pipelines/distilled.py` and `ti2vid_two_stages.py`
- UI wiring: `wgp.py` (Self Refiner settings)
