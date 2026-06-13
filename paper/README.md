# GenToolReal Paper

CoRL/NeurIPS-style draft documenting GenToolReal: generative trajectory prediction for reactive tool manipulation via language-conditioned flow matching (contact-frame trajectories, procedural reactive data, GRPO refinement, closed-loop brush/hammer deployment).

## Build

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf` (~10 pages).

## Contents

| File | Description |
|------|-------------|
| `main.tex` | Full paper source |
| `references.bib` | Bibliography (ported from GenToolReal + GRPO/RL entries) |
| `figures/` | Placeholder images + README with sources for final figures |

## Conference style

To switch to CoRL style, replace `\documentclass{article}` with the conference package (see comment at top of `main.tex`).

## Figures to replace

See `figures/README.md` for suggested renders from `viz_interactive.py`, `render_reactive_rollout_viz.py`, and robot Viser debug node.

## Results

Section 6 is framed qualitatively around brush/hammer deployment highlights.
