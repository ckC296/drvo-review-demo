# DR-VO Minimal Demonstration

This repository contains a minimal two-dimensional implementation of the
distributionally robust velocity obstacle (DR-VO) method described in the
accompanying paper.

The demo covers finite-horizon velocity-obstacle geometry, closed-form
distances from velocity samples to the collision set, and worst-case collision
risk under a Wasserstein ambiguity set.

## Run

```bash
pip install -r requirements.txt
python demo.py
pytest -q
```
