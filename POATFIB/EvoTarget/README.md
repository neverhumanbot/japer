# EvoTarget-FT for pre-incision POAF prediction

The study evaluates prediction of treatment-requiring postoperative atrial fibrillation after isolated CABG using the STS Adult Cardiac Surgery Database v2.81. The primary retrospective evaluation includes 21,241 procedures at 74 hospitals outside model development. The paper reports hospital-level uncertainty and describes how the holdout was used to select the featured run.

- [Compiled manuscript](docs/paper/main.pdf) and [LaTeX source](docs/paper/main.tex)
- [Research and supplement index](docs/README.md), including the complete 103-arm leaderboard, feature screening, full-registry ensembles, and the counterfactual embedding and JAIN experiments
- [POAF and STS literature review](docs/paper/POAF_LITERATURE_REVIEW.md) and [statistical review](docs/paper/RED_TEAM_REVIEW.md)
- [Label-blind matched evaluation](FEATURE_MATCHED_EVALUATION.md) and [dataset interface](dataset/DATASET_README.md)
- [Primary and subgroup confidence intervals](reports/paper_red_team_receipt.json), [benchmark and calibration results](reports/paper_v3_master_receipt.json), and [feature-augmentation intervals](reports/agentic_fe_cluster_receipt.json)

Rebuild the manuscript with pdfLaTeX and BibTeX. The audit checks the source against the shipped aggregate receipts and confirms the figure and bibliography references.

```sh
python3 scripts/20_audit_paper_numbers.py
make -C docs/paper -B
```

The bundle contains aggregate results and publication artifacts. Record-level STS data, split assignments, saved predictions, and model weights are held in the governed research workspace. See the archival notes for original experiment details and the manuscript for the current interpretation.
