# Santa 2026 - The Elf Calculator

Local validator, simulator, baseline solver, and submission portfolio merger for Kaggle's **Santa 2026 - The Elf Calculator**.

## Setup
1. Download `puzzles.csv`, `practice_puzzles.csv`, and `sample_submission.csv` from the Kaggle Data tab into `santa2026/data/`.
2. Validate a submission:
   ```bash
   python3 validator.py data/puzzles.csv data/sample_submission.csv
   ```
3. Run the baseline solver:
   ```bash
   python3 solver.py data/puzzles.csv submissions/baseline.csv
   ```
4. Run unit tests:
   ```bash
   python3 -m unittest test_santa2026.py -v
   ```
