# NHL Moneyline Model

Daily NHL moneyline predictions powered by a logistic regression model trained on season-to-date results.

## Repo Structure

```
├── predx_api_V1.py              # Main model script
├── requirements.txt             # Python dependencies
├── output/
│   ├── history_predictions.csv  # All predictions + results (tracked in git)
│   └── model_coefficients.csv   # Latest trained model weights
├── docs/
│   ├── latest.html              # Published via GitHub Pages (today's picks)
│   └── YYYY-MM-DD.html          # Archive of each day
└── .github/workflows/
    └── daily_predictions.yml    # Runs automatically every morning
```

## GitHub Actions Setup

### 1. Create a Personal Access Token (PAT)
You need a PAT to let the workflow push HTML to your `docs/` folder:
- Go to GitHub → **Settings → Developer settings → Personal access tokens → Fine-grained tokens**
- Create a token with **Contents: Read and Write** on this repo
- Copy the token value

### 2. Add the token as a secret
- In your repo: **Settings → Secrets and variables → Actions → New repository secret**
- Name: `PUBLISH_TOKEN`
- Value: paste your PAT

### 3. Enable GitHub Pages
- Go to **Settings → Pages**
- Source: **Deploy from a branch**
- Branch: `main`, Folder: `/docs`
- Your daily report will be live at: `https://<owner>.github.io/<repo>/latest.html`

### 4. Push the files
```bash
git add predx_api_V1.py requirements.txt .github/ output/
git commit -m "feat: add daily predictions workflow"
git push
```

The workflow runs automatically at **7 AM ET every day**. You can also trigger it manually from the **Actions** tab.

## Running Locally

```bash
pip install -r requirements.txt

# Run today's predictions
python predx_api_V1.py --write_daily --season_start 2025-10-07

# Backfill a date range
python predx_api_V1.py --backfill_start 2025-10-07 --backfill_end 2025-12-31 \
  --season_start 2025-10-07 --backfill_disable_clubstats 1
```

## Model Features

| Feature | Description |
|---|---|
| `points_pct_diff` | Season points % (home − away) |
| `gd_per_g_diff` | Goal differential per game |
| `l10_points_pct_diff` | Last-10-game points % |
| `l10_gd_per_g_diff` | Last-10-game GD per game |
| `pp_pct_diff` | Power play % |
| `pk_pct_diff` | Penalty kill % |
| `sf_per_g_diff` | Shots for per game |
| `sa_per_g_diff` | Shots against per game |
| `rest_days_diff` | Days rest (capped ±5) |
| `b2b_diff` | Back-to-back flag |
| `3in4_diff` | 3 games in 4 days flag |
| `4in6_diff` | 4 games in 6 days flag |

## Key Fixes (v1 → current)

- **`rest_days` capped at 5**: All-Star break / Olympics / schedule gaps were producing values like ±21, completely outside the training distribution, causing extreme skewed predictions. Now capped at 5.
- **Stronger regularization (C=0.3)**: Reduces overconfidence from noisy/colinear features.
- **Removed hardcoded paths and token**: Script uses relative paths and reads token from env.
