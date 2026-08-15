# Experiment Tracking

## Description
Manages experiment sessions: creating session notes, tracking changes, comparing results across runs.

## Triggers
"experiment tracking", "session notes", "compare runs", "log experiment", "training history", "track results"

## Workflow

### Step 1 — List existing sessions
```bash
Get-ChildItem "session-ses_*.md" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    $first_line = ($content -split "`n")[0]
    Write-Host "$($_.Name) — $first_line"
}
```

### Step 2 — Create new session note
Create `session-ses_{timestamp}.md` with:
```markdown
# Session: {date} - {description}

## Changes Made
- {change 1}
- {change 2}

## Config
```json
{ paste gpt_mini3.json here }
```

## Results
| Epoch | Loss | tok/s | Notes |
|-------|------|-------|-------|
| 1 | X.XX | XXXXX | |
| 5 | X.XX | XXXXX | |

## Observations
- {observation 1}
- {observation 2}
```

### Step 3 — Compare checkpoint results
```bash
python -c "
import json
from pathlib import Path
import glob as glob_mod

ckpt_dir = Path('E:/training/checkpoints')
for d in sorted(ckpt_dir.iterdir()):
    if not d.is_dir():
        continue
    resume = d / 'resume.json'
    if not resume.exists():
        continue
    r = json.load(open(resume))
    tiers = [t.name for t in d.iterdir() if t.is_dir()]
    print(f'{d.name}: epoch={r.get(\"epoch\",\"?\")} loss={r.get(\"loss\",\"?\")} tiers={tiers}')
"
```

### Step 4 — Extract loss trajectory
```bash
$hash = "latest_checkpoint_hash"
$status = "E:\training\checkpoints\$hash\checkpoint_status.txt"
if (Test-Path $status) {
    Import-Csv $status -Delimiter "`t" -Header time,epoch,batch,loss,tok_per_s,batch_per_s,total |
        Select-Object epoch,loss,tok_per_s,batch_per_s |
        Format-Table -AutoSize
}
```

### Session file naming
Files follow pattern `session-ses_{hex_id}.md`:
- `session-ses_007f.md` — existing sessions
- `session-ses_0180.md` — existing sessions
- `session-ses_01d6.md` — existing sessions

### Key fields to track per session
| Field | Source |
|-------|--------|
| Config changes | `git diff gpt_mini3.json` |
| Corpus changes | `compute_corpus_hash` output |
| Vocab hash | `get_vocab_hash` output |
| Checkpoint hash | `get_model_hash` output |
| Final loss | Last line of `checkpoint_status.txt` |
| Peak tok/s | Max of `tok/s` column |
| Training duration | `checkpoint_status.txt` first/last timestamp |
