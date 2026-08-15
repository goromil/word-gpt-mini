# .agents/ — Agent Skills for Word-GPT-Mini

Skills are instruction files loaded by opencode when a task matches their description.

| Skill | File | Triggered By |
|-------|------|---|
| Training Lifecycle | `train-run.md` | "start training", "resume", "launch trainer" |
| Diagnostics | `diagnostics.md` | "NaN loss", "OOM", "crash", "segfault" |
| Data Pipeline | `data-pipeline.md` | "download corpus", "build dataset", "corpus status" |
| Config Tuning | `config-tuning.md` | "change architecture", "tune config", "fit model" |
| Cache & Checkpoint | `cache-checkpoint.md` | "check cache", "checkpoint info", "list checkpoints" |
| Experiment Tracking | `experiment-tracking.md` | "session notes", "compare runs", "log experiment" |
| Code Development | `code-dev.md` | "change code", "implement feature", "fix bug" |
| Common Patterns | `common.md` | shared references — not triggered, referenced by other skills |

## How It Works

When you ask opencode to do something (e.g., "loss is NaN"), it matches your request
to the most relevant skill and loads its instructions. Skills contain exact commands,
file paths, project-specific knowledge, and troubleshooting tables.

## Adding a New Skill

1. Create `.agents/my-skill.md`
2. Include: Description, Triggers, Workflow (numbered steps with commands), Key Files
3. Follow the format of existing skills for consistency
