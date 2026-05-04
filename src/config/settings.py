"""Project-wide constants and configuration."""

import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "docs" / "06-figs"

HF_ROOT = "hf://datasets/hao-li/AIDev"

OUTLIER_PERCENTILE = 95

COLOR_MAP: dict[str, str] = {
    "Human": "#56B4E9",
    "OpenAI_Codex": "#D55E00",
    "Devin": "#009E73",
    "Copilot": "#0072B2",
    "Cursor": "#785EF0",
    "Claude_Code": "#DC267F",
}

AGENT_DISPLAY: dict[str, str] = {
    "OpenAI_Codex": "OpenAI Codex",
    "Devin": "Devin",
    "Copilot": "GitHub Copilot",
    "Cursor": "Cursor",
    "Claude_Code": "Claude Code",
}

AGENT_ORDER: list[str] = [
    "OpenAI_Codex",
    "Devin",
    "Copilot",
    "Cursor",
    "Claude_Code",
]

TASK_TYPE_ORDER: list[str] = [
    "feat",
    "fix",
    "perf",
    "refactor",
    "style",
    "docs",
    "test",
    "chore",
    "build",
    "ci",
    "other",
]

EFFORT_DIMENSIONS: list[str] = [
    "time_to_merge_hours",
    "n_formal_reviews",
    "n_review_comments",
    "n_unique_reviewers",
    "churn_per_review_cycle",
]

EFFORT_DIM_LABELS: dict[str, str] = {
    "time_to_merge_hours": "duration (hours)",
    "n_formal_reviews": "iterations",
    "n_review_comments": "depth (comments)",
    "n_unique_reviewers": "breadth (reviewers)",
    "churn_per_review_cycle": "burden (lines/review)",
}

EFFORT_DIM_DESCRIPTIONS: dict[str, str] = {
    "time_to_merge_hours": "wall-clock time from pr creation to merge. captures how long reviewers need to reach consensus.",
    "n_formal_reviews": "count of formal review submissions (approved, changes_requested, commented). captures how many revision rounds were needed.",
    "n_review_comments": "count of inline review comments on specific code lines. captures depth of scrutiny within each review round.",
    "n_unique_reviewers": "count of distinct human reviewers who submitted a formal review. captures breadth of reviewer involvement - whether scrutiny comes from many eyes or few.",
    "churn_per_review_cycle": "total lines changed per formal review round. captures the cognitive load each review round imposes - more code per review means more to evaluate.",
}

RANDOM_STATE = 42

CLUSTER_K_RANGE = range(2, 9)

SILHOUETTE_THRESHOLD = 0.30
DBI_THRESHOLD = 1.50

CLUSTER_PALETTE = {
    "0": "#4C72B0",
    "1": "#DD8452",
    "2": "#55A868",
    "3": "#C44E52",
    "4": "#8172B2",
    "5": "#CCB974",
    "6": "#64B5CD",
    "7": "#8C8C8C",
}

TEST_SIZE = 0.20
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.95
