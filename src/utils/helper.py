"""Project-wide helper functions."""

import numpy as np
import pandas as pd
from src.config.settings import HF_ROOT, OUTLIER_PERCENTILE


def load_data() -> dict[str, pd.DataFrame]:
    """Load AIDev-pop tables from HuggingFace and return as a dict."""
    tables = {
        "pr": f"{HF_ROOT}/pull_request.parquet",
        "human_pr": f"{HF_ROOT}/human_pull_request.parquet",
        "pr_reviews": f"{HF_ROOT}/pr_reviews.parquet",
        "pr_review_comments": f"{HF_ROOT}/pr_review_comments_v2.parquet",
        "pr_commit_details": f"{HF_ROOT}/pr_commit_details.parquet",
        "pr_task_type": f"{HF_ROOT}/pr_task_type.parquet",
        "human_pr_task_type": f"{HF_ROOT}/human_pr_task_type.parquet",
    }
    data = {name: pd.read_parquet(path) for name, path in tables.items()}
    for name, df in data.items():
        print(f"  {name:25s} {len(df):>10,} rows")
    return data


def load_base_parquet(path: str | None = None) -> pd.DataFrame:
    """Load the analysis-ready parquet from A2."""
    from src.config.settings import DATA_DIR

    if path is None:
        path = DATA_DIR / "analysis_ready_prs.parquet"
    return pd.read_parquet(path)


def load_task_types() -> pd.DataFrame:
    """Load agentic PR task types from HuggingFace."""
    return pd.read_parquet(f"{HF_ROOT}/pr_task_type.parquet")


def load_pr_reviews() -> pd.DataFrame:
    """Load agentic PR reviews from HuggingFace."""
    return pd.read_parquet(f"{HF_ROOT}/pr_reviews.parquet")


def filter_agentic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only agentic (AI-authored) PRs with review data."""
    return df[df["is_agentic"] == True].copy()


def compute_effort_dimensions(df: pd.DataFrame, rev_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute all review-effort dimensions for clustering.

    Dimensions:
    1. time_to_merge_hours    - already present (from A1/A2)
    2. n_formal_reviews       - already present (from A1/A2)
    3. n_review_comments      - already present (from A1/A2)
    4. n_unique_reviewers     - distinct reviewers per PR (requires pr_reviews table)
    5. churn_per_review_cycle - total_churn / max(n_formal_reviews, 1)
    """
    df = df.copy()

    if rev_df is not None:
        reviewer_counts = (
            rev_df[rev_df["user_type"] == "User"]
            .groupby("pr_id")["user"]
            .nunique()
            .rename("n_unique_reviewers")
        )
        df = df.merge(reviewer_counts, left_on="id", right_index=True, how="left")
        df["n_unique_reviewers"] = df["n_unique_reviewers"].fillna(0).astype(int)
    elif "n_unique_reviewers" not in df.columns:
        raise ValueError("n_unique_reviewers not in df and rev_df not provided")

    df["churn_per_review_cycle"] = np.where(
        df["n_formal_reviews"] > 0,
        df["total_churn"] / df["n_formal_reviews"],
        df["total_churn"],
    )

    return df


def merge_task_types(df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
    """Merge LLM-classified task types onto the PR dataframe."""
    task_lookup = task_df.drop_duplicates(subset="id")[["id", "type"]].rename(
        columns={"type": "task_type"}
    )
    return df.merge(task_lookup, left_on="id", right_on="id", how="left")


def load_clustering_ready() -> pd.DataFrame:
    """Load the clustering-ready parquet from notebook 01."""
    from src.config.settings import DATA_DIR

    return pd.read_parquet(DATA_DIR / "clustering_ready_prs.parquet")


def load_clustered() -> pd.DataFrame:
    """Load the clustered parquet from notebook 03."""
    from src.config.settings import DATA_DIR

    return pd.read_parquet(DATA_DIR / "clustered_prs.parquet")


def preprocess_effort_dims(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Apply log1p transform to all effort dimensions and return with _log suffixed columns.

    Log1p is chosen because:
    - all dimensions are >= 0 (counts, durations, ratios)
    - distributions are heavily right-skewed (see notebook 01 histograms)
    - log1p preserves zeros (unlike log) which matters for n_formal_reviews etc.
    - reduces the influence of extreme outliers without dropping them
    """
    df = df.copy()
    for col in dims:
        log_col = f"{col}_log"
        df[log_col] = np.log1p(df[col])
    return df


def compute_pairwise_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return spearman rank correlation matrix for the given columns.

    Spearman is preferred over pearson because:
    - effort dimensions are non-normal (right-skewed, zero-inflated)
    - we care about monotonic relationships, not linear ones
    - more robust to outliers
    """
    return df[cols].corr(method="spearman")


def summarize_zeros(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a dataframe with zero-count and zero-percentage for each column."""
    rows = []
    n = len(df)
    for col in cols:
        n_zero = int((df[col] == 0).sum())
        rows.append({
            "dimension": col,
            "n_zeros": n_zero,
            "pct_zeros": 100.0 * n_zero / n,
            "n_total": n,
        })
    return pd.DataFrame(rows)


def build_feature_matrix(df: pd.DataFrame, dims: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build the log1p + standardized feature matrix for clustering.

    Returns:
        X_scaled: ndarray of shape (n_samples, n_features)
        X_log: ndarray of log1p-transformed (but not scaled) values
    """
    from sklearn.preprocessing import StandardScaler

    X_log = np.log1p(df[dims].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    return X_scaled, X_log


def evaluate_k_range(X: np.ndarray, k_range: range, random_state: int = 42) -> pd.DataFrame:
    """Run KMeans for each k and compute silhouette, Davies-Bouldin, Calinski-Harabasz.

    Returns a dataframe with one row per k.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "inertia": km.inertia_,
        })
    return pd.DataFrame(rows)


def fit_gmm(X: np.ndarray, n_components: int, random_state: int = 42):
    """Fit a Gaussian Mixture Model and return the fitted model + labels."""
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=5,
        random_state=random_state,
    )
    labels = gmm.fit_predict(X)
    return gmm, labels


def compute_cluster_profiles(
    df: pd.DataFrame,
    label_col: str,
    effort_dims: list[str],
    external_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute structured cluster profiles.

    For each cluster, returns:
    - median and range (min, max) of each effort dimension
    - count and percentage of total
    - distribution of external columns (e.g., task_type, agent)
    """
    profiles = []
    for cluster_id in sorted(df[label_col].unique()):
        mask = df[label_col] == cluster_id
        subset = df[mask]
        n = len(subset)

        profile = {"cluster": cluster_id, "n": n, "pct": 100.0 * n / len(df)}

        for dim in effort_dims:
            vals = subset[dim]
            profile[f"{dim}_median"] = vals.median()
            profile[f"{dim}_min"] = vals.min()
            profile[f"{dim}_max"] = vals.max()

        profiles.append(profile)

    return pd.DataFrame(profiles)


def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to PRs with non-empty body and return with cluster labels intact."""
    has_body = df["body"].notna() & (df["body"].str.strip() != "")
    return df[has_body].copy().reset_index(drop=True)


def preprocess_text(text: pd.Series) -> pd.Series:
    """Clean PR body text: lowercase, strip markdown artifacts, collapse whitespace."""
    import re

    def _clean(t: str) -> str:
        t = t.lower()
        t = re.sub(r"```[\s\S]*?```", " ", t)
        t = re.sub(r"`[^`]+`", " ", t)
        t = re.sub(r"#+\s", " ", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"[*_~|>]", " ", t)
        t = re.sub(r"https?://\S+", " ", t)
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    return text.apply(_clean)


CLUSTER_LABEL_MAP: dict[int, str] = {
    0: "intensively reviewed",
    1: "auto-merged small",
    2: "cursorily reviewed",
    3: "auto-merged large",
}
