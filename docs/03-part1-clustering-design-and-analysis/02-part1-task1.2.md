# Part 1: Clustering Design and Analysis (for Label Discovery) (30%)

### Task 1.2 (20%):
Apply a clustering algorithm to the multi-dimensional review effort feature space you defined in Task 1.1. You are free to choose your algorithm (e.g., RFM, Subspace clustering, K-means, DBSCAN, hierarchical, Gaussian Mixture Models, Deep Clustering), but you must justify your choice and ensure the clustering’s performance is "acceptable" (you also need to define acceptable).

For each cluster, write a structured “cluster profile” that includes:
- Core review effort statistics: Median and range of each effort dimension for this cluster.
- Basic statistics of PRs belong to this cluster: that are not related to the ones used for clustering purpose. Justify why these statics are important to know. For instance, it could be the types of tasks involved in the pull requests, etc.
- Distinguishing characteristics: What makes this cluster different from adjacent clusters?
- Label: Assign a short, descriptive label. The label should be interpretable to someone unfamiliar with the dataset.

In an excellent answer, the proposed clustering method is justified well with even experimental results support. Acceptable performance defined with two or more named internal metrics and a defended threshold. Each cluster profile is complete: effort statistics, two or more justified external characteristics, distinguishing characteristics, and an interpretable label.
