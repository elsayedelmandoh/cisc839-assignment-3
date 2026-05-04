# Assignment 3: Clustering and Text Classification using Agentic PR Dataset

Till this point, you should have a basic view of the dataset and review efforts of agentic PRs.

In this assignment, we ask a different but related question: what natural groups exist in agentic PRs based on their review efforts, what are the characteristics of these clusters, and can those groups be predicted from the description of the pull request (e.g., what tasks they aim to resolve)?


## Dataset
Same as A1 and A2. The dataset contains GitHub pull requests authored either by human developers or by AI coding agents, made available through the MSR 2026 Mining Challenge.

Information about this dataset can be accessed from: https://2026.msrconf.org/track/msr-2026-mining-challenge?#Call-for-Mining-Challenge-Papers
---

## Deliverables
Submit the following:

A PDF Report (4 pages maximum, font 11, following the provided Word template) that includes your answer to all tasks in this assignment.

A replication package that allows us to reproduce your analysis and validate your usage of AI. This package should include:

- Runnable code supporting your report (Jupyter notebook or script)
- An Excel file documenting the prompts that has influenced the submitted version of your assignment (you could exclude the original trial ones). The file should contain three columns: Purpose (the task the prompt was used for), Prompt (exact prompt you submitted to the AI tool), How you used it (How the AI response influenced your report)
---

## Part 1: Clustering Design and Analysis (for Label Discovery) (30%)

### Task 1.1(10%): 
Identify and define at least three different dimensions of review effort that can be constructed from the dataset. You could bring your A1 and A2’s insights here and reuse your code, etc.

For each dimension:
- State the variable name or formula used to compute it.
- Explain, in domain terms, what aspect of review effort it captures that is not already captured by the others.
- Justify any preprocessing choice (i.e., transfer feature values)

Excellent answer should contain three or more dimensions, each with a clear formula, a unique and well-motivated domain justification, and preprocessing rationale.

### Task 1.2 (20%):
Apply a clustering algorithm to the multi-dimensional review effort feature space you defined in Task 1.1. You are free to choose your algorithm (e.g., RFM, Subspace clustering, K-means, DBSCAN, hierarchical, Gaussian Mixture Models, Deep Clustering), but you must justify your choice and ensure the clustering’s performance is "acceptable" (you also need to define acceptable).

For each cluster, write a structured “cluster profile” that includes:
- Core review effort statistics: Median and range of each effort dimension for this cluster.
- Basic statistics of PRs belong to this cluster: that are not related to the ones used for clustering purpose. Justify why these statics are important to know. For instance, it could be the types of tasks involved in the pull requests, etc.
- Distinguishing characteristics: What makes this cluster different from adjacent clusters?
- Label: Assign a short, descriptive label. The label should be interpretable to someone unfamiliar with the dataset.

In an excellent answer, the proposed clustering method is justified well with even experimental results support. Acceptable performance defined with two or more named internal metrics and a defended threshold. Each cluster profile is complete: effort statistics, two or more justified external characteristics, distinguishing characteristics, and an interpretable label.
---

## Part 2: Text-based Label Prediction (40%)
Use the cluster labels from Part 1 as the ground truth for a supervised text classification task. Your goal is to predict which effort group a PR belongs to using the description of the pull request.

### Task 2.1 (20%): Classification Pipeline 

Describe and justify your text representation and classification pipeline:

- Text source: You will use PR descriptions as the text input. Discuss what linguistic or topical signals in descriptions you expect to correlate with different effort clusters, and whether any preprocessing is needed.
- Classification methodology: Describe your approach for learning the representation from text (e.g., TF-IDF, word embeddings, pre-trained transformer embedding models, fine-tuned language models) and classify the learned representation. The approach should be well justified (even with experimental results supporting). You could use existing SOTA text classification models.

### Task 2.2 (10%): Evaluation 

Implement your proposed classification approach and conduct experiments (you should design the split of data for train and test. Report classification performance. Note that clusters are unlikely to be perfectly balanced, accuracy alone is insufficient.

Excellent answer should provide a reasonably good performing model, report per-class precision, recall, and F1 reported for all clusters. Class imbalance explicitly acknowledged. Confusion matrix presented and well interpreted.


### Task 2.3 (10%): Error Analysis and Cluster Quality Reflection 

Identify the pair of cluster labels your classifier confuses most often. Then answer the following:
- Which two clusters are most often confused, and in which direction (which is misclassified as which)?
- What do you believe is the main cause of this confusion? Support your argument with at least one piece of quantitative or textual evidence.
- What does this confusion tell you about the quality of your clustering solution from Part 1?

Excellent answer should correctly identify the two asked clusters and present a solid way to identify the potential main root cause. Reflection connects to Part 1 correctly.
---

## Part 3: GenAI-Augmented Reflection (30%)
In this part, you will reflect on how AI tools were used for part 1 and part 2 as a co-engineer. 

Following the below steps to complete this task:

- List all major decisions about your final approach for part 1 and part 2 where AI is used.
- For each decision, describe how you evaluated the AI’s suggestion before accepting or rejecting it. What did you check? Did you run any validation? Include a representative prompt you used for that decision.
- Choose the one decision in your analysis where the most meaningful iterative refinement occurred, i.e., where your initial AI interaction produced something, you had to revise.

Excellent answer should have all major decisions documented with prompts and an honest account of how AI suggestions were evaluated (not just accepted). And provide some concrete before/after refinement example with clear articulation of what was wrong and how the revision fixed it; the refinement is analytically meaningful.
---