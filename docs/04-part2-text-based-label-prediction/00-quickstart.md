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
