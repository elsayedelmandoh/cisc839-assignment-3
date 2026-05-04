## Part 2: Text-based Label Prediction (40%)
Use the cluster labels from Part 1 as the ground truth for a supervised text classification task. Your goal is to predict which effort group a PR belongs to using the description of the pull request.

### Task 2.1 (20%): Classification Pipeline 

Describe and justify your text representation and classification pipeline:

- Text source: You will use PR descriptions as the text input. Discuss what linguistic or topical signals in descriptions you expect to correlate with different effort clusters, and whether any preprocessing is needed.
- Classification methodology: Describe your approach for learning the representation from text (e.g., TF-IDF, word embeddings, pre-trained transformer embedding models, fine-tuned language models) and classify the learned representation. The approach should be well justified (even with experimental results supporting). You could use existing SOTA text classification models.