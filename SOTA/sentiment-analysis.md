## [Learning to generate reviews and discovering sentiment](https://arxiv.org/pdf/1704.01444.pdf)

Radford, Alec, Rafal Jozefowicz, and Ilya Sutskever. "Learning to generate reviews and discovering sentiment." arXiv preprint arXiv:1704.01444 (2017).

**Data:** Binary SST 

**Score:** [91.8] Accuracy

**Model:** Byte (char) mLSTM

**Notes:** Single layer mLSTM language model pre-trained on a large Amazon customer review corpus with 4096 units. Used as a feature extractor for a logistic regression classifier. L1 penalty, _"we use an L1 penalty for text classification results instead of L2 as we found this performed better in the very low data regime"._


