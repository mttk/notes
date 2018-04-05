## Learning to generate reviews and discovering sentiment

Radford, Alec, Rafal Jozefowicz, and Ilya Sutskever. "Learning to generate reviews and discovering sentiment." arXiv preprint arXiv:1704.01444 (2017).

**Data:** Binary SST 

**Score:** @ACC 91.8 

**Model:** Byte mLSTM

**Notes:** Single layer mLSTM network with 4096 units with a logistic regression classifier on top. L1 penalty for _small_ datasets. "we use an L1 penalty for text classification results instead of L2 as we found this performed better in the very low data regime".

