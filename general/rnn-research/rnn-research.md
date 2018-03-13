# The Importance of Being Recurrent for Modeling Hierarchical Structure

[paper](https://arxiv.org/pdf/1803.03585.pdf) -- no bibtex reference yet

Comparison of true recurrent (RNN + variants) vs non-recurrent (transformer network - fully attention (FAN), CNNs)

FANs claim to be more interpretable through visualization of attention weights

FANs achieve lower perplexity on the subject-verb agreement task, but LSTMs predict more accurately.

"The  lack of correlation  between  perplexity  and agreement accuracy  indicates  that FANs might capture other aspects of language better than LSTMs. We leave this question to future work."

"Why artificial data?
Despite the simplicity of the
language, this task is not trivial. To correctly clas-
sify logical relations, the model must learn nested
structures  as  well  as  the  scope  of  logical  oper-
ations.    We  verify  the  difficulty  of  the  task  by
training  three  bag-of-words  models  followed by
sum/average/max-pooling.  The best of the three
models achieve less than 59% accuracy on the log-
ical inference versus 77% on the Stanford Natu-
ral Language Inference (SNLI) corpus (Bowman
et al., 2015a). This shows that the SNLI task can be
largely solved by exploiting shallow features with-
out understanding the underlying linguistic struc-
tures."