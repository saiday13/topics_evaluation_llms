# Topic Model Evaluation with Large Language Models
The aim of this project is to evaluate the correlation between the automated topic model and large language models (LLMs) evaluations. Following the work of Hoyle et al. [Is Automated Topic Model Evaluation Broken?:
The Incoherence of Coherence](https://arxiv.org/pdf/2107.02173), we replace the human evaluation with LLMs evaluation in two tasks: intrusion detection and rating.

To run the project, first download the [topics](https://github.com/ahoho/topics?tab=readme-ov-file) and follow the installation instructions there.

The automated topic models used for evaluation are - one classical and two neural- LDA, DVAE and ETM.

The LLMs used for evaluation are GPT-2, GPT-3, GPT-3.5 and Bloom. 
Later, GPT-2 model was exclused from further evaluation due to incoherent results, while GPT-3.5 was not considered in experiments due to problems with subscription.

The datasets used for experiments are 20Newsgroups and WikiText-103.

The final report you can find [hier]()


```
@inproceedings{hoyle-etal-2021-automated,
    title = "Is Automated Topic Evaluation Broken? The Incoherence of Coherence",
    author = "Hoyle, Alexander Miserlis  and
      Goel, Pranav  and
      Hian-Cheong, Andrew and
      Peskov, Denis and
      Boyd-Graber, Jordan and
      Resnik, Philip",
    booktitle = "Advances in Neural Information Processing Systems",
    year = "2021",
    url = "https://arxiv.org/abs/2107.02173",
}
```
