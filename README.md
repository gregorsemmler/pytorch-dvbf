# Deep Variational Bayes Filters in PyTorch

This repository contains an implementation of Deep Variational Bayes Filters for image sequences based on the paper [Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data](https://arxiv.org/abs/1605.06432). To be exact, a Deep Variational Bayes Filter with locally linear transitions (DVBF-LL) is implemented. The encoders and decoders are based on convolutional and transposed convolutional layers, respectively.

To try it yourself on an OpenAI gym environment, you can use `keyboard_agent.py` to play some environments by hand and save the frames.

To train, use `training.py` and for evaluation `evaluation.py`.


