
# Activation Steering

This repository is an exploration of LLM activation steering/representation engineering, as described in [this LessWrong post](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) and [this arXiv paper](https://arxiv.org/pdf/2310.01405), among other places. Code and datasets generously made available at [nrimsky/CAA](https://github.com/nrimsky/CAA/) and [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) were used as starting points.

The concept of activation steering is simple, and it is remarkable that it works. You identify an activation pattern in a model (generally in the residual stream input or output) corresponding to a high-level behavior like "sycophancy" or "honesty" by running pairs of inputs with and without the behavior through the model, create a behavior vector from the differences in the pairs' activations, then add that vector, scaled by +/- various coefficients, to the model's activations as it generates new output, and you get output that has more or less of the behavior, as you desire. This would seem quite interesting from the perspective of LLM interpretability, and potentially safety.

Beneath that simplicity, there are a lot of details, with a number of differing approaches having been reported and many more possible, and this work is intended to explore a number of them.

`enhanced_hooking.py` implements activation inspection and manipulation in a generalized, flexible way that can target arbitrary model layers and sequence positions independently, and can handle different decoder architectures.
