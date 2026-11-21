---
title: What Is a Decision Model (System One)?
description: "Language models that think out loud are slow. A decision model skips the writing: give it a situation and a list of options, and it returns a probability for each one in a single pass."
date: 2026-11-21 10:00:00 +0100
series: decision-models
part: 1
tags: [decision models, system one, embeddings, contrastive learning]
cover: assets/img/system_one/system_one_overview.png
giscus_comments: true
---
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/system_one_overview.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    A decision model in one picture: a question and three possible answers go through a frozen language model and two small trained heads, land on a circle, and become probabilities.
</div>

# Welcome

Welcome to a new series! In the [Stable Diffusion series](/blog/series/stable-diffusion/), we took apart a model that *creates*. This time we build a model that *decides*.

Over the last two years, the big story in language models has been reasoning: models that write out a chain of thought before answering, and get better the longer they think. That works remarkably well for hard problems. But a lot of the decisions an AI system makes are not hard problems. An agent choosing its next shell command, a verifier picking the best of four attempts at a bug fix, a game character deciding whether to jump: these are choices between a handful of options, made over and over, often under a time limit. Writing an essay before each one is wasteful.

This series is about a different kind of model, a **decision model**, that looks at a situation and its options and simply *knows* which one fits, in a single pass. We will rebuild one from scratch: the data, the maths, the training, and the results, including the experiments that did not work.

# Two ways of thinking

In *Thinking, Fast and Slow*[^kahneman2011], Daniel Kahneman describes two modes of thought. **System 1** is fast, automatic and effortless: you recognize a friend's face, read the anger in a voice, or know that 2 + 2 = 4 without doing anything. **System 2** is slow, deliberate and effortful: you multiply 17 × 24 in your head, or check a contract clause by clause.

Here is the puzzle Kahneman uses to show the difference:

> A bat and a ball cost \\$1.10 in total. The bat costs \\$1.00 more than the ball. How much does the ball cost?

An answer pops into your head immediately: 10 cents. That is System 1. It is fast, confident, and wrong: if the ball cost 10 cents, the bat would cost \\$1.10 and the total \\$1.20. Slowing down and writing the equation, $$b + (b + 1) = 1.10$$, gives 5 cents. That is System 2.

The lesson is not that System 1 is bad. You use it for almost everything you do, and it is right most of the time. The lesson is that the two systems have different jobs: one is cheap and fast and good enough for most decisions, the other is expensive and careful and saved for the cases that need it.

Language models that write out their reasoning are System 2. Each word of the reasoning is one more full pass through billions of parameters. A decision model is the other half, which is why it is also called a **System One model**: it looks at a situation and its options and answers in one step.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/generate_vs_score.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Generating versus scoring. The generating model has to write its answer one token at a time. The decision model reads the question and scores the three options at once.
</div>

# What a decision model does

A decision model takes two kinds of input. The first is a **state**: the situation the decision is about. It can be a question with a supporting document, the conversation an agent has had so far, or the current screen of a game. The second is a set of **options**: the possible answers or actions, such as three candidate answers to the question, the shell commands an agent could run next, or the moves available in the game.

For each option, the model returns a probability, and it never writes any text. The options have to be given to it: a decision model chooses, it does not invent.

The idea was made popular by [Jev](https://typesafe.ai), a closed-source decision model from TypeSafe. Researchers at Stanford and NVIDIA then released an open one, **CLM-8B**[^clm2026], with its weights and serving code but without the code that trained it. In this series, we try to replicate what CLM-8B did: rebuild its training from the settings stored in its public checkpoints, train our own model, and compare the two. The code is in my [system-one-model repository](https://github.com/MKDYasserH/system-one-model). If you prefer to watch before you read, Vizuara AI has a great [video on building a Jev-like decision model](https://www.youtube.com/watch?v=vEpkc44bHVQ&t=5549s).

# The architecture: two towers

The model has three parts. Here is the whole thing:

```text
State   -> Qwen3-8B (frozen) -> (4096,) -> State head  (9.44M) -> (512,)
Option  -> Qwen3-8B (frozen) -> (4096,) -> Action head (9.44M) -> (512,)
Score   =  s · cos(state embedding, option embedding)
```

Every piece of text, state or option, first goes through Qwen3-8B[^qwen3], an open language model with 8 billion parameters. We keep its weights fixed and never train it. From its output we keep a single vector of dimension 4096, written $$\mathbb{R}^{4096}$$ in maths or shape (4096,) in code, which is simply a list of 4096 numbers that summarizes the text. We call this vector the text's **embedding**. Part 2 explains why we keep the language model frozen, and how one embedding can summarize a whole text.

On top of the language model sit two small networks that we do train. The **state head** maps a state embedding to a new vector of dimension 512, shape (512,), and the **action head** does the same for options. Each head is a small network, 4096 → 1536 → 1536 → 512, with about 9.44 million parameters. Together, the two heads have about 18.9 million parameters, roughly 0.2% of the language model they sit on, and they are the only part of the model that learns anything.

The score of an option is then how well its 512-dimensional vector lines up with the state's, multiplied by a learned number $$s$$.

This design is called a **two-tower model**, or bi-encoder, because states and options go up two separate towers that only meet at the very end, in the score. It is the same design used by CLIP[^radford2021] to match images with captions, and by dense retrieval systems[^karpukhin2020] to match questions with documents.

# The maths of a score

Let's make "how well two vectors line up" precise.

## Cosine similarity

The heads turn a state into a vector $$\mathbf{u}$$ and an option into a vector $$\mathbf{v}$$, both of dimension 512. The **cosine similarity** of the two is

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert} = \frac{\sum_{k=1}^{512} u_k v_k}{\sqrt{\sum_k u_k^2}\,\sqrt{\sum_k v_k^2}}$$

where $$\theta$$ is the angle between the two vectors. When the vectors point the same way, $$\theta = 0$$ and $$\cos\theta = 1$$, a perfect match. When they are at a right angle, the cosine is 0 and the two are unrelated, and when they point in opposite directions it is -1.

Notice what the cosine ignores: the *length* of the vectors. Only their direction matters. So we can drop the lengths from the start by dividing every vector by its length, $$\hat{\mathbf{u}} = \mathbf{u} / \lVert \mathbf{u} \rVert$$. Every vector then has length 1 and lies on the surface of a sphere (a circle, in the two-dimensional pictures of this post), and the cosine becomes a plain dot product:

$$\cos\theta = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$$

This matters in practice: a dot product of two 512-dimensional vectors is 512 multiplications and 511 additions, which a GPU does for thousands of pairs at once.

## From scores to probabilities

A cosine is not yet a probability. To compare the $$n$$ options of one state, we turn their cosines into a probability distribution with a **softmax**:

$$p_i = \frac{e^{\,s \cos\theta_i}}{\sum_{j=1}^{n} e^{\,s \cos\theta_j}}$$

The exponential makes every value positive, and dividing by the sum makes the probabilities add up to 1.

The number $$s$$ is the **scale**, also called the inverse temperature ($$s = 1/\tau$$). Cosines live between -1 and 1, so without it the softmax could never be confident: even a perfect match against an unrelated option would only give $$e^{1}/(e^{1} + e^{0}) \approx 0.73$$. Multiplying by $$s$$ stretches the differences. The model *learns* $$s$$ during training, capped at 100. Part 3 shows how the temperature shapes training.

## A worked example

Take the question in the animation, *"Which planet is closest to the Sun?"*, with three options. Suppose the trained heads give these cosines with the question:

| Option | $$\cos\theta$$ | $$s \cos\theta$$ with $$s = 20$$ | Probability |
|---|---|---|---|
| Mercury | 0.92 | 18.4 | 0.80 |
| Venus | 0.85 | 17.0 | 0.20 |
| Mars | 0.20 | 4.0 | 0.00 |

The probability of Mercury is

$$p_{\text{Mercury}} = \frac{e^{18.4}}{e^{18.4} + e^{17.0} + e^{4.0}} = \frac{1}{1 + e^{-1.4} + e^{-14.4}} \approx 0.80$$

Two things stand out. First, Mars is out of the picture: its cosine is far below the others, and the exponential crushes it to almost zero. Second, Mercury and Venus are close (0.92 against 0.85), so the probability is split 0.80 / 0.20: the model is fairly sure, not certain. The probability is not just a choice, it is a measure of confidence.

I used $$s = 20$$ to keep the numbers readable. The trained model learns a scale of around 90, which makes it much more decisive: with the same cosines and $$s = 90$$, Mercury would get 0.998.

# Why two towers make it fast

There are two other ways to make a language model choose between options, and comparing them shows where the speed comes from.

The first is to **generate the answer**. You ask the model "Which planet is closest to the Sun?" and let it write. The model reads the question, then produces the answer one token at a time, and each token is another pass through all 8 billion parameters. If it reasons first ("Let's think…"), that can be hundreds of passes, and you still have to read its text and match it to an option at the end.

The second is to **score each option together with the state**. You feed the model "question + Mercury", then "question + Venus", then "question + Mars", and compare how likely it finds each. This is accurate, because the model sees the question and the option together, but it costs one full pass per option, over the whole state every time. This design is called a **cross-encoder**[^reimers2019].

A two-tower model avoids both costs. The state goes through the language model **once**. The options go through it once too, and since their embeddings do not depend on the state, any option that comes back (a fixed set of moves, a list of tools, a catalogue of answers) can be embedded once and reused. What is left per decision is the state's pass, two tiny heads, and $$n$$ dot products. There is no decoding at all. In the T-Rex game that ships with the CLM code, a full decision took about 17 milliseconds (median) on a single L40S GPU in our runs, fast enough to play in real time.

The price of this speed is that the state and the options never see each other inside the language model. Everything the model knows about the match has to fit in two independent vectors. That is a real limitation, and it is the reason the training in parts 3 and 4 matters so much: the heads have to learn to put states and their right options close together on the sphere.

# What it can and can't do

A decision model picks among given options quickly and tells you how confident it is. It is also cheap to reuse: the language model is shared across tasks, and adapting to a new one means training, or fine-tuning, only the small heads.

What it can't do is come up with options of its own, follow a multi-step chain of reasoning, or know more than what the frozen language model put into its embeddings.

This is why decision models are best seen as a partner to reasoning models, not a replacement. In Kahneman's picture, System 1 handles the routine and System 2 steps in when something feels off. One natural design does the same: the decision model makes every routine choice, and when its probabilities are spread out, the decision goes to a slower model that thinks it through.

# Implementation

Here is the head, as it is defined in the [repository](https://github.com/MKDYasserH/system-one-model). It is a small multi-layer perceptron: a linear layer to 1536 dimensions, a second linear layer followed by layer normalization, and a final projection to 512, with GELU activations in between.

```python
class ProjectionHead(nn.Module):
    """input_dim -> width -> width (LayerNorm) -> projection_dim, with GELU in between."""

    def __init__(self, dims: HeadDims):
        super().__init__()
        self.inp = nn.Linear(dims.input_dim, dims.width)               # 4096 -> 1536
        self.hidden = nn.ModuleList([nn.Linear(dims.width, dims.width)])  # 1536 -> 1536
        self.norms = nn.ModuleList([nn.LayerNorm(dims.width)])
        self.out = nn.Linear(dims.width, dims.projection_dim)          # 1536 -> 512
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.inp(x))
        x = self.activation(self.norms[0](self.hidden[0](x)))
        return self.out(x)
```

The parameters add up as promised: $$4096 \times 1536 + 1536$$ for the first layer, $$1536 \times 1536 + 1536$$ for the second, $$2 \times 1536$$ for the layer norm and $$1536 \times 512 + 512$$ for the projection, which is 9,443,840, or 9.44M.

The model holds one head for states, one for actions, and the learned scale, stored as a logarithm so that it always stays positive. Choosing between options then takes four lines:

```python
@torch.no_grad()
def choose(model: ContrastiveHeads, state: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
    """state: (1, 4096) embedding. options: (n, 4096) embeddings. Returns one probability per option."""
    u = model.embed_states(state)              # (1, 512), length 1
    v = model.embed_actions(options)           # (n, 512), length 1
    scores = model.logit_scale() * u @ v.T     # s · cos θ for every option
    return scores.softmax(dim=-1)[0]
```

`embed_states` and `embed_actions` run a head and divide by the length, so `u @ v.T` is exactly the cosine from the maths above, and `logit_scale()` is $$s$$.

# What's next

That's the whole model: a frozen language model, two small heads, a cosine and a softmax. The interesting part is everything we have not explained yet: how a frozen model can be useful at all, and how two small heads learn to place a question and its answer next to each other on the sphere. Here is the plan for the series:

1. **What is a decision model?** This post.
2. **Fine-tuning a frozen LLM: why we train heads instead of LoRA.** Why we never touch the 8 billion parameters, and how a frozen model turns text into an embedding.
3. **Contrastive learning.** The loss that trains the heads: InfoNCE, the temperature, and hard negatives.
4. **Training in three stages.** The data, how it was processed, and the pre-, mid- and post-training recipes.
5. **Does it work?** A verifier for coding agents, the results, and what we learned.

# References

[^kahneman2011]: D. Kahneman. *Thinking, Fast and Slow.* Farrar, Straus and Giroux, 2011.
[^clm2026]: J. Kwok, H. Kang, T. Suresh, J. Saad-Falcon, M. Pavone, C. Ré and A. Mirhoseini. *CLM-8B* (project page). [contrastive-lm.notion.site](https://contrastive-lm.notion.site). Code: [github.com/Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM).
[^qwen3]: Qwen Team. *Qwen3 Technical Report.* [arXiv:2505.09388](https://arxiv.org/abs/2505.09388), 2025.
[^radford2021]: A. Radford et al. *Learning transferable visual models from natural language supervision.* International Conference on Machine Learning, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
[^karpukhin2020]: V. Karpukhin et al. *Dense passage retrieval for open-domain question answering.* Empirical Methods in Natural Language Processing, 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
[^reimers2019]: N. Reimers and I. Gurevych. *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* Empirical Methods in Natural Language Processing, 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
