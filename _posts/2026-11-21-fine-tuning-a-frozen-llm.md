---
title: "Fine-Tuning a Frozen LLM: Why We Train Heads Instead of LoRA"
description: "Our decision model never changes a single weight of its 8-billion-parameter language model. Here is why training two small heads on frozen embeddings suits this job better than LoRA, and how a frozen model turns text into a vector."
date: 2026-11-21 11:00:00 +0100
series: decision-models
part: 2
tags: [decision models, fine-tuning, lora, embeddings]
cover: assets/img/system_one/fine_tuning_spectrum.png
giscus_comments: true
---
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/fine_tuning_spectrum.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Four ways to adapt an 8-billion-parameter model to a new task. Besides how many parameters each one trains, they differ in how far the gradient has to travel.
</div>

# Introduction

In the [previous post](/blog/2026/what-is-a-decision-model/), we met the decision model: a frozen language model, Qwen3-8B, that turns every text into an embedding, and two small heads that we train to place states and their right options close together. I said we never change a single weight of the language model, and promised to explain why.

That choice deserves a post of its own, because it goes against a common instinct. When you want a large language model to do something new, you fine-tune it, and today that often means LoRA, a method that has made fine-tuning affordable for almost everyone. Our model does something simpler: it treats the language model as a fixed function, computes its output once, and trains a small network on top. Both are good tools. This post explains why, for this particular job, the simpler one is the better fit, and understanding why says a lot about where the cost of training comes from.

The second half of the post looks at the frozen model itself: how a language model that was trained to predict the next word can produce one vector that summarizes a whole text, and what we do to that vector before the heads see it.

# Four ways to adapt a language model

Suppose you have a pretrained language model and a new task. There are four broad ways to adapt it, and they differ in which parameters you let training change.

The most direct is **full fine-tuning**: you keep training the whole model on your task, all 8.2 billion parameters of Qwen3-8B. This is the most flexible option, and the most expensive. Training with the Adam optimizer keeps several numbers for every parameter: the weight itself, its gradient, a full-precision copy of the weight, and two running averages that Adam uses to adapt the step size. In standard mixed-precision training, that adds up to about 16 bytes per parameter (2 for the weight, 2 for the gradient, 4 for the copy and 4 for each average). For Qwen3-8B that is around 130 GB before counting a single activation, more than any single GPU holds.

**LoRA**[^hu2021], short for low-rank adaptation, freezes the model and learns a small correction next to some of its weight matrices. Take a layer that multiplies its input $$x$$ by a weight matrix $$W$$ of shape $$(d_{\text{out}}, d_{\text{in}})$$. Instead of changing $$W$$, LoRA learns two thin matrices, $$B$$ of shape $$(d_{\text{out}}, r)$$ and $$A$$ of shape $$(r, d_{\text{in}})$$, and the layer computes

$$W' x = W x + B A x$$

where $$W$$ stays frozen, $$x$$ is the layer's input, and the rank $$r$$ is a small number, typically 8 to 64. The product $$BA$$ has the same shape as $$W$$, so it can correct any part of it, but it only costs $$r(d_{\text{in}} + d_{\text{out}})$$ parameters: $$d_{\text{out}} \times r$$ for $$B$$ plus $$r \times d_{\text{in}}$$ for $$A$$. Applied with $$r = 16$$ to the four attention matrices of each of Qwen3-8B's 36 layers, that is about 15.3 million trained parameters, less than 0.2% of the model. Adapters[^houlsby2019] follow the same idea with small extra layers instead of low-rank corrections.

**Prompt tuning**[^lester2021] and prefix tuning[^li2021] go even lighter: the model is frozen, and training only learns a few vectors that are placed in front of the input, like a prompt written in the model's own language instead of in words. Twenty such vectors of dimension 4096 are about 82 thousand parameters.

The last option is the one we use: keep the model frozen, use its output as **features**, and train a separate small network on top of them. In its simplest form, a single linear layer, this is called a linear probe[^alain2016], and it is a classic way to test what a pretrained model already knows. Our version uses two heads with 18.9 million parameters in total.

Look at the parameter counts again: 8.2 billion, 15.3 million, 82 thousand, 18.9 million. Our heads train slightly *more* parameters than LoRA does. So the number of trained parameters cannot be what separates the two for us. What separates them is where those parameters sit.

# What training costs

Training a network means two passes over every example. The forward pass computes the output and the loss, and the backward pass sends the gradient of the loss back through the network, layer by layer, to find how each trained parameter should change.

A useful rule of thumb[^kaplan2020] gives the cost of these passes. In a forward pass, each parameter takes part in about one multiplication and one addition for every token, so a model with $$N$$ parameters costs about $$2N$$ floating-point operations (FLOPs) per token. The backward pass costs about twice as much, $$4N$$, because it computes two gradients at every layer: one with respect to the layer's input, to keep passing the gradient down, and one with respect to its weights, to update them.

The backward pass has to travel through every layer between the loss and the parameters being trained. With LoRA, the trained matrices sit inside all 36 layers of the language model, so the gradient has to go back through the whole network to reach them, even though only 15 million parameters get updated at the end. The same is true for prompt tuning, whose trained vectors sit at the very bottom, below every layer. In the animation above, the arrows on the first three panels show that path. This is not a flaw of LoRA: it is the price of changing what happens *inside* the model, which is exactly what LoRA is for.

With frozen features, the language model sits *before* the training loop. The gradient starts at the loss, goes back through the two small heads, and stops. The 8 billion parameters never take part in training.

Let's put numbers on it. Take a training example made of a state and an action of about 500 tokens each (the exact length changes the numbers, not the conclusion). The language model has $$N = 8.2$$ billion parameters, that is $$N = 8.2 \times 10^9$$, and each text has $$T = 500$$ tokens.

With LoRA, each text needs a forward pass through the language model, $$2NT$$ FLOPs, and a backward pass. The backward pass skips the weight gradients of the frozen matrices and only computes the input gradients, which costs about another $$2NT$$. So each text costs about $$4NT$$, and an example, with its two texts, about twice that:

$$\text{LoRA, per example} \approx 2 \times 4NT = 8 \times (8.2 \times 10^9) \times 500 \approx 3.3 \times 10^{13} \text{ FLOPs}$$

With frozen embeddings, an example is two vectors, each going through one head with 9.44 million parameters, forward ($$2 \times$$) and backward ($$4 \times$$), so $$6 \times$$ the parameters per vector, with no tokens involved since each text is already a single vector:

$$\text{Heads, per example} \approx 6 \times (18.9 \times 10^6) \approx 1.1 \times 10^8 \text{ FLOPs}$$

That is a ratio of about **300,000** per example. What it means in time depends on the hardware, so here is our pre-training run, 30,000 steps of 16,384 examples (about 490 million examples), on a few common setups. The times use each GPU's peak dense BF16 throughput from the vendor's specifications. Real runs reach perhaps a third to a half of that, so actual times would be two to three times longer.

| Setup | Peak throughput | LoRA through the 8B model | Heads on frozen embeddings |
|---|---|---|---|
| 1 × L40S | 181 TFLOP/s | about 2.8 years | about 5 minutes |
| 8 × A100 | 8 × 312 TFLOP/s | about 74 days | about 22 seconds |
| 8 × H100 | 8 × 989 TFLOP/s | about 23 days | about 7 seconds |
| 8 × B200 (Blackwell) | 8 × 2,250 TFLOP/s | about 10 days | about 3 seconds |

A TFLOP/s is a trillion ($$10^{12}$$) floating-point operations per second. In our run, on a single L40S, a thousand steps took about four minutes, and the whole run of 30,000 steps took two hours, evaluations included: at this scale, the real work is moving a terabyte of embeddings from disk to the GPU, not the arithmetic.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/embed_once_train_many.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    With LoRA, every training step runs the language model forward and backward. With a frozen language model, the texts are embedded once, and training only ever reads the stored embeddings.
</div>

# Embed once, train many times

The frozen model has a second advantage, and it is the one that makes everything else possible. Since the language model never changes, the embedding of a text never changes either. We can compute it once, store it, and reuse it for every epoch, every training stage and every experiment.

That is how the training data for this project is distributed. The CLM authors ran Qwen3-8B over 62.45 million question-and-answer pairs and published the embeddings as a dataset. Each embedding has shape (4096,) and is stored in half precision, 2 bytes per number, so one text takes 8 KB and the whole dataset, a state and an action per pair, takes about 1 TB.

Producing it took one forward pass per text: $$62.45 \times 10^6$$ pairs, times 2 texts, times $$2NT$$ FLOPs per text, about $$10^{21}$$ FLOPs in total. At peak throughput, that is about 64 days on a single L40S, 4.6 days on 8 A100s, a day and a half on 8 H100s, or about 15 hours on 8 Blackwell B200s. That cost was paid once, by the people who released the data, and every training run since has reused it.

With LoRA, nothing can be stored. The adapters change at every step, so the embedding of the same text is different at every step, and the language model has to run again each time it sees the text. Every epoch, every stage and every new experiment pays the full price again.

Our own experiments show what this means. When we tried new post-training recipes, the whole study, including embedding the new data, training several variants and running the benchmarks, cost \\$2.85 in cloud compute. The two training sweeps themselves, once the embeddings existed, cost less than four cents. When trying an idea is almost free, you try more ideas.

# Why contrastive learning needs this

There is a third reason, specific to how the heads are trained. Part 3 will cover the loss in detail, but here is the short version: the model learns by looking at a batch of states and actions, and for every state, every other action in the batch counts as a wrong answer. A batch of 16,384 examples gives each state 16,383 wrong answers to beat, and more wrong answers make for a harder, more informative task. Contrastive learning works best with very large batches.

For the heads, a batch of 16,384 is 16,384 pairs of vectors, a few hundred megabytes. For LoRA, it would be 16,384 pairs of texts going through the 8-billion-parameter model at once, with every intermediate activation kept in memory for the backward pass: millions of tokens' worth of activations across 36 layers, far beyond what a GPU holds. There are clever workarounds that split a large contrastive batch into chunks[^gao2021], but they add complexity and do not reduce the compute.

# When LoRA is the better choice

None of this makes LoRA a worse method. It makes it a different one, and there are many jobs where it is the right call.

If you want the model to *generate* differently, write in a new style, follow a new format or learn a new skill, the change has to happen inside the model, and frozen features cannot help: the heads only read the embedding, they never change what the model writes. LoRA is also the better choice when the information you need is not in the frozen representation at all. Our heads can only rearrange what Qwen3-8B already puts into its 4096 dimensions; if the difference between two answers is not there, no head can recover it, while LoRA can change what the model pays attention to. And when the dataset is small and you train for one or two epochs, there is little to gain from storing embeddings, and LoRA's cost per example matters much less.

Our task sits at the other end. It scores options instead of generating text, it trains on tens of millions of pairs for many epochs and several stages, it needs very large batches, and we wanted to try many recipes cheaply. Those are exactly the conditions where computing the embeddings once pays off.

There is also a subtler limit on our side. Qwen3-8B was trained to predict the next word, not to produce embeddings, and its last hidden state was never meant to summarize a text for retrieval. The heads' job is to find, inside that state, the directions that matter for matching states with actions. The results in part 5 show they find enough of them to come close to the released CLM-8B, and to beat it on some tests, but a model trained end to end for embeddings could do better.

In exchange, the frozen setup is cheap to serve. One copy of the language model can serve any number of tasks, each with its own pair of heads. The DeepSWE verifier we will meet in part 5 is just another pair of heads: the file is 75 MB, against more than 16 GB for the language model.

## Intuition

Picture an expert with broad knowledge, and a specialist trained for one kind of decision.

LoRA sends the expert on a short course, so he comes back thinking about your problem a little differently. That can change his judgment deeply, but every new idea means another course.

Our approach leaves the expert as he is and asks him to read every case once and write careful notes. The specialist then learns to decide from those notes: his skill is the part that knows the task. The efficiency comes from the notes being written once and reused, so each new task gets its own specialist without taking the expert's time again. The one limit: the specialist can only use what the notes contain, and an 8-billion-parameter expert writes very detailed notes.

# How a frozen model turns text into a vector

So far we have treated "the embedding" as given. Let's see where it comes from.

A language model like Qwen3-8B reads a text as a sequence of tokens and computes, for every token, a hidden state of dimension 4096 at every layer. A hidden state is the model's internal representation of a token at that point of the network, a vector that the next layer reads and refines. After the last layer, each token has a final hidden state, normally used to predict the next token. We want one vector for the whole text, so we have to pick, or combine, some of these states.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/causal_last_token.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    The causal attention mask. Each token can look only at itself and the tokens before it, so only the last token's hidden state is computed from the whole text.
</div>

The answer comes from how the model reads. Qwen3-8B is a *causal* language model: when it computes the hidden state of a token, attention lets it look only at that token and the ones before it, never at the ones after. This is what allows it to generate text one token at a time. The animation shows the resulting mask: the hidden state of "planet" knows about "Which" and "planet", but nothing about "Sun".

Only one token has seen everything: the last one. Its final hidden state is computed from the entire text, so it is the natural summary. This is called **last-token pooling**, and it is the recipe behind the embeddings we use. Formally, for a text of $$T$$ tokens $$x_1, \dots, x_T$$, the embedding is the final-layer hidden state of the last token:

$$\mathbf{e} = \mathbf{h}^{(L)}_T(x_1, \dots, x_T) \in \mathbb{R}^{4096}$$

where $$\mathbf{e}$$ is the embedding of the text, $$x_1, \dots, x_T$$ are its tokens, and $$\mathbf{h}^{(L)}_T$$ is the hidden state of the last token, number $$T$$, after the last layer, number $$L$$ (Qwen3-8B has $$L = 36$$ layers). The notation $$(x_1, \dots, x_T)$$ is a reminder that this state depends on every token of the text.

An alternative you may know from sentence embeddings is mean pooling, which averages the hidden states of all the tokens. It works well with models like BERT[^reimers2019], where every token sees the whole text. In a causal model, early tokens have only seen the beginning, so averaging mixes complete and very partial views. Last-token pooling is the usual choice for causal models used as embedders[^ma2023].

## Normalizing the embedding

The last hidden state can have any length, and that length carries little meaning for matching. As in the previous post, we divide it by its length:

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\lVert \mathbf{e} \rVert}$$

where $$\lVert \mathbf{e} \rVert = \sqrt{e_1^2 + \dots + e_{4096}^2}$$ is the length of the embedding, and $$\hat{\mathbf{e}}$$ is the normalized embedding: same direction, length 1.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/normalize_to_sphere.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Dividing each embedding by its length keeps only its direction, so every embedding lands on the unit sphere and the cosine between two of them is a plain dot product.
</div>

Every embedding now lies on the unit sphere of $$\mathbb{R}^{4096}$$. This is done once, when the embeddings are computed, and the result is stored in half precision. The heads then map these normalized embeddings to dimension 512 and normalize again, so the final score is a cosine, as we saw in part 1.

## States, actions and long texts

The question-and-answer data came already embedded. For the agent data used in post-training, we computed the embeddings ourselves, and had to follow the same recipe exactly: an embedding computed differently would land in a different place on the sphere, and the heads trained on the published data would not understand it.

Two details matter there. The first is formatting. An agent's state is a conversation: a system prompt, a task, the agent's previous actions and what the environment answered. Qwen3-8B was trained on conversations written in a specific chat format, where each turn is wrapped in special tokens marking who is speaking, so we write the state in that format before embedding it. An action, like an answer, is embedded as plain text.

The second is length. We run the model with a context of 2048 tokens and keep at most 2047 tokens per text. When a state is longer, we keep its **end**: the most recent observations matter most for choosing the next action. When an action is longer, we keep its **beginning**, because the start of an action says what the agent is doing.

# Implementation

Here is how the [repository](https://github.com/MKDYasserH/system-one-model) embeds the agent data. The language model is loaded with vLLM in its "pooling" mode, which runs one forward pass per text and returns the last token's hidden state instead of generating anything:

```python
self.llm = LLM(
    model=str(model_dir), runner="pooling", max_model_len=settings.max_model_len,
    gpu_memory_utilization=settings.gpu_memory_utilization,
)
```

A state is written in Qwen's chat format and cut to its newest tokens. An action keeps its first tokens:

```python
def _state_tokens(self, state: list[ChatMessage]) -> list[int]:
    """The state in Qwen's chat format (<|im_start|>role ... <|im_end|>), keeping its newest tokens."""
    from jinja2 import TemplateError

    messages = [message.model_dump() for message in recent_messages(state, settings.state_char_budget)]
    try:
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    except TemplateError:
        # Qwen's template refuses some role orders (e.g. no user turn left after the cut): embed plain text.
        tokens = self.tokenizer("\n\n".join(m["content"] for m in messages), add_special_tokens=False)["input_ids"]
    return list(tokens)[-settings.max_tokens:]

def _action_tokens(self, action: str) -> list[int]:
    """Plain text, like the Q&A answers. The start of an action says what the agent does."""
    return self.tokenizer(action, add_special_tokens=False)["input_ids"][: settings.max_tokens]
```

Finally, the texts go to the model a few thousand at a time, and the embeddings are normalized and stored in half precision:

```python
def _embed(self, token_lists: list[list[int]]) -> np.ndarray:
    """(len(token_lists), 4096). Texts go to vLLM texts_per_call at a time to bound memory."""
    from vllm.inputs import TokensPrompt

    parts = []
    for texts in batched(token_lists, settings.texts_per_call):
        # TokensPrompt: already tokenized, so vLLM embeds exactly the tokens we kept.
        outputs = self.llm.embed([TokensPrompt(prompt_token_ids=tokens) for tokens in texts], use_tqdm=False)
        parts.append(np.stack([np.asarray(output.outputs.embedding, dtype=np.float32) for output in outputs]))
    return l2_normalize(np.concatenate(parts))

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is a cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / (norms + 1e-12)).astype(np.float16)
```

This is the only place in the whole project where the language model runs. Everything after it, every training stage and every experiment, only reads the stored embeddings.

# What's next

We now have the inputs: one normalized embedding per text, computed once by a frozen model, and two small heads waiting to be trained on top. What we have not seen is how the heads learn. What tells them that "Mercury" belongs near "Which planet is closest to the Sun?" and "Mars" does not? That is the job of the loss, and of the many wrong answers each batch provides. Part 3 is about contrastive learning: InfoNCE, the temperature, and the hard negatives that make the model sharp.

# References

[^hu2021]: E. J. Hu et al. *LoRA: Low-rank adaptation of large language models.* International Conference on Learning Representations, 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
[^houlsby2019]: N. Houlsby et al. *Parameter-efficient transfer learning for NLP.* International Conference on Machine Learning, 2019. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
[^lester2021]: B. Lester, R. Al-Rfou and N. Constant. *The power of scale for parameter-efficient prompt tuning.* Empirical Methods in Natural Language Processing, 2021. [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
[^li2021]: X. L. Li and P. Liang. *Prefix-tuning: Optimizing continuous prompts for generation.* Association for Computational Linguistics, 2021. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
[^alain2016]: G. Alain and Y. Bengio. *Understanding intermediate layers using linear classifier probes.* [arXiv:1610.01644](https://arxiv.org/abs/1610.01644), 2016.
[^kaplan2020]: J. Kaplan et al. *Scaling laws for neural language models.* [arXiv:2001.08361](https://arxiv.org/abs/2001.08361), 2020.
[^gao2021]: L. Gao, Y. Zhang, J. Han and J. Callan. *Scaling deep contrastive learning batch size under memory limited setup.* Workshop on Representation Learning for NLP, 2021. [arXiv:2101.06983](https://arxiv.org/abs/2101.06983)
[^reimers2019]: N. Reimers and I. Gurevych. *Sentence-BERT: Sentence embeddings using Siamese BERT-networks.* Empirical Methods in Natural Language Processing, 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
[^ma2023]: X. Ma, L. Wang, N. Yang, F. Wei and J. Lin. *Fine-tuning LLaMA for multi-stage text retrieval.* [arXiv:2310.08319](https://arxiv.org/abs/2310.08319), 2023.
