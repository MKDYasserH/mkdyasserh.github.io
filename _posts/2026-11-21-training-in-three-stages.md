---
title: "Training a Decision Model: The Data, and Pre-, Mid- and Post-Training"
description: "Where the training pairs come from, how a terabyte of vectors is fed to one GPU, the three training stages, and the general post-training recipe that worked best: terminal steps, agent environments, Q&A replay and answer preferences with critiques."
date: 2026-11-21 13:00:00 +0100
series: decision-models
part: 4
tags: [decision models, training, curriculum, preference learning, agents]
cover: assets/img/system_one/stage_curriculum.png
giscus_comments: true
---
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/stage_curriculum.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    The three stages, with the data each one trains on and two examples of it. Each stage starts from the best checkpoint of the previous one. The examples are illustrative, except the two post-training questions, which come from our data.
</div>

# Introduction

The [previous post](/blog/2026/contrastive-learning/) gave us the learning machinery: a batch of pairs, a similarity matrix, the InfoNCE loss, a learned temperature and hard negatives. A loss, however, is only half of training. The other half is what the model sees, in which order, and how we decide that a checkpoint is good enough to keep.

The heads are trained in three stages. **Pre-training** teaches them, from scratch, to put 62 million questions next to their answers. **Mid-training** sharpens them on a smaller set of questions that come with tricky wrong answers. **Post-training** turns them toward what a decision model is actually for: agents choosing their next action, and judging which of two answers is better.

This post follows that path from the data on disk to the last checkpoint. We start with the data and how it reaches the GPU, go through the three stages one by one, and spend the most time on the last one: the general post-training recipe that gave our final checkpoint. The results themselves are the subject of part 5.

# The data

## 62 million question-and-answer pairs

The first two stages train on question-and-answer pairs from NVIDIA's **Nemotron-CC v2.1**[^nemotron], a large pretraining corpus built from web pages. It includes a subset called **High-Quality-DQA**: short STEM questions with their answers, which the language model Qwen3-30B-A3B wrote from high-quality science and technology web documents, following the "diverse QA" recipe of the original Nemotron-CC[^su2024]. Each document gives several pairs, which is why the same-document mask of part 3 is needed.

The CLM authors[^clm] took these pairs and, as we saw in [part 2](/blog/2026/fine-tuning-a-frozen-llm/), released them not as texts but as embeddings: they ran every question and every answer through the frozen Qwen3-8B. That is the version we train on. There are 62.45 million pairs, each a state embedding and an action embedding of shape (4096,), stored in half precision.

The pairs are cut into **chunks** of 100,000, about 625 in total, spread over six shards. A chunk is three files: the 100,000 question vectors, the 100,000 answer vectors, and a small table that records which document each pair came from, the `doc_id` that the same-document mask of part 3 needs. Each vector file has shape (100000, 4096) and weighs about 0.8 GB, so a chunk is about 1.6 GB and the whole dataset about 1 TB.

The first shard, `shard_00`, also has **hard negatives**: for each pair, up to 10 wrong answers that the CLM authors generated with Gemini to be close to the right one, then embedded like everything else, stored as a file of shape (100000, 10, 4096), with a second file that counts how many of the 10 slots are real. That is about 8.2 GB per chunk and 0.25 TB in all, for about 3 million pairs and 26.7 million hard negatives, roughly 9 per pair.

Four chunks are set aside and never trained on, the same four the authors held out: two ordinary chunks, `shard_05/0083` and `0084`, to measure the in-batch loss, and two chunks with hard negatives, `shard_00/0028` and `0029`, to measure how often the right answer beats its tricky alternatives. Of their 200,000 pairs, 191,161 have at least one hard negative, and this is the "tricky Q&A" test that comes back in part 5.

## Feeding a terabyte to one GPU

All our training ran on a single GPU, an NVIDIA L40S with 48 GB of memory, in a machine with 256 GB of ordinary memory (RAM). The Q&A data, about 1 TB, sits on a network disk. So the data is four times larger than the RAM, and twenty times larger than the GPU's memory: it can never be loaded all at once. It has to be brought in piece by piece, and in a way that never leaves the GPU waiting.

Two facts shape the solution. The first is that a training step is fast: the heads are small, so the GPU finishes a batch of 16,384 pairs in a fraction of a second. Reading from the disk is slow by comparison: loading a single 1.6 GB chunk takes far longer than a step. If every batch were read straight from the disk, the GPU would spend most of its time idle. The second is that a batch should mix pairs from all over the data, so that the wrong answers it provides for free are varied.

The solution is a small pipeline with two workers running side by side, a **reader** and a **dealer**, and a buffer of data in RAM between them, called a **pool**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/reader_dealer.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    The reader loads 24 chunks into a pool in RAM. The dealer shuffles the pool and sends it to the GPU batch after batch, one training step per batch, until every pair has been used once. Then it reshuffles the same pool and goes through it again, three times in all, while the reader loads the next pool.
</div>

The **reader** picks 24 chunks, in a random order that changes every epoch, loads them from the disk and joins them into one pool: 2.4 million pairs, about 39 GB of RAM.

The **dealer** then shuffles the pool, meaning it puts its 2.4 million pairs in a random order, and cuts that order into consecutive batches of 16,384 pairs. A pool of 2.4 million pairs gives 146 batches (the few pairs left over at the end wait for the next shuffle). The dealer sends these batches to the GPU **one after another**, and each one is one training step: the GPU computes the loss on the batch, and the heads are updated, before the next batch arrives. After 146 steps, every pair of the pool has been used exactly once. This is one **pass** over the pool.

Only then does the dealer shuffle the same pool again. The new random order puts every pair in a different batch, next to different neighbours, so it meets different wrong answers than in the first pass. The dealer cuts it into 146 new batches and sends them one after another, as before. It does this three times in total, so one pool gives 3 × 146 = 438 training steps. For the first pool:

| Pool | Pass | Training steps |
|---|---|---|
| 1 (chunks 1–24) | 1st shuffle | 1 to 146 |
| 1 | 2nd shuffle | 147 to 292 |
| 1 | 3rd shuffle | 293 to 438 |
| 2 (chunks 25–48) | 1st shuffle | 439 to 584 |

Meanwhile, the reader has not been idle. As soon as it handed over the first pool, it started loading the next 24 chunks, so the second pool is usually ready by the time the dealer finishes with the first one, and the GPU moves on without waiting. Reusing each pool three times, with a fresh shuffle each time, is what gives the reader enough time to do this.

Two queues connect the workers, and both are kept small: at most one finished pool waits for the dealer, and at most a few batches wait for the GPU. However large the dataset, RAM never holds more than three pools at once: the one being dealt, one waiting, and the one being loaded, about 120 GB.

Over a whole epoch, the 621 training chunks make 26 pools, so an epoch is about 11,300 steps, and every pair is seen three times, each time in a different batch. Our pre-training run stops at 30,000 steps, a little over two and a half epochs.

# Stage 1: pre-training

Pre-training starts from two randomly initialized heads and trains them on every chunk except the four held-out ones, with the symmetric InfoNCE loss and the same-document mask of part 3. There are no hard negatives yet: every wrong answer comes from the batch.

The optimizer is AdamW[^loshchilov2019] with no weight decay, gradients are clipped to a norm of 1, and the temperature starts at 0.07 and is learned. The learning rate is constant, and its value comes from a simple scaling rule:

$$\eta = \eta_0 \sqrt{\frac{1024}{w}} \sqrt{\frac{B}{1024}} = 0.002 \times \sqrt{\frac{1024}{1536}} \times \sqrt{16} \approx 0.00653$$

where $$\eta$$ is the learning rate, $$\eta_0 = 0.002$$ is a base rate for a reference setup with layers of width 1024 and batches of 1024, $$w = 1536$$ is the width of the heads' hidden layers, and $$B = 16{,}384$$ is the batch size. The first factor lowers the rate for wider layers, where each update moves the output more. The second raises it for larger batches, whose gradients are less noisy and can be trusted with larger steps. The square root, rather than a straight proportion, is the rule that works for Adam-style optimizers[^malladi2022].

Every 1,000 steps, the trainer measures the in-batch loss on the two held-out ordinary chunks. Whenever this loss is the lowest so far, the weights are saved as the stage's best checkpoint. If nine evaluations in a row bring no improvement, training stops early; otherwise it stops at 30,000 steps. A second checkpoint, overwritten at every evaluation, holds the weights together with the optimizer and the step counter, so a run that crashes resumes exactly where it was.

At this stage the heads learn the broad geometry of questions and answers: a question about planets lands near answers about planets, a question about cooking near answers about cooking. As we saw in part 3, the loss falls from its chance level of about 9.7 to about 1.9 within 8,000 steps.

# Stage 2: mid-training

Mid-training starts from the best pre-training checkpoint and adds the hard negatives. Only `shard_00` has them, so this stage trains on its 28 non-held-out chunks, about 2.8 million pairs. Each state's hard negatives become extra columns in its row of the similarity matrix, exactly as in part 3.

A few settings change with the data. The learning rate drops to 0.001, because the heads are no longer random and need adjusting rather than building. Pools hold 6 chunks instead of 24, since a chunk with its hard negatives is about 10 GB, and each pool is shuffled once. An epoch is now only about 168 steps, so the trainer evaluates every 42 steps, four times per epoch, and this time it keeps the checkpoint with the lowest loss on the **hard-negative** held-out chunks. Our best mid-training checkpoint came at step 546, a little over three epochs in.

Why not use hard negatives from the start? One reason is practical: they exist for fewer than 5% of the pairs, so pre-training on them alone would throw away most of the data. The other is that they are only useful once the model knows the basics. To a random model, every answer is equally confusing, and the subtle difference between a right answer and an almost-right one is lost in the noise. Training on easy examples first and harder ones later is an old idea called **curriculum learning**[^bengio2009], and it is what the split between pre- and mid-training does.

## Intuition

Think of a medical student. In his first years, he reads the whole library: anatomy, chemistry, hundreds of diseases, without going deep into any of them. That is pre-training. Then come the exam questions written to trap him, where two diagnoses fit the symptoms and only a detail separates them. These would have been useless in his first week, when everything was new, but now they are exactly what he needs. That is mid-training. What remains is the residency, where he applies all this to real patients, which is what post-training does for our model.

# Stage 3: post-training

After mid-training, the heads are very good at one task: matching a question with its answer. A decision model has to do more. An agent's state is a long history of commands and outputs, its options are actions rather than answers, and sometimes the question is not "which option is right?" but "which of these two is better?". Post-training adapts the heads to these situations.

Our first post-training run reproduced the original recipe: about 847,000 agent decisions, filling 60% of each batch, with the other 40% replayed from the Q&A chunks with hard negatives, for 400 steps. It was a useful baseline, but it left two problems. The first is the wrong answers. In a batch that mixes dozens of agent datasets, the wrong actions for a shopping state are mostly shell commands, SQL queries and household moves, which the model can reject at a glance, so the batch says little about what a decision model most needs: telling apart two actions that both make sense in the situation. The second is that the only signal is "this is what the agent did". Nothing ever says that one option is **better** than another, which is what a verifier or a judge needs.

The recipe we kept, and the one the rest of this post is about, addresses both. It produces a single general checkpoint, trained without any benchmark data, that learns from four sources at every update.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/four_source_batch.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    One update of post-training: four sources, each with its own loss, added with fixed weights.
</div>

The total loss is

$$\mathcal{L} = 0.25\,\mathcal{L}_{\text{terminal}} + 0.25\,\mathcal{L}_{\text{Q\&A}} + 0.20\,\mathcal{L}_{\text{env}} + 0.30\,\mathcal{L}_{\text{pref}}$$

where each term is the loss of one source, described below, and the weights say how much each source counts in an update. Each source comes with its own data, and its own way of preparing it.

## Terminal steps

The first source is coding agents working in a terminal, from the LiteCoder dataset[^litecoder]. Its authors already turned each trajectory into (state, action) steps and ran them through the same Qwen3-8B, so the embeddings come ready to use, and encoding them cost us nothing. There are 278,817 training steps, and 4,096 steps from separate tasks are kept for development. Each update samples 256 training steps and applies the ordinary InfoNCE loss of part 3, with steps of the same task masked from each other, like pairs from the same document.

## Q&A replay

Training only on new data would make the heads forget what they learned about questions and answers, a well-known problem called **catastrophic forgetting**[^mccloskey1989]. The usual remedy is **replay**[^rolnick2019]: keep mixing in some of the old data. Each update draws 128 Q&A pairs, **with their hard negatives**, from 24,576 rows taken from three `shard_00` chunks. Keeping the hard negatives matters: replaying plain pairs would protect the broad geometry of pre-training, but let the sharpness of mid-training fade. A fourth chunk provides a separate development sample, and none of these chunks is a held-out test chunk.

## Choices within an agent environment

The third source fixes the problem of easy wrong answers. It uses four environments from ADP v2[^adp], a collection that converts many public agent datasets into one format: a household simulator, a web shop, a database and a knowledge graph[^zeng2023]. Each one is a set of **trajectories**, the full record of an agent solving a task: the task, then a sequence of agent messages and tool calls, each followed by what the environment returned.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/trajectory_to_pairs.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Every agent message is a decision. The history before it is the state, and the message itself, with its tool call, is the action.
</div>

Each agent message in a trajectory is a decision, so each one gives a pair: the **state** is everything that happened before it, and the **action** is the message itself, with its tool call written as `name(arguments)`. The pairs of one trajectory overlap heavily, since the state of step 6 contains the whole state of step 4, so at most 6 random pairs are kept per trajectory. Histories longer than 2,000 characters keep their first 480, where the task is stated, and their last 1,500, the most recent steps. Pairs whose state was already seen, and empty actions such as "OK.", are dropped.

The trajectories are split by **goal**: a hash of the task's instruction sends one goal in five to development, so the model is always tested on goals it has never trained on, and two runs of the same task never end up on both sides. Each environment contributes up to 640 training pairs and 160 development pairs.

The key step is choosing the wrong answers. For each pair, the model must pick the recorded action among 8 options, where the 7 others are actions recorded for **other goals in the same environment**. A shopping state is now compared with other shopping actions: searches for other products, clicks on other buttons.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/same_env_distractors.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Against actions from other environments, the right one wins without effort. Against seven actions from other shopping goals, the model has something to learn. The scores are illustrative.
</div>

The loss is the cross-entropy of this 8-way choice:

$$\mathcal{L}_{\text{env}} = -\log \frac{e^{s\,\mathbf{u}\cdot\mathbf{v}^{+}}}{e^{s\,\mathbf{u}\cdot\mathbf{v}^{+}} + \sum_{m=1}^{7} e^{s\,\mathbf{u}\cdot\mathbf{d}_m}}$$

where $$\mathbf{u}$$ is the state vector, $$\mathbf{v}^{+}$$ the recorded action, $$\mathbf{d}_1, \dots, \mathbf{d}_7$$ the seven distractors, all three produced by the heads, and $$s$$ the frozen scale. It is the InfoNCE row loss of part 3 with a hand-picked set of wrong answers instead of the batch. Each training pair gets four different draws of distractors, and each update takes 16 examples from each environment, so no environment dominates.

One caveat: another goal's action is not always wrong for this one. Two shopping tasks can both start with a sensible search. This source measures how well the model retrieves the recorded action, which is a useful signal but not a perfect one.

## Answer preferences

The fourth source teaches the heads to judge. It comes from UltraFeedback[^cui2023], a dataset in which several models answered the same instructions, and AI annotators gave each answer a fine-grained score and a written critique. From six of its source files we took 320 training and 64 development instructions each, 1,920 and 384 in all, split by a hash of the instruction's text. For each instruction, we keep the highest- and the lowest-rated answer, provided their scores differ by at least one point, together with their two critiques. How the heads learn from them is the subject of the next section.

## Embedding once

Apart from the terminal steps, all these texts are new: environment states and actions, instructions, answers and critiques. They are embedded once, with the same frozen Qwen3-8B and the same last-token recipe as everything else, and cached as half-precision vectors. Every training run afterwards reads the cache, so it only ever trains the heads, and trying a new recipe costs minutes rather than hours.

# Learning from preferences

The preference loss deserves its own section, because it is the one new idea of this stage. Take the instruction "Describe the capital of Spain in 25 words." The higher-rated answer describes Madrid, though in 17 words instead of 25. The lower-rated one says that Santo Domingo is the capital, of the Dominican Republic. The model should score the first one higher.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/preference_margin.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    The difference between the two answers' scores, the margin, goes through a sigmoid to become a probability. The critiques give a second margin, which also serves as a soft target for the first.
</div>

The score of each answer is the usual one, its cosine with the instruction times the scale, and what matters is the difference:

$$m_a = s\,(\mathbf{u}\cdot\mathbf{v}^{+} - \mathbf{u}\cdot\mathbf{v}^{-}), \qquad P(\text{better} \succ \text{worse}) = \sigma(m_a) = \frac{1}{1 + e^{-m_a}}$$

where $$\mathbf{u}$$ is the instruction's state vector, $$\mathbf{v}^{+}$$ and $$\mathbf{v}^{-}$$ are the action vectors of the higher- and lower-rated answers, $$m_a$$ is the answer margin, and $$\sigma$$ is the sigmoid, which turns any number into a probability between 0 and 1. A margin of 0 means the model cannot tell the two apart, $$\sigma(0) = 0.5$$. A large positive margin means it is confident that the better answer is better. This way of turning a difference of scores into a probability of preference is the Bradley–Terry model[^bradley1952], the same one used to train the reward models of chat assistants[^ouyang2022]. The loss is $$-\log \sigma(m_a)$$, the cross-entropy of a two-way choice, which pushes the margin up until the model is confident.

The critiques add a second view of the same comparison. The critique of the good answer reads like praise with small reservations; the critique of the bad one points out a factual error. We score both critiques against the instruction, exactly like answers, which gives a critique margin $$m_c = s\,(\mathbf{u}\cdot\mathbf{c}^{+} - \mathbf{u}\cdot\mathbf{c}^{-})$$, where $$\mathbf{c}^{+}$$ and $$\mathbf{c}^{-}$$ are the action vectors of the two critiques. The full preference loss is

$$\mathcal{L}_{\text{pref}} = -0.45\log\sigma(m_a) - 0.45\log\sigma(m_c) + 0.10\,\mathrm{BCE}\big(m_a,\ \sigma(m_c)\big)$$

where the first two terms ask both margins to favour the better side, and the third is a binary cross-entropy that asks the answer margin to match the confidence of the critique margin:

$$\mathrm{BCE}\big(m_a,\ t\big) = -\,t \log\sigma(m_a) - (1 - t)\log\big(1 - \sigma(m_a)\big), \qquad t = \sigma(m_c)$$

where $$t$$ is the **soft target**, a probability rather than a hard 0 or 1. The target is treated as a constant: no gradient flows back through it, so this term moves the answer margin toward the critiques and never the reverse. Learning from another model's probabilities rather than from hard labels is the idea behind knowledge distillation[^hinton2015]; here the "teacher" is the critique view of the same heads. When the critiques make a clear case, the answer margin is pushed to be confident too; when the two critiques are close, it is not pushed to an extreme.

The critiques are only used in training. At serving time, the model sees an instruction and candidate answers, exactly like any other decision.

## Intuition

Picture an apprentice judge at a cooking contest. After each round, he is told which of two dishes the panel scored higher, and that alone teaches him something. But he is also given the panel's written notes on each dish. The notes tell him why one dish won, and how clear the verdict was: "a slight lack of salt" and "burnt and inedible" are very different reasons to lose. A judge who reads the notes learns not only to pick the winner, but also how sure to be.

# Training and choosing the checkpoint

Post-training starts from our mid-training checkpoint at step 546. Each update uses AdamW at a learning rate of 0.00008, gradients clipped to 1, and the temperature frozen.

The runs are small enough that we could compare recipes side by side, with identical batches and the same random seed, so that any difference comes from the recipe alone. The first round trained two versions for 320 updates: one with the answer margin only, one with the answers and the critiques. The first round's winner was then continued for 320 more updates at half the learning rate, 0.00004, again in two versions: one with the ordinary same-environment distractors, and one where, on every other update, three of the seven distractors are **mined**, chosen as the actions the current heads find most similar to the state. Each round also tried the average of the late checkpoints of both versions, a cheap trick that often helps[^wortsman2022].

With this many candidates, the choice of the final one matters as much as the training, and it is easy to fool oneself by picking whichever looks best on the benchmarks. So the selection rule was fixed **before** looking at any benchmark, and it only uses development data: held-out terminal steps, a Q&A sample from a chunk not used in training, development goals of the four environments, and development instructions of the preference data.

The rule compares each family to the mid-training checkpoint with the **relative error reduction**:

$$r = \frac{a - a_0}{1 - a_0}$$

where $$a$$ is the candidate's accuracy, $$a_0$$ the mid-training checkpoint's accuracy, and $$1 - a_0$$ the errors that were left to fix. Going from 90% to 95% halves the remaining errors, $$r = 0.5$$, while going from 50% to 55% only fixes a tenth of them, $$r = 0.1$$. This puts families with very different accuracy levels on the same scale. The score is the average of $$r$$ over the four families, with the answer accuracy counting twice as much as the critique accuracy inside the preference family. Two conditions guard against forgetting: a candidate is only eligible if its terminal-step accuracy is at least as high as mid-training's, and its Q&A accuracy at most 0.5 points lower. The mid-training checkpoint itself is eligible, so a recipe that makes things worse is never kept.

Under this rule, the critique version beat the answer-only version in the first round, with a development score of 0.24 against 0.20. In the second round, the ordinary distractors beat the mined ones, 0.30 against 0.28, and in both rounds the weight average scored lower than the best single run. The final checkpoint is therefore the plain one: answers with critiques for 320 updates, then 320 more with ordinary same-environment distractors, 640 updates in all. Mining hard distractors, which sounds like the obvious next step after part 3, did not help here, perhaps because two goals in the same environment already provide confusable actions, and mined ones are more often actions that would be correct too.

Since everything is embedded once, these runs are very cheap. The two training sweeps cost about \\$0.04 of GPU time combined, and the whole general post-training study, including encoding the new texts and running all the evaluations, cost about \\$2.85.

# Implementation

All the stages share one training loop in the [repository](https://github.com/MKDYasserH/system-one-model), and each stage only changes its data and its validation score. Here is the dealer of the data pipeline: for each pool, it shuffles, cuts the shuffled order into batches and sends them one by one, and repeats this as many times as configured:

```python
def _deal_batches(self) -> None:
    rng = np.random.default_rng(self.settings.seed)
    batch_size = self.settings.batch_size
    while (pool := self._get(self.pools)) is not END:
        for _ in range(self.settings.pool_repeats):
            shuffled = rng.permutation(pool.num_rows)
            # Rows left over after the last full batch are skipped; each shuffle skips different ones.
            for start in range(0, pool.num_rows - batch_size + 1, batch_size):
                # Sorted indices read memory in order, which makes the copy faster.
                rows = np.sort(shuffled[start : start + batch_size])
                if not self._put(self.batches, pool.take(rows)):
                    return
    self._put(self.batches, END)
```

One post-training update draws from the four sources and adds their losses with the weights above. The environment choices are rows of 8 action indices, the recorded one first, scored against their state and trained with an ordinary cross-entropy whose target is column 0:

```python
agent_loss = sample_loss(train, 256, rng)
qa_loss = sample_loss(replay, 128, rng)
pref_rows = train_rows[rng.choice(len(train_rows), 64, replace=False)]
answer, review = margins(model, vectors, pref_rows)
distractors = mined if recipe == "continued_mined" and step % 2 else train_choices
sampled = np.concatenate([choices[rng.integers(len(choices), size=16)]
                          for choices in distractors.values()])
logits = choice_logits(model, actions, torch.as_tensor(sampled, device=device))
domain_loss = F.cross_entropy(logits, torch.zeros(len(logits), device=device, dtype=torch.long))
loss = (0.25 * agent_loss + 0.25 * qa_loss + 0.20 * domain_loss +
        0.30 * preference_loss(answer, review, recipe != "answer_only"))
```

The preference loss is three lines, with `softplus(-m)` standing for $$-\log\sigma(m)$$ and `detach()` making the critique margin a constant target:

```python
def preference_loss(answer, review, use_reviews):
    if not use_reviews:
        return F.softplus(-answer).mean()
    # The critique supplies a second supervised view and a soft target for the answer-only view.
    teacher = review.detach().sigmoid()
    return (0.45 * F.softplus(-answer).mean() + 0.45 * F.softplus(-review).mean()
            + 0.10 * F.binary_cross_entropy_with_logits(answer, teacher))
```

And the selection score is the relative error reduction, averaged over the four families:

```python
def selection_score(metrics, baseline):
    # Equal weight for the four data families. Normalize each gain by remaining errors.
    def error_reduction(metric):
        return (metrics[metric] - baseline[metric]) / max(1e-6, 1 - baseline[metric])
    preference = (2 * error_reduction("answer_preference") + error_reduction("review_preference")) / 3
    return (preference + error_reduction("agent_top1") + error_reduction("qa_top1") + error_reduction("action_macro")) / 4
```

# What's next

We now have a trained decision model: heads pre-trained on 62 million pairs, sharpened on hard negatives, and post-trained with a general recipe on terminal steps, agent environments, Q&A replay and answer preferences with critiques. What remains is the question that matters: how good is it? Part 5 puts it to the test on the tricky Q&A set, on choosing between coding agents' attempts, and in a game that needs decisions in milliseconds, compares it with the released CLM-8B, and looks back at what worked, what did not, and what we would do differently.

# References

[^nemotron]: NVIDIA. *Nemotron-CC v2.1*, part of the Nemotron pretraining datasets. [Dataset on Hugging Face](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1).
[^su2024]: D. Su et al. *Nemotron-CC: transforming Common Crawl into a refined long-horizon pretraining dataset.* [arXiv:2412.02595](https://arxiv.org/abs/2412.02595), 2024.
[^clm]: J. Kwok, H. Kang, T. Suresh, J. Saad-Falcon, M. Pavone, C. Ré and A. Mirhoseini. *CLM-8B* ([project page](https://contrastive-lm.notion.site)) and its [pre-training embeddings](https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron).
[^loshchilov2019]: I. Loshchilov and F. Hutter. *Decoupled weight decay regularization.* International Conference on Learning Representations, 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
[^malladi2022]: S. Malladi, K. Lyu, A. Panigrahi and S. Arora. *On the SDEs and scaling rules for adaptive gradient algorithms.* Advances in Neural Information Processing Systems, 2022. [arXiv:2205.10287](https://arxiv.org/abs/2205.10287)
[^bengio2009]: Y. Bengio, J. Louradour, R. Collobert and J. Weston. *Curriculum learning.* International Conference on Machine Learning, 2009.
[^adp]: *Agent Data Protocol (ADP) v2*, NeuLab. [Dataset on Hugging Face](https://huggingface.co/datasets/neulab/adp-v2).
[^mccloskey1989]: M. McCloskey and N. J. Cohen. *Catastrophic interference in connectionist networks: the sequential learning problem.* Psychology of Learning and Motivation, vol. 24, 1989.
[^rolnick2019]: D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap and G. Wayne. *Experience replay for continual learning.* Advances in Neural Information Processing Systems, 2019. [arXiv:1811.11682](https://arxiv.org/abs/1811.11682)
[^litecoder]: *LiteCoder terminal SFT, Qwen3-8B embeddings.* [Dataset on Hugging Face](https://huggingface.co/datasets/Hkang/litecoder-terminal-sft-qwen3-8b-embeddings).
[^zeng2023]: A. Zeng et al. *AgentTuning: enabling generalized agent abilities for LLMs.* [arXiv:2310.12823](https://arxiv.org/abs/2310.12823), 2023.
[^cui2023]: G. Cui et al. *UltraFeedback: boosting language models with high-quality feedback.* [arXiv:2310.01377](https://arxiv.org/abs/2310.01377), 2023.
[^bradley1952]: R. A. Bradley and M. E. Terry. *Rank analysis of incomplete block designs: I. The method of paired comparisons.* Biometrika, 39(3/4), 1952.
[^ouyang2022]: L. Ouyang et al. *Training language models to follow instructions with human feedback.* Advances in Neural Information Processing Systems, 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
[^hinton2015]: G. Hinton, O. Vinyals and J. Dean. *Distilling the knowledge in a neural network.* [arXiv:1503.02531](https://arxiv.org/abs/1503.02531), 2015.
[^wortsman2022]: M. Wortsman et al. *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.* International Conference on Machine Learning, 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)
