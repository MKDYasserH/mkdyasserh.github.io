---
title: "Contrastive Learning: InfoNCE, Temperature and Hard Negatives"
description: "How two small heads learn to put a question next to its answer, using nothing but matching pairs: the in-batch similarity matrix, the InfoNCE loss, the temperature, and the hard negatives that make the model sharp."
date: 2026-11-21 12:00:00 +0100
series: decision-models
part: 3
tags: [decision models, contrastive learning, infonce, hard negatives]
cover: assets/img/system_one/similarity_matrix.png
giscus_comments: true
---
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/similarity_matrix.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    A batch of five (state, action) pairs. Every state is compared with every action, the right answers sit on the diagonal, and each row and each column becomes a small classification problem.
</div>

# Introduction

In the [first post](/blog/2026/what-is-a-decision-model/) we built the decision model: a frozen Qwen3-8B that turns every text into an embedding, and two small heads that map states and actions to vectors of dimension 512, whose cosine is the score. In the [second post](/blog/2026/fine-tuning-a-frozen-llm/) we saw why the language model stays frozen, and how the embeddings are computed once and stored.

What we have not seen is how the heads learn. When training starts, they are random networks: the vector of "Which planet is closest to the Sun?" lands somewhere on the sphere, and the vector of "Mercury" lands somewhere else, with no reason to be close. Training has to move them together, and move "Mars" away.

The surprising part is what the training data looks like. It contains only **matching pairs**: a question and its right answer, an agent's state and the action it took. Nobody labeled the wrong answers. This post is about how a model can learn to reject wrong answers it was never shown, a family of methods called **contrastive learning**, and about the three ingredients that make it work for us: the InfoNCE loss, the temperature, and hard negatives.

# Wrong answers for free

The idea is simple. Training goes through the data in batches, and our batches are large: 16,384 pairs at a time. Inside a batch, every pair has its own right action, and the actions of the 16,383 other pairs are, almost always, wrong answers for it. The answer to a question about planets is a wrong answer to a question about cooking. So every batch provides, for free, 16,383 wrong answers for each state.

Contrastive learning turns this into a training signal: for each state, push its vector toward its own action and away from all the other actions in the batch. The model never needs to be told what is wrong; it learns by **contrast** with what is right.

# The similarity matrix

Let's write it down. Take a batch of $$B$$ pairs. The state head turns the $$i$$-th state into a normalized vector $$\mathbf{u}_i$$, and the action head turns the $$j$$-th action into a normalized vector $$\mathbf{v}_j$$, both of dimension 512. Stacking them gives two matrices, $$U$$ and $$V$$, each of shape $$(B, 512)$$.

Comparing every state with every action gives a matrix of scores:

$$Z = s \, U V^\top, \qquad z_{ij} = s \, \mathbf{u}_i \cdot \mathbf{v}_j = s \cos\theta_{ij}$$

where $$Z$$ has shape $$(B, B)$$, $$z_{ij}$$ is the score of state $$i$$ against action $$j$$, $$s$$ is the learned scale from part 1, and $$\theta_{ij}$$ is the angle between the two vectors. Because both vectors have length 1, their dot product is the cosine of that angle.

This one matrix product does all the work. With $$B = 16{,}384$$, it computes about 268 million scores at once, which a GPU does in a fraction of a second. The right answers are on the diagonal: entry $$z_{ii}$$ compares state $$i$$ with its own action. Everything off the diagonal compares a state with someone else's action.

# InfoNCE: a classification over the batch

Now read the matrix one row at a time. Row $$i$$ holds the scores of state $$i$$ against all $$B$$ actions, exactly the situation of part 1: one state, a list of options, and a softmax that turns scores into probabilities. The only difference is that the options are the actions of the batch, and we know which one is right, the $$i$$-th. So each row is a classification problem with $$B$$ classes, and we can train it with the usual classification loss, the cross-entropy:

$$\mathcal{L}_i = -\log p_i^{+}, \qquad p_i^{+} = \frac{e^{z_{ii}}}{\sum_{k=1}^{B} e^{z_{ik}}}$$

where $$p_i^{+}$$ is the probability the model gives to the right action of state $$i$$ (the $$+$$ marks the right answer), $$z_{ii}$$ is the score of that right action, the denominator adds up the exponentials of all $$B$$ scores in row $$i$$, and $$\mathcal{L}_i$$ is the loss of row $$i$$. The loss is small when $$p_i^{+}$$ is close to 1, that is, when the state's own action stands out from all the others, and it grows quickly as $$p_i^{+}$$ drops: $$-\log 0.9 \approx 0.1$$, but $$-\log 0.01 \approx 4.6$$. In the animation, state 2 gives its own action a probability of 0.90, so its loss is $$-\log 0.90 \approx 0.11$$.

This loss is called **InfoNCE**[^oord2018], from "noise-contrastive estimation": the other actions in the batch play the role of noise that the right answer has to be told apart from. Averaging over the rows gives the loss of the batch, read from the states' side:

$$\mathcal{L}_{\text{rows}} = \frac{1}{B} \sum_{i=1}^{B} -\log \frac{e^{z_{ii}}}{\sum_{k=1}^{B} e^{z_{ik}}}$$

which is the row loss $$\mathcal{L}_i$$ averaged over the $$B$$ states of the batch.

## Both directions

The matrix can also be read by columns. Column $$i$$ holds the scores of action $$i$$ against every state, and asks the mirror question: which state does this action belong to? The same loss, written for columns, is

$$\mathcal{L}_{\text{columns}} = \frac{1}{B} \sum_{i=1}^{B} -\log \frac{e^{z_{ii}}}{\sum_{k=1}^{B} e^{z_{ki}}}$$

where the denominator now runs down column $$i$$: $$z_{ki}$$ is the score of state $$k$$ against action $$i$$. Our loss is the average of the two:

$$\mathcal{L} = \tfrac{1}{2}\left(\mathcal{L}_{\text{rows}} + \mathcal{L}_{\text{columns}}\right)$$

where $$\mathcal{L}_{\text{rows}}$$ reads the matrix from the states' side and $$\mathcal{L}_{\text{columns}}$$ from the actions' side.

This symmetric version is the one CLIP[^radford2021] uses to match images with captions. Reading both directions gives each head two kinds of signal: a state must beat the other actions, and an action must beat the other states, so neither head can cheat by placing all its vectors in one corner of the sphere.

## Intuition

Picture a hall with 16,384 people, each holding a question, and a table with 16,384 answer cards. Each person has to find the card that answers his question, and each card has to be claimed by the right person. Nobody ever tells a person which cards are wrong; he only learns, after each round, which one was his. Over many rounds, people get better at telling their card apart from the others, and the cards that look alike are the ones that teach the most.

# What the gradient does

To see how the heads actually learn, we need the derivative of the loss with respect to each score: it says how much the loss changes when that score goes up a little. Training uses gradient descent, which moves every value by a small step in the direction that lowers the loss, that is, against the sign of its derivative:

$$z_{ij} \leftarrow z_{ij} - \eta \, \frac{\partial \mathcal{L}_i}{\partial z_{ij}}$$

where $$\eta$$ is the learning rate, a small step size, and $$\leftarrow$$ means "is replaced by". A positive derivative means that raising the score would raise the loss, so the update lowers the score; a negative derivative means the opposite. (In practice the update is applied to the weights of the heads, which then change the scores, but the direction is the same.)

For the row loss, the derivative has a remarkably clean form:

$$\frac{\partial \mathcal{L}_i}{\partial z_{ij}} = p_{ij} - \mathbf{1}[j = i], \qquad p_{ij} = \frac{e^{z_{ij}}}{\sum_{k=1}^{B} e^{z_{ik}}}$$

where $$p_{ij}$$ is the probability the model gives to action $$j$$ for state $$i$$ (the right action's $$p_i^{+}$$ is the case $$j = i$$), and $$\mathbf{1}[j = i]$$ is 1 for the right action and 0 for the others.

Now plug the two cases into the update. For the right action, the derivative is $$p_i^{+} - 1$$, which is negative, so the update raises its score by $$\eta \, (1 - p_i^{+})$$: the less sure the model is, the stronger the pull. For a wrong action, the derivative is $$p_{ij}$$, which is positive, so the update lowers its score by $$\eta \, p_{ij}$$: the more probability a wrong action steals, the harder it is pushed down. Take row 2 of the opening animation. The right action has a probability of 0.90, action 3 has 0.10 and action 1 only 0.001 (shown as 0.00). One step raises the right score by $$0.10\,\eta$$, lowers action 3 by the same amount, and leaves action 1 almost where it is: the model already knows it is wrong.

The same reasoning applies to the state vector itself:

$$\frac{\partial \mathcal{L}_i}{\partial \mathbf{u}_i} = s \left( \bar{\mathbf{v}}_i - \mathbf{v}_i \right), \qquad \bar{\mathbf{v}}_i = \sum_{j=1}^{B} p_{ij} \mathbf{v}_j$$

where $$\bar{\mathbf{v}}_i$$ is the average of all the actions weighted by their probability, the action the model currently expects for state $$i$$, $$\mathbf{v}_i$$ is the right action and $$s$$ is the scale. The update $$\mathbf{u}_i \leftarrow \mathbf{u}_i - \eta \, \partial \mathcal{L}_i / \partial \mathbf{u}_i$$ therefore adds $$\eta \, s \, (\mathbf{v}_i - \bar{\mathbf{v}}_i)$$ to the state: a small step along the arrow that goes from the expected action to the right one. The state moves toward its own action and away from the actions it currently confuses with it, and when the model is already sure, $$\bar{\mathbf{v}}_i$$ is almost $$\mathbf{v}_i$$ and the step is almost zero.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/pull_push_sphere.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Gradient steps on the circle. The right action is pulled toward the state, and each wrong action is pushed away in proportion to its probability, so the closest one moves the most.
</div>

The animation shows these forces on a circle: the wrong action closest to the state has the highest probability and is pushed the hardest, while the far ones barely move. In real training, the heads change rather than the points themselves, and states move as well as actions, but the forces are the same.

This also explains a weakness. Once most wrong answers in a batch are far away, their probabilities are all close to 0, and the batch stops teaching. We will come back to this with hard negatives.

# Why bigger batches help

With $$B$$ actions in a row, a model that knows nothing gives each the same probability, $$1/B$$, and its loss is $$\log B$$. For our batches, that is $$\log 16{,}384 \approx 9.7$$: the loss at chance level. In our pre-training run, the loss was down to about 1.9 after 8,000 steps.

Larger batches help for two reasons. The first is that a harder test is a more informative one: picking the right answer among 16,384 is a much stronger statement than picking it among 16. Van den Oord and colleagues made this precise[^oord2018]: the InfoNCE loss gives a lower bound on the mutual information between states and actions,

$$I(\text{state}; \text{action}) \geq \log B - \mathcal{L}_{\text{rows}}$$

where $$I(\text{state}; \text{action})$$ is the **mutual information**: how much knowing the state reduces our uncertainty about which action goes with it. It is 0 when the state says nothing about the action, and it grows as the state pins the action down. It is measured in nats, the unit of the natural logarithm, which is also the unit of the loss. $$\log B$$ is the chance-level loss from above and $$\mathcal{L}_{\text{rows}}$$ is the row loss. So the further training pushes the loss below chance level, the more shared information the heads are guaranteed to have captured. Since the loss can never go below 0, the bound can never exceed $$\log B$$, so a larger batch lets the loss measure, and reward, more of what the two share. The second reason is the one from the gradient: the more wrong answers a batch contains, the more likely it is to contain a few close ones, which are the ones that teach. Contrastive methods like SimCLR[^chen2020] and CLIP rely on very large batches for exactly this reason, and it is why being able to afford batches of 16,384, as we saw in part 2, matters so much.

# Same-document masking

There is one catch in the "wrong answers for free" idea. Our question-and-answer data was built from documents, and a single document usually produces several questions. Two questions about the same article can easily have answers that fit both. Counting one as a wrong answer for the other would punish the model for being right.

So, before the softmax, we mask them: if pairs $$i$$ and $$j$$ come from the same document, with $$i \neq j$$, we set $$z_{ij} = -\infty$$. Since $$e^{-\infty} = 0$$, that action simply disappears from the denominator of row $$i$$, and it is neither rewarded nor punished. The diagonal is never masked: a pair's own action always stays the right answer. The last part of the opening animation shows this for two pairs from the same document.

# Temperature

The scale $$s$$ deserves a closer look, because it shapes training as much as it shapes the final probabilities. It is often written as an inverse temperature, $$s = 1/\tau$$, a name borrowed from physics: a high temperature spreads probability evenly, a low temperature concentrates it on the best option.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/temperature_softmax.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    The same five cosines turned into probabilities with three different scales. With s = 1 the model can never be confident; at the cap, s = 100, a small difference in cosine decides everything.
</div>

Cosines only range from -1 to 1, so without a scale the scores are too close together for the softmax to ever be confident, as the animation shows at $$s = 1$$. Multiplying by $$s$$ stretches the differences. But too large a scale too early would make the model confident before it knows anything, with very sharp, unstable gradients. The answer, borrowed from CLIP, is to **learn** $$s$$.

We store it as a logarithm, $$s = e^{t}$$ with $$t$$ the trained parameter, so that it can never become negative, and start it at $$\tau = 0.07$$, that is $$s \approx 14.3$$. As the heads improve and the right answers stand out, the loss can be lowered further by raising $$s$$, so training raises it. To stop it from growing without limit, it is capped at 100. In our pre-training run, the scale had climbed to between 85 and 89 by step 8,000. In post-training we freeze it, because the CLM-8B checkpoints after mid-training and after post-training have the same scale.

## Intuition

Think of $$s$$ as the contrast setting of a photograph. With the contrast too low, everything is the same shade of grey and you cannot tell the objects apart. Turned up, the differences pop out. Turned up too far, every small difference becomes black or white, and you lose detail. Learning $$s$$ lets the model turn the contrast up at the pace its knowledge allows.

# Hard negatives

Back to the weakness we saw in the gradient. After a while, the in-batch wrong answers are mostly easy: a question about planets against answers about cooking, football and tax law. Their probabilities are close to 0, they stop teaching, and the model can reach a low loss while still confusing answers that are only *slightly* wrong. A model that knows "Mercury" fits a question about planets better than "lasagna" has not yet learned that it fits better than "Venus".

**Hard negatives** fix this. For each question, the dataset provides a few wrong answers chosen to be close to the right one: plausible, on topic, but wrong. These were generated with Gemini and embedded by the people who released the data, up to $$k = 10$$ per pair, for a subset of about 2.8 million pairs. Mined hard negatives are a standard tool in retrieval[^karpukhin2020] and contrastive learning[^robinson2021].

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/hard_negative_columns.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Each state gets up to k extra wrong answers of its own, appended to its row. States with fewer than k have their empty slots masked, and the extra answers only extend the rows.
</div>

In the loss, each state's hard negatives become extra columns in its own row. Write $$\mathbf{n}_{i,1}, \dots, \mathbf{n}_{i,c_i}$$ for the hard negatives of state $$i$$, passed through the action head like any other action, where $$c_i \leq k$$ is how many state $$i$$ actually has. The probability of the right action becomes

$$p_i^{+} = \frac{e^{z_{ii}}}{\sum_{j=1}^{B} e^{z_{ij}} + \sum_{m=1}^{c_i} e^{s \, \mathbf{u}_i \cdot \mathbf{n}_{i,m}}}$$

where the first sum runs over the in-batch actions, as before, and the second over the state's own hard negatives. Since some states have fewer than $$k$$ hard negatives, the empty slots are filled with $$-\infty$$, which removes them from the sum exactly like the same-document mask.

Hard negatives only extend the **rows**. They belong to one state each, so there is no sensible column for them: "which state does this wrong answer belong to?" has no answer. The column loss stays as it was.

The effect is large. The test that measures it, which we will use throughout the series, asks the model to pick the right answer among its own hard negatives on held-out questions. In the published CLM numbers, accuracy on that test goes from about 52% after the first training stage, without hard negatives, to about 69% after the second, with them.

## Intuition

Hard negatives are the difference between two multiple-choice exams. In the first, the wrong options are absurd: "Which planet is closest to the Sun? Mercury, lasagna, the Eiffel Tower." You can pass it without knowing any astronomy. In the second, the options are Mercury, Venus and Mars, and you actually have to know. A student who only ever takes the first kind of exam will feel confident and learn very little. The in-batch negatives are the first exam; hard negatives are the second.

# Implementation

Here is the loss as it is written in the [repository](https://github.com/MKDYasserH/system-one-model). The in-batch logits are one matrix product, followed by the same-document mask:

```python
def in_batch_logits(embedded: EmbeddedBatch) -> torch.Tensor:
    """[rows, rows]: row i scores state i against every action. Pair i's own action is column i."""
    logits = embedded.scale * embedded.states @ embedded.actions.t()
    same_doc = embedded.doc_ids[None] == embedded.doc_ids[:, None]
    same_doc.fill_diagonal_(False)  # keep each pair's own action
    # Another pair from the same document is probably a fine answer too, so don't count it as wrong.
    return logits.masked_fill(same_doc, float("-inf"))
```

The hard negatives arrive as a tensor of shape $$(B, k, 512)$$, one row of $$k$$ wrong answers per state. Each state is compared only with its own, and the empty slots are masked using the count of each row:

```python
def hard_negative_logits(embedded: EmbeddedBatch) -> torch.Tensor:
    """[rows, k]: each state against its own hard negatives. Empty slots are -inf."""
    logits = embedded.scale * torch.einsum("rd,rkd->rk", embedded.states, embedded.hard_negatives)
    slot = torch.arange(logits.shape[1], device=logits.device)
    return logits.masked_fill(slot[None] >= embedded.negative_counts[:, None], float("-inf"))
```

The loss reads the matrix both ways. The columns use the in-batch matrix as is; the rows get the hard negatives appended, when the batch has them. In both cases, the right answer for row (or column) $$i$$ is index $$i$$, which is what `targets` says:

```python
def contrastive_loss(embedded: EmbeddedBatch) -> torch.Tensor:
    logits = in_batch_logits(embedded)
    targets = torch.arange(len(logits), device=logits.device)
    # Read by column: for each action, which state is its own?
    action_to_state = F.cross_entropy(logits.t(), targets)
    # Read by row: for each state, which action is its own? Hard negatives are extra wrong actions for that row.
    if embedded.hard_negatives is not None:
        logits = torch.cat([logits, hard_negative_logits(embedded)], dim=1)
    state_to_action = F.cross_entropy(logits, targets)
    return (state_to_action + action_to_state) / 2
```

Finally, the scale lives in the model, stored as a logarithm and capped:

```python
# Stored as a log so exp() keeps the scale positive. scale = 1 / temperature.
self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1 / loss.initial_temperature)))
self.max_logit_scale = loss.max_logit_scale

def logit_scale(self) -> torch.Tensor:
    return self.log_logit_scale.exp().clamp(max=self.max_logit_scale)
```

# What's next

We now have the whole learning machinery: pairs, a batch, a similarity matrix, a symmetric InfoNCE loss with a learned temperature, a mask for pairs that share a document, and hard negatives for the answers that are almost right. What remains is the data itself and the order in which the model sees it. Part 4 follows the training from start to finish: the 62 million question-and-answer pairs and how they are fed to the GPU, the three training stages, pre-, mid- and post-training, and the post-training recipe that worked best.

# References

[^oord2018]: A. van den Oord, Y. Li and O. Vinyals. *Representation learning with contrastive predictive coding.* [arXiv:1807.03748](https://arxiv.org/abs/1807.03748), 2018.
[^radford2021]: A. Radford et al. *Learning transferable visual models from natural language supervision.* International Conference on Machine Learning, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
[^chen2020]: T. Chen, S. Kornblith, M. Norouzi and G. Hinton. *A simple framework for contrastive learning of visual representations.* International Conference on Machine Learning, 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
[^karpukhin2020]: V. Karpukhin et al. *Dense passage retrieval for open-domain question answering.* Empirical Methods in Natural Language Processing, 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
[^robinson2021]: J. Robinson, C.-Y. Chuang, S. Sra and S. Jegelka. *Contrastive learning with hard negative samples.* International Conference on Learning Representations, 2021. [arXiv:2010.04592](https://arxiv.org/abs/2010.04592)
