---
title: "Does It Work? A Verifier for Coding Agents, the Results, and What We Learned"
description: "Our decision model against the released CLM-8B: tricky questions, choosing between a coding agent's attempts, and a game played in real time. And the lessons, including the ones the benchmarks taught us about themselves."
date: 2026-11-21 14:00:00 +0100
series: decision-models
part: 5
tags: [decision models, evaluation, verifiers, best-of-n, benchmarks]
cover: assets/img/system_one/tricky_qa_progress.png
giscus_comments: true
---
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/tricky_qa_progress.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Accuracy on the tricky Q&A test after each stage, from our training logs. Pre-training creeps up to about 52%, mid-training jumps past 65% in its first 42 steps, and post-training ends at 70.8%. The dashed lines are the CLM authors' published numbers.
</div>

# Introduction

Over the last four posts we built a decision model from scratch: a frozen Qwen3-8B that turns any text into an embedding, two small heads that map states and options to vectors of dimension 512, a contrastive loss that teaches them to put each state next to its right option, and three stages of training. The result is a single checkpoint, our general post-training checkpoint. This post asks the only question that matters: does it work?

We test it in three ways, each closer to real use than the last. The first is a held-out set of tricky questions, where every right answer comes with up to ten wrong answers designed to look right. The second is the job decision models were made for in coding: a coding agent attempts the same task several times, and the model must pick the attempt to keep. The third is a game played in real time, where the model has about 17 milliseconds per decision.

In every test, we compare our checkpoint with the released CLM-8B[^clm], the model we set out to replicate, and whenever we could, we ran both ourselves on exactly the same data. Remember from part 4 that our checkpoint was chosen with a rule fixed in advance, on development data only. None of the numbers below played any part in choosing it.

The last part of the post is about what we learned. Some lessons are about decision models, but the most useful ones are about the tests themselves.

# Test 1: tricky questions

The first test reuses the two chunks of question-and-answer pairs that were held out from training, the ones with hard negatives. For each of their 191,161 questions, the model sees the right answer and that question's hard negatives, up to ten wrong answers generated to be close to it, and must rank the right one first. The score is the share of questions where it does, the **top-1 accuracy**. A model that guesses at random gets about 10%, since a question has about nine hard negatives on average.

| Checkpoint | Ours | CLM-8B (published) |
|---|---:|---:|
| After pre-training | 51.6% | 52.1% |
| After mid-training | 68.1% | 69.2% |
| After post-training | **70.8%** | 68.5% |

Our run follows the published one closely through the first two stages, about half a point and one point below. Post-training is where the two diverge. CLM-8B's post-training costs it a little accuracy on these questions, while ours gains almost three points. We also measured the released CLM-8B checkpoint ourselves on the same 191,161 rows and got 68.6%, in line with the published number, so the comparison is fair.

The interesting part is the shape of the curve in the opening animation, which comes straight from our training logs. During pre-training, the heads become very good at the in-batch task they are trained on: by the end, they pick the right answer among 1,024 candidates from other questions about 83% of the time. Yet on the tricky test, accuracy creeps from 44% at step 4,000 to 52% at step 29,000. The heads learn to tell a question about planets from a question about cooking, but not Mercury from Venus.

Mid-training changes this almost immediately. After only 42 steps, about four minutes on our GPU, tricky accuracy jumps from 51.6% to 65.2%, a gain that 25,000 steps of pre-training could not produce. The remaining 500 steps add three more points. This is the curriculum of part 4 at work: the broad geometry learned in pre-training is what lets the hard negatives teach so fast, and the hard negatives are where the precision comes from.

Post-training then adds 2.7 points, even though none of its four sources is new Q&A data. Part of it comes from the Q&A replay, which keeps the hard negatives in the loss, and part from the other sources: choosing between close actions in one environment and ranking two answers to the same instruction are, after all, the same skill.

# A verifier for coding agents

The second test is the one decision models were built for. A coding agent, given a task, writes code, runs commands and reads their output, step after step, until it believes it is done. Its success varies from one attempt to the next, so a common strategy is to let it make several attempts and keep the best one. This is called **best-of-N**. At deployment there are no hidden tests to tell which attempt is right, so the choice needs a **verifier**: a model that looks at each attempt and says how promising it is[^cobbe2021].

A decision model is a natural verifier, because an agent's attempt is exactly the kind of data it was trained on. Every step of the attempt is a decision: the state is everything that happened before the step, and the action is what the agent did. The model scores each step, and the score of the attempt is the average over its last 12 steps:

$$\text{score}(\text{attempt}) = \frac{1}{12}\sum_{t=T-11}^{T} s\,\mathbf{u}_t\cdot\mathbf{v}_t$$

where $$T$$ is the number of steps in the attempt, $$\mathbf{u}_t$$ and $$\mathbf{v}_t$$ are the state and action vectors of step $$t$$ produced by the two heads, and $$s$$ is the scale. Each term is the familiar score of part 1, how well the action fits its state. The attempt with the highest score is kept. Scoring every step rather than only the final answer is the idea behind **process reward models**[^lightman2023], and the last 12 steps are where an attempt either converges on a fix or goes in circles.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/best_of_four.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    Best-of-four verification. Every step of every attempt gets a score, the last 12 are averaged, and the attempt with the highest average is kept. Only then do the task's tests say whether it passed. The scores are illustrative.
</div>

## The DeepSWE benchmark

The CLM authors released a benchmark for this, which they call DeepSWE: 113 coding tasks, each attempted several times by a coding agent, with every attempt marked as passing or failing by the task's tests, and all the steps already embedded with Qwen3-8B. The tasks are split into 75 for training and 38 held out, and on each held-out task the verifier picks one of up to four attempts. A task counts as **resolved** if the picked attempt passed.

The protocol also adapts the model to this kind of data before testing. Starting from the checkpoint being tested, a short fine-tuning run trains the heads contrastively on the steps of the **passing** attempts of the training tasks, 22,576 steps from 66 tasks, with the authors' own scripts and settings, which we ran unchanged. A model trained on what successful attempts look like tends to give lower scores to the steps of failing ones, which repeat themselves, revert changes or drift away from the task. The fine-tuning run depends on a random seed, so the whole procedure is repeated with five seeds and the results are averaged.

## Intuition

Think of a coach who has to bet on one of four runners after watching only the last lap of each. He has never seen this race before, but he has watched thousands of good runners finish. A runner who finishes like those, steady and in control, is a better bet than one who is weaving and slowing down. The coach does not know who will win, and sometimes the steady runner falls on the last step. But averaged over many races, betting on the finish that looks most like a winner's beats betting at random.

## Results: read the ceiling first

Before looking at any score, look at what the test allows. Of the 38 held-out tasks, 21 are resolved by **every** attempt, and 4 by **none**. On those 25 tasks, the verifier's choice cannot change anything. Only the 13 remaining tasks have both passing and failing attempts, so every verifier, good or bad, lands between 21 and 34 resolved tasks. Picking an attempt at random resolves 28 on average, and a perfect verifier would resolve 34.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/deep_swe_ceiling.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    The 38 held-out DeepSWE tasks. Only the 13 grey ones can tell verifiers apart, so every result lies between 21 and 34, with a random pick at 28.
</div>

| Verifier | Seeds 1234, 1, 2, 3, 4 | Mean resolved, out of 38 |
|---|---|---:|
| Random pick | | 28 |
| Ours, after the verifier fine-tuning | 29, 30, 32, 30, 31 | 30.4 |
| CLM-8B, after the verifier fine-tuning | 31, 32, 30, 31, 31 | **31.0** |
| Perfect pick | | 34 |

CLM-8B comes out ahead by 0.6 tasks on average. That is less than the spread between seeds of the same model, which goes from 29 to 32 for ours, and it is played out on 13 tasks, where one task is worth more than two and a half percentage points. The honest summary is a near tie, with a slight edge to CLM-8B. Measured from a random pick, ours closes 40% of the gap to a perfect verifier, and CLM-8B half of it.

We also tried both checkpoints **directly**, without the verifier fine-tuning: both resolve 27 tasks, one fewer than a random pick. On this benchmark, a general decision model is not a useful verifier out of the box. It needs the short adaptation to what passing attempts look like, which is cheap, since everything is already embedded.

## A second coding test

DeepSWE is small, so we added a test from outside the CLM release: 95 held-out issues from SWE-bench Verified[^jimenez2024], a standard benchmark of real GitHub issues, each with up to 15 attempts by coding agents, taken from a public dataset of embedded attempts. The metric is the **expected best-of-four success**: for every possible group of four attempts on an issue, the verifier picks one, and we average whether it passed. Here both checkpoints are used directly, without any fine-tuning.

| Verifier | Expected best-of-four success |
|---|---:|
| Random pick | 70.7% |
| CLM-8B | 69.4% |
| Ours | **71.3%** |
| Perfect pick | 83.3% |

Ours is ahead of CLM-8B by about two points, but look at the random pick again: ours is only 0.6 points above it, and CLM-8B is below it. Without adaptation, neither model is a strong verifier for these attempts. As we will see in the lessons, a head trained on this kind of data does much better.

# Test 3: a game in real time

The CLM release comes with a demonstration that puts decision speed to the test: a side-scrolling runner game, where a dinosaur has to jump over or duck under obstacles. Each course lasts 60 seconds, and the model makes about 3,300 decisions per course, one every 18 milliseconds or so. Our checkpoint answered in about 17 ms per decision (median), on par with CLM-8B, which is the speed part 1 promised.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid cache_bust=true path="assets/video/system_one/game_decision.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true %}
    </div>
</div>
<div class="caption">
    One decision in the game. The state describes what is ahead, and each option comes with the planner's note about it. If the model proposes an unsafe action, the shield replaces it with the most probable safe one. The descriptions and probabilities are illustrative.
</div>

At each decision, the game builds a state, such as "Dino runner game. Cactus ahead, 120 px away.", and three options, jump, duck and run. A built-in **planner** works out which actions are safe and which is best, and its notes are written into each option's text: "Safe. … Best." for its preferred action, "Unsafe. … Collision." for the dangerous ones. The model picks the option with the highest probability. A **shield** then checks the choice: if the model proposed an unsafe action, it executes the most probable safe action instead.

Two numbers describe a run. **Planner agreement** is the share of decisions where the model's proposal has the same effect as the planner's choice, measured before the shield acts. **Survival** counts the courses finished without a single crash. We ran both checkpoints on five fresh courses, with the shield on:

| Checkpoint | Planner agreement | Courses survived | Shield interventions |
|---|---:|---:|---:|
| CLM-8B | **66.9%** | 5 / 5 | 4,956 |
| Ours | 36.1% | 5 / 5 | **3,083** |

CLM-8B follows the planner far more often. Ours agrees with it only about a third of the time, yet proposed an unsafe action less often: the shield had to step in 3,083 times for ours, against 4,956 for CLM-8B. In other words, our model often picks a **safe** action that is not the planner's favourite, like running instead of jumping when both are fine. Both survive every course with the shield. Without it, our checkpoint is not a good player: on the five original courses of the demonstration, it crashed 45 times and finished none without a crash.

This game is the test where CLM-8B is clearly better, and also the one where the numbers need the most care, as the lessons will show.

# The full picture

| Test | Ours | CLM-8B | Random | Perfect |
|---|---:|---:|---:|---:|
| Tricky Q&A, top-1 accuracy | **70.8%** | 68.6% | ≈ 10% | 100% |
| DeepSWE, resolved out of 38, with verifier fine-tuning | 30.4 | **31.0** | 28 | 34 |
| DeepSWE, resolved out of 38, used directly | 27 | 27 | 28 | 34 |
| SWE-bench Verified, expected best of four | **71.3%** | 69.4% | 70.7% | 83.3% |
| Game, planner agreement with the shield on | 36.1% | **66.9%** | | |
| Game, courses survived with the shield on | 5 / 5 | 5 / 5 | | |

Our decision model, built from public data and recipes we reconstructed ourselves, matches CLM-8B on the coding benchmark it was designed for, beats it on tricky questions and on the external coding test, and falls behind on following the game's planner. The whole general post-training study, including encoding its new data and running all these evaluations, cost about \\$2.85 of GPU time.

# What we learned

## Know the floor and the ceiling

The most important number in any benchmark table is often the one that is missing: what a random choice would score. On DeepSWE, 25 of the 38 tasks are decided before the verifier even looks at them, a random pick already resolves 28, and the difference between the two models is a fraction of a task. On SWE-bench, a random pick beats one of the two models. Without those two reference points, "31.0 against 30.4" or "71.3% against 69.4%" look like results. With them, they look like what they are: small differences on a narrow margin.

## A benchmark can contain its own answer

The game's options carry the planner's notes, and the planner's favourite option literally ends with the word "Best.". A model that learns to look for that word scores perfectly on planner agreement without understanding anything about the game.

We checked this. We trained a specialist head on 8,640 planner-labeled decisions from other courses, adjusting only the action head. It reached **100%** planner agreement on fresh courses, against 66.9% for CLM-8B, and the shield had to intervene only 206 times. It looks like a triumph. But with the shield off, it crashed on every one of the five courses. It had learned to read the label, not to play. Planner agreement on this prompt measures how well a model follows a hint written in its options, which is worth knowing, but is not the same as making good decisions.

## Specialists win their benchmark and lose the others

We also trained a coding specialist: starting from an earlier post-trained head, it learned to prefer passing attempts over failing ones on 318 SWE-bench issues, kept separate from the 95 test issues. On the SWE-bench test, it reached 74.0% expected best-of-four success, by far the best result, almost three points above our general checkpoint. On DeepSWE, it fell to 28.8, below every other head we tested. The gain did not transfer from one set of coding tasks to another. This is why the checkpoint we report is the general one, chosen with no benchmark in sight: it is never the best at one test, but it holds up on all of them.

## Hard negatives are where the precision comes from

The opening animation is the clearest result of the whole project. Pre-training on 62 million pairs taught the heads to find the right answer among 1,024 random ones, but left them guessing between close alternatives. Forty-two steps on hard negatives did more for that than the last 25,000 steps of pre-training. And keeping the hard negatives in the replay during post-training is what let our final checkpoint keep improving on them, where CLM-8B's post-training lost a little.

## Fix the selection rule before looking

With dozens of candidate checkpoints and half a dozen benchmarks, it is easy to fool oneself by keeping whichever checkpoint looks best on the tests. We fixed the selection rule of part 4, on development data only, before running any benchmark, and stuck to it even where a different choice would have looked better on one test. The results above are less flattering than a hand-picked table would be, but they are the ones a new task would likely see.

## Frozen embeddings make experiments cheap

Because the language model is frozen and every text is embedded once, the cost of an idea is the cost of training two small heads. Our pre-training run, 30,000 steps on 62 million pairs, took two hours on one GPU. Mid-training reached its best checkpoint in 21 minutes. Each post-training recipe took minutes and cost cents. That is what made it possible to compare recipes side by side, on identical batches, and to throw away the ones that did not work.

## What we would do next

The weak point of our checkpoint is the gap between "plausible" and "best": it proposes dangerous actions less often than CLM-8B, but it often disagrees with the planner on which safe action is better. More preference data, where two good options must still be ranked, is the most direct fix. On the coding side, the verifier only ever learns from passing attempts. Training it on pairs of passing and failing attempts, as the coding specialist did, but on a broader mix of tasks, should keep the specialist's gains without its brittleness. And for evaluation, we would want coding benchmarks with many more mixed tasks, and game prompts that describe the situation without labelling the answer.

# Conclusion

A decision model is a simple idea: let a frozen language model read, let two small heads compare, and replace generation with a cosine. Over this series we built one from scratch: we embedded once, trained with a contrastive loss, followed a three-stage curriculum from 62 million questions to agents and preferences, and tested the result against the model we set out to replicate. It answers in 17 milliseconds, it can be retrained for cents, and on the tests that measure what it was trained for, it holds its own against CLM-8B.

The code is in the [repository](https://github.com/MKDYasserH/system-one-model). If you want to see a decision model built step by step on video, the Vizuara AI [walkthrough](https://www.youtube.com/watch?v=vEpkc44bHVQ&t=5549s) is a great companion to this series. Thank you for reading.

# References

[^clm]: J. Kwok, H. Kang, T. Suresh, J. Saad-Falcon, M. Pavone, C. Ré and A. Mirhoseini. *CLM-8B* ([project page](https://contrastive-lm.notion.site)) and its [code](https://github.com/Contrastive-LM/CLM), including the DeepSWE benchmark and the game.
[^cobbe2021]: K. Cobbe et al. *Training verifiers to solve math word problems.* [arXiv:2110.14168](https://arxiv.org/abs/2110.14168), 2021.
[^lightman2023]: H. Lightman et al. *Let's verify step by step.* International Conference on Learning Representations, 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
[^jimenez2024]: C. E. Jimenez et al. *SWE-bench: can language models resolve real-world GitHub issues?* International Conference on Learning Representations, 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
