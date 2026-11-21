---
title: Decision Models (System One) from Scratch
tagline: Teach a frozen LLM to choose between options in a single pass.
description: A fast, intuitive model that scores options instead of writing text. Frozen Qwen3-8B embeddings, two small heads, contrastive learning and a three-stage training recipe, explained with the maths and rebuilt end to end.
cover: assets/img/system_one/series_cover.png
level: Intermediate
status: complete
order: 2
---

Most of what we ask language models to do today goes through writing: the model thinks out loud, token after token, and the answer comes at the end. That is powerful, and slow. Many decisions do not need it. An agent picking its next command, a verifier choosing the best of four attempts, a game character deciding to jump: these are choices between options, and they can be made in one glance.

This series rebuilds a decision model, also called a System One model, that does exactly that. We keep a large language model frozen, train two small networks on top of it, and end up with a model that returns a probability for each option in a single pass. Each part mixes intuition, the equations that matter, and the code that trained it. Read them in order if you are new to contrastive learning, or jump to the part you need.
