# BehaviorHMM

A Hidden Markov Model that learns from sequences of raw behavioral observations and infers hidden emotional states. Feed it labeled examples, and it figures out transition patterns between states — then uses those patterns to decode new observations and predict what comes next.

Built in C++, no external dependencies.

---

## What it does

You give it observations like "she sent a long, positive message at 2pm after replying within 5 minutes" and a label like `happy`. Over time it learns what kinds of messages tend to come from which states, and how states tend to flow into each other. Then you give it a new sequence of unlabeled observations and ask: *what state is she in, and what happens next?*

Three core operations:

- **`learn_sequence`** — train on labeled `(hidden_state, observation)` pairs
- **`decode`** — run Viterbi to infer the most likely hidden state sequence for new observations  
- **`predict_next_observation`** — given recent observations, predict the most probable next one

---

## How observations work

Each `RawObservation` has six fields:

```cpp
struct RawObservation {
    std::string type;       // "positive_message", "negative_message", "neutral_message", "no_reply"
    int hour;               // 0-23
    int reply_delay_min;    // minutes since last message
    int msg_length;         // character count
    double sentiment;       // -1.0 (very negative) to +1.0 (very positive)
    bool she_initiated;     // true if she sent first
};
```

These get bucketed automatically into a string like `pos_afternoon_instant_medium_pos_init`. The vocabulary grows as new bucket combinations appear — you don't predefine it.

---

## Quick start

```cpp
// Define your hidden states
std::vector<std::string> states = {"happy", "neutral", "frustrated"};
BehaviorHMM hmm(states);

// Train on labeled sequences
std::vector<std::pair<std::string, RawObservation>> training = {
    {"happy",      {"positive_message", 14,   2, 120,  0.9, true}},
    {"happy",      {"positive_message", 15,   5,  95,  0.8, false}},
    {"neutral",    {"neutral_message",  10,  45,  20,  0.1, false}},
    {"frustrated", {"negative_message", 22, 180,   5, -0.7, false}},
    {"frustrated", {"no_reply",         23, 720,   0,  0.0, false}},
};
hmm.learn_sequence(training);

// Decode a new sequence
std::vector<RawObservation> recent = {
    {"positive_message", 14, 3, 110, 0.85, true},
    {"positive_message", 15, 4, 105, 0.82, false}
};

auto states_inferred = hmm.decode(recent);
auto next_obs = hmm.predict_next_observation(recent);
```

---

## Build

```bash
g++ -std=c++17 -O2 -o behavior_hmm behavior_hmm.cpp
./behavior_hmm
```

---

## Design notes

**Laplace smoothing** — all probability estimates add a pseudocount of 1, so unseen transitions and emissions don't collapse to zero.

**Time decay** — older training examples contribute less over time. The halflife is 30 days by default. Observations from a month ago count half as much as recent ones. Change `decay_halflife_days` to tune this.

**Dynamic vocabulary** — observation buckets are added to the vocabulary on the fly. The emission matrices grow automatically when new combinations appear, so you don't need to enumerate all possible observations upfront.

**Log-space Viterbi** — decoding runs in log space to avoid underflow on longer sequences.

**Prediction** — `predict_next_observation` properly marginalizes over the transition distribution: `P(next_obs = o) = Σⱼ P(cur → j) * P(o | j)`. It does not just read the current state's emissions directly.

---

## Bucket reference

| Field | Buckets |
|---|---|
| type | `pos`, `neg`, `neu`, `none` |
| hour | `night` (0-5), `morning` (6-11), `afternoon` (12-17), `evening` (18-23) |
| delay | `instant` (<5m), `fast` (<1h), `slow` (<6h), `very_slow` (6h+) |
| length | `very_short` (<10), `short` (<50), `medium` (<200), `long` (200+) |
| sentiment | `neg` (<-0.33), `neu`, `pos` (>0.33) |
| initiated | `init`, `reply` |

Example bucket: `pos_afternoon_instant_medium_pos_init`

---

## Limitations

The model is supervised — it needs labeled training data. With few examples, Laplace smoothing keeps it from crashing but the predictions won't mean much. It also treats each observation as a discrete bucket, so two messages with sentiment 0.34 and 0.9 look identical to the model. That's a reasonable tradeoff for simplicity, but worth knowing.