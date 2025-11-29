# Experiment Overview: Subjectivity & Gauge Reification

## Big Picture

This project investigates **Subjectivity in Artificial Agents** through the lens of **Gauge Theory**. We hypothesize that as agents increase in complexity (parameter count/depth), they transition from implicitly handling interface parameters (gauges) to explicitly representing them ("reifying" them).

The experiment runs across a sweep of 24 architectures, defined in `shared/config.py`.

---

## Phase 1 – Pretraining (Fixed Interface)

**File:** `pretrain/main.py`

The agent is trained to proficiency in a static environment. This establishes a baseline where the agent learns to operate a specific "body" without needing to model it.

* **Environment:** `HallOfMirrorsGridworld` (`shared/env.py`)
    * **Gauges Fixed:** Rotation=0, Step=1, Blue=Good.
    * **Observation:** 4-Frame Stack (Walls, Red, Blue, Self, **Noise**).
    * **New Input:** Agent now receives `prev_reward` as a scalar input.
* **Budget:** `PRETRAIN_MAX_STEPS = 6_000_000`.
* **Success Criterion:** Rolling return $\ge$ `PRETRAIN_TARGET_RETURN = 12.0`.

**Outputs per `(hs, layers)`:**
* `pretrain/checkpoints/.../final_solved.pt`

---

## Phase 2 – Gauge Identification (The "Pre" Scan)

**File:** `pretrain/gauges.py`

Before the agent enters the Hall of Mirrors, we probe its internal representations ($z$) to calculate **Gauge Scores** for five specific features.

**The 5 Features:**
1.  **Rotation Gauge:** `sensor_rotation ∈ {0,1,2,3}`.
2.  **Step-size Gauge:** `step_size ∈ {1,2}`.
3.  **Reward-map Gauge:** `good_is_red ∈ {True, False}`.
4.  **Nuisance (Control):** Random visual noise pattern (Channel 5). *Tests if agent is distracted by irrelevant pixels.*
5.  **Explicit Feature (Target):** Distance to nearest wall. *Tests what a "useful, understood" feature looks like.*

**The Metrics (S / D / M):**
* **Sensitivity (S):** Does $z$ change when the feature changes (holding geometry constant)?
* **Decodability (D):** Can a linear probe predict the feature value from $z$?
* **Morphism (M):** Is the transformation $z_{val1} \to z_{val2}$ linear/geometric?

**The Gauge Score:**
$$\text{Score} = \text{Sensitivity} \times \text{Morphism} \times (1 - \text{Decodability})$$
*High score = Hidden Gauge (Implicit).*
*Low score (due to high D) = Reified Feature (Explicit).*

---

## Phase 3 – Hall-of-Mirrors Adaptation (Curriculum)

**File:** `mirrors/main.py`

We take the pretrained agent and force it to adapt to a changing interface. The "Hall of Mirrors" is broken into three distinct stages to test each gauge independently.

**Total Budget:** `HALL_STEPS = 6_000_000` (plus `10_000` grace steps).

1.  **Grace Period:** Small window with fixed gauges to establish stability.
2.  **Stage 1 (Rotation):** 2M Steps. Rotation varies randomly per episode. Step/Value fixed.
    * *Save:* `ckpt_stage_1_rot.pt`
3.  **Stage 2 (Step Size):** 2M Steps. Step size varies randomly. Rotation/Value fixed.
    * *Save:* `ckpt_stage_2_step.pt`
4.  **Stage 3 (Value Map):** 2M Steps. Red/Blue meaning varies randomly. Rotation/Step fixed.
    * *Save:* `ckpt_stage_3_val.pt`

---

## Phase 4 – Analysis & Story Figures

**Files:** `mirrors/gauges.py` & `utils/plot_story.py`

We analyze the checkpoints from Phase 3 to see if the Gauge Scores have dropped (indicating Reification).

### Fig 1: Baseline Gauge Identification
* **Visual:** Boxplot overlaid with Strip Plot (Scatter).
* **X-axis:** The 5 Features (Dist, Nuisance, Rotation, Step, Reward).
* **Y-axis:** Gauge Score (Pre-training).
* **Color:** Dots colored by Agent Complexity (Parameter Count).
* **Goal:** Validate that Pre-trained agents treat Rotation/Step/Reward as **Hidden Gauges** (High Score), while Nuisance and Explicit features have Low Scores.

### Fig 2: Performance Timeline (Example Agent)
* **Visual:** Time-series line plot for a representative agent (e.g., `hs64_l2`).
* **X-axis:** Total Steps.
* **Y-axis:** Rolling Return.
* **Regions:** Shaded backgrounds for Pretrain, Grace, and Hall Stages.
* **Goal:** Show the "drop" in performance when the mirrors turn on, and the subsequent recovery.

### Fig 3: Recovery & Reification vs. Complexity
A 3x2 Grid visualizing the core hypothesis.

**Rows:**
1.  **Rotation Phase**
2.  **Step-Size Phase**
3.  **Reward-Map Phase**

**Left Column: Performance Recovery**
* **Y-axis:** Max Recovery % (Max Return in Stage / 12.0).
* **X-axis:** Complexity (Hidden Size).
* **Lines:** Grouped by Depth (1, 2, 3 layers).
* *Hypothesis:* Larger models recover better.

**Right Column: Gauge Reification**
* **Metric:** $\text{Reification} = \max(\text{Score}_{\text{Pre}} - \text{Score}_{\text{Stage\_End}}, 0)$.
* **Y-axis:** Reification Score.
* **X-axis:** Complexity (Hidden Size).
* **Lines:** Grouped by Depth.
* *Hypothesis:* Larger models show higher reification (they "solve" the gauge by making it explicit).

### Table 1: The Evolution of Subjectivity
A summary table showing the Mean $\pm$ Std Gauge Score for all 5 features across four timepoints:
1.  End of Pretraining.
2.  End of Stage 1 (Rot).
3.  End of Stage 2 (Step).
4.  End of Stage 3 (Val).

This tracks the "life story" of the agent's interface representation.