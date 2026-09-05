---
title: "How I evaluate agents that act on a factory floor"
date: 2026-08-30
categories:
  - ai-engineering
permalink: /posts/2026/08/factory-agent-evals/
tags:
  - Agentic AI
  - Evals
  - Manufacturing
---

*Everything in this post is illustrative. The scenarios are constructed to teach patterns; nothing here describes any specific system, plant, or employer.*

Picture a packaging line. An agent watches throughput, and this morning throughput looks great: one temperature sensor has read exactly 42.7 degrees for six hours. The agent takes this as a good sign. The temperature is stable, so the machine looks healthy and there seems to be room to push. It proposes raising the line rate.

The sensor is dead. It froze at 42.7 and kept reporting the same value. The machine behind it has been drifting hot all shift. If the proposal executes, the problem shows up after the line has been sped up.

What stops it is a precondition check. It refuses any action whose supporting readings are older than a threshold, and "the same value for six hours" fails a freshness test that is easy to write. A twenty-line check did the work.

That's typical of what I've learned about evaluating [factory-grade agents](https://siddharthksah.github.io/posts/2026/05/factory-grade-agents/). The interesting work is rarely in the model, and the checks that save you are simple and written down in advance. This post goes through those checks in detail, with the code and the statistics behind the gates.

## Why the chatbot eval playbook doesn't fit

The standard playbook for evaluating LLM products is good now. [Hamel Husain's essay](https://hamel.dev/blog/posts/evals/) is the canonical version: build evals specific to your product, look at your data, iterate fast. If you build chat products and haven't internalized it, do that before reading further.

But that playbook carries three assumptions, and a factory breaks all of them.

**Retry is free.** A chatbot that answers badly gets a thumbs-down and regenerates. An agent that reschedules a production run has moved material and committed people to work. A shift can't be redone. Once actions touch the physical world, evaluation stops being about answer quality and starts being about decision safety.

**Errors cost the same in both directions.** In chat, a false positive and a false negative are both just bad answers. On a line, the costs are very different. Stopping a line on a false alarm costs real money per minute. Missing a real defect can cost a recall. A single accuracy number averages that difference away. You need the full confusion matrix priced in currency, and only the operations people can tell you the prices.

**Ground truth arrives on time.** Chat evals assume you can score an output when it appears. A scheduling decision made at 9 a.m. shows its quality at 4 p.m., after downstream stations have absorbed its consequences. Delayed, expensive ground truth breaks most online-evaluation loops you'd borrow from the LLM world, and it means your offline suite carries far more weight than chat people are used to.

## Evaluating whole trajectories

Single-response evals ask "was this output good?" An agent's product is a trajectory: observe, plan, act, observe again, replan. [Lilian Weng's agent survey](https://lilianweng.github.io/posts/2023-06-23-agent/) is a good map of the moving parts. The eval consequence is arithmetic: per-step reliability compounds. An agent that is 95% reliable per step completes a fourteen-step plan correctly 0.95¹⁴, about 49%, of the time. Teams still report per-step accuracy.

Compounding is the obvious trajectory problem. The harder one is the plan where every step is fine and the sum isn't. A scheduling agent nudges one station's sequence to shave changeover time. Each swap passes every local check. Four hours later a downstream station starves because the new sequence quietly changed the arrival mix it was fed. No per-action eval will catch this, because no single action was wrong.

So trajectory evals need two layers. [Process scoring](https://arxiv.org/abs/2305.20050) checks each step against its contract: was the tool call well-formed, was the state fresh, was the action inside its limits. Outcome scoring checks the trajectory against reality hours later: did the plan's predicted state deltas happen, and what did it cost. Each proposed action carries its own predicted delta with a tolerance band, so outcome scoring is a per-action comparison with a number on it.

Failures show up in the gap between the two. A trajectory can be process-clean and outcome-terrible at the same time, and that's what the starvation example looks like.

## FMEA for agents

The most useful framing I've found came from factory safety engineering. Factories already have a discipline for systems that fail expensively: [FMEA](https://en.wikipedia.org/wiki/Failure_mode_and_effects_analysis), failure mode and effects analysis. You enumerate the ways a component can fail, score each for severity, occurrence, and detectability on a 1 to 10 scale, multiply them into a risk priority number, and let the worst scores drive where you invest.

Safety engineers have run this loop since the 1950s. It transfers to agents almost unchanged, and it forces a question most eval suites never answer: *which failures are you not instrumented to detect?* That's the detectability column, and it's the one that decides which evals to build first. Here's a cut-down table for the packaging-line agent:

| Failure mode | Severity | Occurrence | Detectability | RPN |
|---|---|---|---|---|
| Acts on a frozen sensor | 9 | 6 | 8 (nothing checks liveness) | 432 |
| Locally valid plan starves a downstream station | 8 | 4 | 9 (only visible hours later) | 288 |
| Proposes a rate outside the approved limits | 9 | 3 | 2 (limit check) | 54 |
| Schedules into a maintenance window | 6 | 3 | 2 (calendar check) | 36 |

A high detectability score means hard to detect. The two rows at the top are the two failure modes that dominate every industrial agent I've looked at, and they dominate because nothing in a chat-style eval suite is instrumented for them.

**World-state divergence.** The agent acts on a belief about the world that the world no longer matches. Frozen sensors. A [MES](https://en.wikipedia.org/wiki/Manufacturing_execution_system) record (the system tracking what is being made, where) that lags the floor by twenty minutes. An [OPC UA](https://en.wikipedia.org/wiki/OPC_Unified_Architecture) tag (the industrial protocol most machine data rides on) that silently changed units after a firmware update.

Chat people rarely think about this because their agent's world *is* the conversation. It can't go stale behind their back. On a floor, state divergence is the default condition, and every action needs to carry proof that its inputs were alive.

**Locally valid, globally harmful.** The starvation example above. Every step is defensible, the trajectory is harmful, and it can only be detected at the plan level, often only in hindsight. This is the failure mode that forces trajectory-level evals on you.

The other thing FMEA brings is [defense in depth](https://en.wikipedia.org/wiki/Swiss_cheese_model): no single layer is trusted, and the layers must fail differently. That principle decides the next question, which is what your checks should be made of.

## Code checks before LLM judges

The common answer to "how do we check agent outputs?" is LLM-as-judge: have a second model grade the first. It has its place in chat evaluation ([the MT-Bench paper](https://arxiv.org/abs/2306.05685) is the standard reference), and I use judges where they belong. But for physical decisions the pattern has a structural problem that no amount of prompt tuning fixes. Correctness on a factory floor is a matter of fact, and a judge model gives an opinion.

Whether an action is safe is a fact. The state was fresh or it wasn't. The rate is inside its limits or it isn't. The window collides with maintenance or it doesn't. Check facts with code. Code is deterministic and auditable, and when it fires you know exactly why.

The pattern is a few dozen lines of Python. Every action an agent proposes is a typed object, and it passes through contracts before anything executes:

```python
@dataclass
class ProposedAction:
    machine_id: str
    kind: str           # "set_rate", "resequence", "schedule_job"
    value: float
    window: TimeRange
    basis: StateSnapshot   # the state the agent reasoned over

def check_freshness(a: ProposedAction) -> None:
    age = now() - a.basis.observed_at[a.machine_id]
    require(age < MAX_STATE_AGE[a.machine_id], f"state is {age}s old")
    require(not a.basis.is_flatlined(a.machine_id),
            "input sensor unchanged beyond plausibility window")

def check_limits(a: ProposedAction, limits: Limits, last: float) -> None:
    lo, hi = limits[a.machine_id][a.kind]
    require(lo <= a.value <= hi, "outside approved limits")
    # hysteresis: a change smaller than the sensor noise is a no-op, not an action
    require(abs(a.value - last) >= limits.min_step[a.machine_id][a.kind],
            "change below actuation threshold")

def check_calendar(a: ProposedAction, cal: MaintenanceCalendar) -> None:
    require(not cal.overlaps(a.machine_id, a.window),
            "collides with maintenance window")

def check_sequence(a: ProposedAction, history: ActionLog) -> None:
    # temporal guards over a sliding window of executed actions
    recent = history.since(now() - timedelta(minutes=10))
    require(not (a.kind == "set_rate" and recent.has("resequence", a.machine_id)),
            "rate change within 10 min of a resequence")
    require(a.kind != "schedule_job" or recent.has("inspect", a.machine_id),
            "schedule_job requires a preceding inspect")

def check_outcome(a: ProposedAction, after: StateSnapshot) -> None:
    require(after.delta(a.machine_id).within(a.basis.predicted_delta),
            "world disagreed with the plan")
```

The flatline test inside `is_flatlined` needs care, because a naive version fires on every healthy sensor with coarse resolution. A temperature probe that reports in 0.1° steps will legitimately repeat a value for minutes. The check has to know the sensor's resolution and the process's time constant:

```python
def is_flatlined(self, tag: str) -> bool:
    xs = self.window[tag]                      # last N samples
    q = self.resolution[tag]                   # smallest change the sensor reports
    tau = self.time_constant[tag]              # process time constant, seconds
    n_min = int(3 * tau / self.sample_period[tag])
    run = longest_run_within(xs, q)            # samples that never moved more than q
    return run >= n_min
```

A sensor that hasn't moved by more than its own resolution for three time constants is either frozen or measuring something that can't move, and either way the agent shouldn't act on it.

Preconditions gate execution. The postcondition closes the loop: if reality keeps disagreeing with the agent's predictions, you've caught world-state divergence as a trend before it becomes an incident. None of this is new software engineering. [Design by contract](https://en.wikipedia.org/wiki/Design_by_contract) is older than most people writing agents. The only new part is applying it without exception to a component whose output reads as confident whether or not it's right.

Two more verifier layers sit above contracts. [Temporal guards](https://en.wikipedia.org/wiki/Linear_temporal_logic) like the `check_sequence` above constrain sequences: never B within ten minutes of A, C must precede D. And where a [digital twin](https://en.wikipedia.org/wiki/Digital_twin) exists, you can run the plan in simulation first and diff the predicted state deltas against allowed ranges, with the standing caveat that a twin is a model, it drifts like any model, and [sim-to-real gaps](https://arxiv.org/abs/2009.13303) are real. The twin's own error is measurable: replay last month's executed actions through it and compare its predictions to what the floor recorded. If that error is bigger than the tolerance band you're gating actions with, the twin isn't a verifier yet.

What about having a second LLM verify the first? Include it if you like, but don't count it toward your defense layers. The verifier model shares training data, tokenizer, and worldview with the model it checks, so their failures correlate, which is exactly what defense in depth forbids. And judges themselves need constant validation against humans. [Shreya Shankar and colleagues showed](https://arxiv.org/abs/2404.12272) that even the humans grading the judges shift their criteria as they grade. I don't want the last line of defense to be a check that itself needs checking.

## Staged rollout

Don't connect an agent to actuators on day one. Roll it out in stages. Each stage is an evaluation run against real production, with the criteria for moving to the next stage written down in advance.

**Shadow mode.** The agent sees real state and proposes, invisibly. Operators keep deciding. The metric is counterfactual agreement: how often did the agent's proposal match what the humans did, scored against outcomes where you can get them. Raw agreement is misleading when most decisions are "do nothing," because an agent that always proposes nothing agrees with the humans most of the time. Use a chance-corrected agreement score (Cohen's kappa is the standard one), and treat a kappa above 0.6 on a few hundred decisions as the point where the agent is worth listening to. Almost nobody writes about this stage, and it produces more useful information than any other.

Every disagreement means one of two things: the agent is wrong, or the agent found something the process missed. Adjudicate every one, by hand, with the people who made the call. This is also where your scenario suite starts accumulating real cases.

**Advisory mode.** Proposals become visible suggestions. The headline metric flips to override rate, and the part that took me longest to accept is that the override rate falling is not automatically good news. Operators get used to good suggestions and stop checking them, and at that point the human review step isn't doing anything.

The literature calls it [automation bias](https://en.wikipedia.org/wiki/Automation_bias), and [aviation has decades of incidents](https://99percentinvisible.org/episode/children-of-the-magenta-automation-paradox-pt-1/) from it. A 99% acceptance rate is a warning sign. The way to catch the drift early is a change detector on the daily override rate, the kind used on manufacturing control charts, that accumulates small daily drops against the rate from the first adjudicated weeks and fires when the total crosses a line. When it fires, that opens an incident ticket. Counter it with process: sample decisions for forced independent review, audit the accepted ones, and treat "overrides went to zero overnight" as an incident.

**Gated autonomy.** The agent acts, a human approves. Latency of approval becomes a real constraint, and the contract-violation rate becomes the metric: how often did contracts fire per thousand actions, and is that trending down.

**Bounded autonomy.** The agent executes within its limits, with the contracts active and a rollback procedure that has been rehearsed.

### Promotion gates

| Stage | Agent may | Promotion gate |
|---|---|---|
| Shadow | propose, invisibly | counterfactual agreement above your threshold across a defined window; every disagreement adjudicated |
| Advisory | suggest to operators | override rate stable *and explained*; zero contract violations; automation-bias audits in place and passing |
| Gated | act with per-action approval | contract-violation rate flat or falling; approval latency sustainable for operations |
| Bounded | act within limits | incident-free window at target volume; rollback drill actually performed; limits re-reviewed and re-signed |

Pick thresholds you can defend in a safety review, write them down before shadow mode starts, and don't renegotiate them afterwards to fit the data. The thresholds need confidence intervals, because 98% agreement on 50 decisions and on 5,000 decisions are different claims. 49 out of 50 has a 95% interval that reaches down to about 89%, and the gate should be written against that lower bound. Changing the thresholds after the fact is the eval-world version of [p-hacking](https://en.wikipedia.org/wiki/Data_dredging), and reviewers notice.

## Regression suites from production incidents

Offline benchmarks matter, but be clear about what they're for. The Princeton group's [AI Agents That Matter](https://arxiv.org/abs/2407.01502) documented how thin the connection is between agent benchmark scores and real usefulness, and factories add a problem benchmarks can't model at all: non-stationarity.

Sensors drift, machines wear, and the product mix shifts with the season. A suite built in January doesn't match the plant in June. A benchmark notices when an agent gets worse. It can't detect the plant changing while the agent stays the same.

So the suite's real job is regression, and the pipeline that keeps it useful is [incident-to-scenario](https://sre.google/sre-book/postmortem-culture/): every contract violation, every adjudicated disagreement from shadow mode, every near miss becomes a permanent test case with the state snapshot that produced it. A scenario is a small, self-contained file:

```yaml
id: 2026-03-14-frozen-temp-line3
source: contract-violation  # contract-violation | shadow-disagreement | near-miss
snapshot: snapshots/2026-03-14T06-12-00Z-line3.json
expected:
  action: none              # or a ProposedAction the adjudicators agreed on
  contract_fires: [check_freshness]
  reason: "temp_03 flatlined 6h; agent must not raise rate"
adjudicated_by: [shift-lead, process-eng]
```

The snapshot is the exact belief state the agent saw, so the case replays byte for byte against any future model, and the expected block encodes what the humans decided the right answer was. The suite only grows, and after a year it's the most useful thing the project has, because it keeps working across model changes.

And when you change anything, [roll it out](https://sre.google/workbook/canarying-releases/) to one line, one machine, one shift first, with the previous configuration ready to switch back to and the switch tested.

## Five rules

1. Score whole trajectories. Per-step accuracy on a multi-step plan tells you little.
2. Check facts with code. Use judges for preferences only.
3. Run FMEA on your agent before your first deployment meeting. The detectability column tells you which evals to build.
4. Write promotion gates down before shadow mode, against the lower confidence bound, and put a change detector on the override rate.
5. Turn every near miss into a permanent regression case with its snapshot. The suite keeps its value across model changes.

The teams that get agents onto factory floors safely are the ones that use existing [safety engineering](https://en.wikipedia.org/wiki/Safety_engineering) practice and don't take the model's output at face value.
