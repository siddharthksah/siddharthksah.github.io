---
title: "What deploying agentic AI on factory floors taught me about evals"
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

Picture a packaging line. An agent watches throughput, and this morning throughput looks wonderful: one temperature sensor has read exactly 42.7 degrees for six hours. The agent finds this very convincing. Stable temperature, healthy machine, room to push. It proposes raising the line rate.

The sensor is dead. It froze at 42.7 and kept repeating itself, the way dead sensors do. The machine behind it has been drifting hot all shift. If the proposal executes, the plant discovers this at the worst possible moment, at the highest possible speed.

What stops it is nothing clever. A precondition check refuses any action whose supporting readings are older than a threshold, and "the same value for six hours" fails a freshness test that a first-year engineer could write. No model got smarter. A twenty-line check did its job.

That is the shape of almost everything I have learned about evaluating agents in industrial settings: the interesting work is rarely in the model, and the things that save you are boring, explicit, and written down before anyone gets excited.

## The chatbot eval playbook doesn't transfer

The standard playbook for evaluating LLM products is genuinely good now. [Hamel Husain's essay](https://hamel.dev/blog/posts/evals/) is the canonical version: build evals specific to your product, look at your data, iterate fast. If you build chat products and haven't internalized it, do that before reading further.

But that playbook carries three silent assumptions, and a factory violates all of them.

**Retry is free.** A chatbot that answers badly gets a thumbs-down and regenerates. An agent that reschedules a production run has moved material and committed people to work. You cannot regenerate a shift. Once actions touch the physical world, evaluation stops being about answer quality and starts being about decision safety.

**Errors cost the same in both directions.** In chat, a false positive and a false negative are both just bad answers. On a line, the cost matrix is violently asymmetric. Stopping a line on a false alarm costs real money per minute. Missing a genuine defect can cost a recall. Any eval that reports a single accuracy number has averaged away the only thing that matters. You need the full confusion matrix priced in currency, and only the operations people can tell you the prices.

**Ground truth arrives on time.** Chat evals assume you can score an output when it appears. A scheduling decision made at 9 a.m. reveals its quality at 4 p.m., after downstream stations have eaten its consequences. Delayed, expensive ground truth breaks most online-evaluation loops you would naively borrow from the LLM world, and it means your offline suite carries far more weight than chat people are used to.

## The unit of evaluation is the trajectory

Single-response evals ask "was this output good?" An agent's product is a trajectory: observe, plan, act, observe again, replan. [Lilian Weng's agent survey](https://lilianweng.github.io/posts/2023-06-23-agent/) is a good map of the moving parts. The eval consequence is simple arithmetic: per-step reliability compounds. An agent that is 95% reliable per step completes a fourteen-step plan correctly about 49% of the time. Nobody would ship a coin flip, yet teams routinely report per-step accuracy and feel good.

Compounding is the obvious trajectory problem. The insidious one is the plan whose every step is fine and whose sum is not. A scheduling agent nudges one station's sequence to shave changeover time. Each swap passes every local check. Four hours later a downstream station starves because the new sequence quietly changed the arrival mix it was fed. Nothing in a per-action eval will ever catch this, because no single action was wrong.

So trajectory evals need two layers. [Process scoring](https://arxiv.org/abs/2305.20050) checks each step against its contract: was the tool call well-formed, was the state fresh, was the action inside its envelope. Outcome scoring checks the trajectory against reality hours later: did the plan's predicted state deltas actually happen, and what did it cost.

The gap between the two layers is where agents hide their failures. A trajectory can be process-clean and outcome-terrible at the same time, and that combination is the signature of the failure mode above.

## Treat the eval suite as a safety case

The framing that changed how I work is not from machine learning at all. Factories already have a discipline for systems that fail expensively: [FMEA](https://en.wikipedia.org/wiki/Failure_mode_and_effects_analysis), failure mode and effects analysis. You enumerate the ways a component can fail, score each for severity, likelihood, and detectability, and let the worst scores drive where you invest.

Safety engineers have run this loop since the 1950s. It transfers to agents almost unchanged, and it forces a question most eval suites never answer: *which failures are you not instrumented to detect?*

Run that analysis on an industrial agent and two failure modes dominate the ranking every time.

**World-state divergence.** The agent acts on a belief about the world that the world no longer honors. Frozen sensors. A [MES](https://en.wikipedia.org/wiki/Manufacturing_execution_system) record (the system tracking what is being made, where) that lags the floor by twenty minutes. An [OPC UA](https://en.wikipedia.org/wiki/OPC_Unified_Architecture) tag (the industrial protocol most machine data rides on) that silently changed units after a firmware update.

Chat people rarely think about this because their agent's world *is* the conversation; it cannot go stale behind their back. On a floor, state divergence is the default condition, and every action needs to carry proof that its inputs were alive.

**Locally valid, globally harmful.** The starvation example above. Every step defensible, the trajectory harmful, detection only possible at the plan level and often only in hindsight. This is the failure mode that forces trajectory-level evals on you.

The other lesson FMEA smuggles in is [defense in depth](https://en.wikipedia.org/wiki/Swiss_cheese_model): no single layer is trusted, and the layers must fail differently. That principle decides the next question, which is what your checks should even be made of.

## Verifiers beat judges

The fashionable answer to "how do we check agent outputs?" is LLM-as-judge: have a second model grade the first. It earned its place in chat evaluation ([the MT-Bench paper](https://arxiv.org/abs/2306.05685) is the standard reference), and I use judges where they belong. But for physical decisions the pattern has a structural problem that no amount of prompt tuning fixes: correctness on a factory floor is not a matter of preference, and a judge is a preference machine.

Whether an action is safe is a fact. The state was fresh or it wasn't. The rate is inside the envelope or it isn't. The window collides with maintenance or it doesn't. Facts should be checked by code, and code that checks facts has properties no judge will ever have: it is deterministic, it is auditable, it never has an off day, and when it fires you know exactly why.

The load-bearing pattern is twenty-odd lines of unglamorous Python. Every action an agent proposes is a typed object, and it passes through contracts before anything executes:

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
    require(age < MAX_STATE_AGE, f"state is {age}s old")
    require(not a.basis.is_flatlined(a.machine_id),
            "input sensor unchanged beyond plausibility window")

def check_envelope(a: ProposedAction, limits: Limits) -> None:
    lo, hi = limits[a.machine_id][a.kind]
    require(lo <= a.value <= hi, "outside approved envelope")

def check_calendar(a: ProposedAction, cal: MaintenanceCalendar) -> None:
    require(not cal.overlaps(a.machine_id, a.window),
            "collides with maintenance window")

def check_outcome(a: ProposedAction, after: StateSnapshot) -> None:
    require(after.delta(a.machine_id).within(a.basis.predicted_delta),
            "world disagreed with the plan")
```

Preconditions gate execution. The postcondition closes the loop: if reality keeps disagreeing with the agent's predictions, you have caught world-state divergence as a trend before it becomes an incident. None of this is novel software engineering. [Design by contract](https://en.wikipedia.org/wiki/Design_by_contract) is older than most people writing agents. The novelty is only in applying it without exception to a component that speaks fluent English and sounds sure of itself.

Two more verifier layers sit above contracts. [Temporal guards](https://en.wikipedia.org/wiki/Linear_temporal_logic) constrain sequences rather than single actions: never B within ten minutes of A, C must precede D. And where a [digital twin](https://en.wikipedia.org/wiki/Digital_twin) exists, you can run the plan in simulation first and diff the predicted state deltas against allowed ranges, with the standing caveat that a twin is a model, it drifts like any model, and [sim-to-real gaps](https://arxiv.org/abs/2009.13303) have a sense of humor about your confidence.

What about having a second LLM verify the first? Include it if you like, but do not count it toward your defense layers. The verifier model shares training data, tokenizer, and worldview with the model it checks; their failures correlate, which is precisely what defense in depth forbids. And judges themselves need constant validation against humans. [Shreya Shankar and colleagues showed](https://arxiv.org/abs/2404.12272) that even the humans grading the judges shift their criteria as they grade. A check whose checker needs checking is not where I want my last line of defense.

## Autonomy is earned in stages

Nobody sane connects an agent to actuators on day one. The deployment I trust is a ladder, and the honest way to see it is that each rung is itself an eval, run against production reality, with promotion criteria written down before you start climbing.

**Shadow mode.** The agent sees real state and proposes, invisibly. Operators keep deciding. The metric here is counterfactual agreement: how often did the agent's proposal match what the humans did, scored against outcomes where you can get them. Almost nobody writes about this stage, and it is the most information-dense rung of the whole ladder.

Every disagreement is a gift with two possible readings: the agent is wrong, or the agent found something the process missed. Adjudicate every one, by hand, with the people who made the call. This is also where your scenario suite starts accumulating real cases.

**Advisory mode.** Proposals become visible suggestions. The headline metric flips to override rate, and here is the twist that took me longest to internalize: the override rate falling is not automatically good news. Operators habituate. Suggestion quality earns trust, trust becomes rubber-stamping, and one day your human safety layer has quietly become a pass-through.

The literature calls it [automation bias](https://en.wikipedia.org/wiki/Automation_bias), and [aviation has scar tissue](https://99percentinvisible.org/episode/children-of-the-magenta-automation-paradox-pt-1/) about it going back decades. A 99% acceptance rate is an alarm. Counter it deliberately: sample decisions for forced independent review, audit the accepted ones, and treat "overrides went to zero overnight" as an incident.

**Gated autonomy.** The agent acts, a human approves. Latency of approval becomes a real constraint, and tripwire rate becomes the metric: how often did contracts fire per thousand actions, and is that trending down.

**Bounded autonomy.** The agent executes within envelopes, contracts armed, rollback rehearsed. It operates inside a fence whose posts you measured yourself.

### Promotion gates

| Stage | Agent may | Promotion gate |
|---|---|---|
| Shadow | propose, invisibly | counterfactual agreement above your threshold across a defined window; every disagreement adjudicated |
| Advisory | suggest to operators | override rate stable *and explained*; zero contract violations; automation-bias audits in place and passing |
| Gated | act with per-action approval | tripwire rate flat or falling; approval latency sustainable for operations |
| Bounded | act within envelopes | incident-free window at target volume; rollback drill actually performed; envelopes re-reviewed and re-signed |

Pick thresholds you can defend in a safety review, write them down before shadow mode starts, and do not renegotiate them after the fact to fit the data. Moving the goalposts post hoc is the eval-world version of [p-hacking](https://en.wikipedia.org/wiki/Data_dredging), and everyone can smell it.

## Production is the only benchmark that counts

Offline benchmarks matter, but be clear about what they are for. The Princeton group's [AI Agents That Matter](https://arxiv.org/abs/2407.01502) documented how thin the connection is between agent benchmark scores and real usefulness, and factories add a problem benchmarks cannot model at all: non-stationarity.

Sensors drift, machines wear, and the product mix shifts with the season. The distribution your suite froze in January is fiction by June. A benchmark notices when an agent gets worse. The world changing underneath an unchanged agent is invisible to it.

So the suite's real job is regression, and the pipeline that keeps it honest is [incident-to-scenario](https://sre.google/sre-book/postmortem-culture/): every tripwire fire, every adjudicated disagreement from shadow mode, every near miss becomes a permanent test case with the state snapshot that produced it. The suite only grows. Give it a year and it becomes the most valuable artifact the whole effort owns, worth more than the agent, because it survives model swaps and the agent does not.

And when you change anything, [canary](https://sre.google/workbook/canarying-releases/) like you mean it: one line, one machine, one shift, with the previous configuration warm behind a switch you have actually flipped in anger at least once.

## Five rules I would start with

1. Score whole trajectories. Per-step accuracy on a multi-step plan is a vanity metric.
2. Facts get code, preferences get judges. Never let a preference machine check a fact.
3. Run FMEA on your agent before your first deployment meeting. Detectability scores tell you which evals to build.
4. Write promotion gates down before shadow mode, and treat a plummeting override rate as an alarm.
5. Turn every near miss into a permanent regression case. The suite is the asset; the model is a tenant.

None of this is glamorous. That is rather the point. The teams that get agents onto factory floors safely are the ones that imported forty years of [safety engineering](https://en.wikipedia.org/wiki/Safety_engineering) and refused to be impressed by fluent English.
