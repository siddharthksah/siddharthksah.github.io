---
title: "Factory-grade agents: a definition"
date: 2026-05-24
categories:
  - ai-engineering
permalink: /posts/2026/05/factory-grade-agents/
tags:
  - Agentic AI
  - Manufacturing
  - Factory-Grade Agents
---

*Everything in this post is illustrative. The scenarios are constructed to teach patterns; nothing here describes any specific system, plant, or employer.*

The same model that drafts a marketing email can propose slowing a filling line. A bad email gets deleted. A bad line change has already used up machine time and moved material by the time anyone reads the log.

That difference is what this post is about. I've started calling the systems on the far side of it factory-grade agents, and this is my attempt to pin down what the term means, what the machinery around such an agent has to look like, and why I think the distinction is useful.

## The definition

> An agent is **factory-grade** when three things hold at once:
>
> 1. Its actions are costly to reverse, so correctness is enforced by machinery outside the model.
> 2. Its picture of the world can silently go stale, so every action must carry proof that its inputs were alive.
> 3. Its autonomy is earned in observable stages, and every action it takes can be attributed and audited afterward.

Each criterion is a test you can run from outside the system, without access to the weights or the prompts. That's on purpose. A definition you can only check from inside the vendor's codebase is no use to the person buying the system.

**Costly to reverse.** A wrong marketing email gets a correction. A wrong line-speed change has already consumed machine hours, moved material, and committed people by the time anyone reads the log. When undo stops being an option, the burden of correctness moves out of the model and into hard boundaries around it: limits on what values an action may take, contracts on what state it may touch, and checks that stop an action before its consequences compound. I go through that machinery in [my post on evals for factory agents](https://siddharthksah.github.io/posts/2026/08/factory-agent-evals/). The definition only requires that it exists and sits outside the model's control.

Irreversibility also changes how errors cost. In chat, a bad answer and a missed good answer cost about the same. In a plant, the two directions of error carry very different prices, and both are in money. Halting a line on a false alarm costs money by the minute. Waving through a real defect can cost a recall. Any single accuracy number a vendor quotes has averaged those two prices together, which hides the one figure the plant manager cares about.

The right way to set a decision threshold is to price both errors and apply the Bayes rule: act when P(defect) > c_FP / (c_FP + c_FN), where c_FP is the cost of a false stop and c_FN the cost of a missed defect. If a false stop costs 2,000 in lost output and a missed defect costs 50,000 in scrap and rework, the threshold is 2,000 / 52,000, about 4%. The agent should stop the line at a 4% belief in a defect, which no accuracy-maximizing classifier would ever do. The threshold is a business number, and only the operations people can supply the two costs.

**Stale-able world state.** A chat agent's world is the transcript, and the transcript never changes behind its back. A factory-grade agent reasons over sensors that freeze, an [MES](https://en.wikipedia.org/wiki/Manufacturing_execution_system) that lags the floor, and calibrations that drift between maintenance windows. The world it believes in and the world that exists diverge by default. That's why criterion two demands a freshness proof attached to every action.

A freshness proof is a small, concrete thing. Every state snapshot the agent reasons over carries, per sensor tag, the time of the last observed change and how much the value moved over the last window. A tag is live when it's younger than a bound and has moved more than the sensor's own resolution. The age bound comes from the process: a thermal loop with a ten-minute time constant can't drift meaningfully in sixty seconds, so sixty seconds is the bound for that tag, while a fill-level sensor on a fast line gets five. The proof travels with the proposed action, so the verifier can reject an action whose inputs were already dead when the agent reasoned over them.

**Earned, auditable autonomy.** A new operator doesn't get full authority on day one, and neither should software. Factory-grade agents go through stages: propose invisibly, then suggest, then act with approval, then act within set limits. Every stage leaves records a person can inspect afterwards. An agent that can't be audited fails the definition however clever it is, because in a plant someone has to answer for a decision long after it was made.

The audit record is an append-only log, one entry per proposed action, whether or not it executed. Each entry carries the action, the freshness proof it was based on, the verdict of every contract that checked it, who approved it, and the observed outcome once it's known. Each entry also includes a hash of the previous one, so a deleted or edited entry is detectable. That's enough for a safety review to reconstruct any decision months later, and it's enough to answer the question the plant manager asks first after an incident: who knew what, and when.

## Coding agents, chat agents, and robots

Coding agents come closest, and the comparison is useful. A coding agent's mistakes can be reverted with git, its claims get checked by a test suite, and the damage is contained to a branch. A factory provides none of those.

Chat agents sit further out for the reason above. Their world can't go stale because their world is the conversation.

[Embodied agents](https://en.wikipedia.org/wiki/Embodied_agent), the robotics tradition, overlap but run on a different axis. There the agent is the body, and the research centers on perception and motor control. A factory-grade agent typically commands infrastructure it doesn't inhabit. It has authority over machines it isn't part of, and that's why the authority needs limits.

The line blurs in interesting places, and that's what you get from defining by criteria. A coding agent with deploy rights to production infrastructure starts failing the reversibility test the moment its migrations touch customer data. An agent that books freight or moves money faces stale-able state and audit demands with no factory in sight. Run the three tests and let them decide.

## What changes in the stack

| Dimension | Chat agent | Factory-grade agent |
|---|---|---|
| Undo | regenerate the answer | machine hours already spent |
| World state | the transcript | sensors that can be wrong |
| Ground truth | arrives with the next message | arrives hours later, with a cost attached |
| Verification | a judge model or a reader | executable contracts and limits |
| Autonomy | full on day one | granted in stages |
| Record | a chat log | an audit trail per action |

Each row reshapes a layer of the build. A reference architecture for the whole thing has six parts, and the order matters:

1. **Perception** pulls tag streams over [OPC UA](https://en.wikipedia.org/wiki/OPC_Unified_Architecture) subscriptions, reconciles them with what the MES claims and what the shift supervisor logged, and stamps every value with the freshness fields above.
2. **Belief state** is a typed snapshot, versioned, immutable once the agent starts reasoning over it.
3. **The planner** is where the model lives. It reads the snapshot and emits typed proposed actions, each pointing at the snapshot it reasoned over.
4. **Contracts** check every proposal against freshness, limits, calendars, and sequence rules, in code, before anything moves.
5. **The executor** turns an approved action into commands, with idempotency keys so a retry can't double-apply.
6. **Audit** writes the hash-chained record and, hours later, the observed outcome.

The sequence for one action is propose, verify, approve, execute, observe, and then the postcondition check that compares what happened to what the plan predicted.

Two of those parts carry lessons from older fields. Perception stops meaning "call the API" and starts meaning managing sensor subscriptions, where a change threshold set too wide is a sensor that reports nothing while the process drifts inside it. Execution inherits the oldest lesson in distributed systems. Commands to physical equipment must be [idempotent](https://en.wikipedia.org/wiki/Idempotence), because a timeout doesn't tell you whether the command arrived. Every action carries a unique ID, the executor keeps a table of applied IDs, and a retry with a seen ID is a no-op. Where two executors could race, each command also carries a sequence number, and the equipment gateway rejects anything older than the last one it applied.

Logging changes too. The audit criterion turns it from a debugging aid into a requirement.

The architecture also has to live inside a plant's existing hierarchy, the layered OT world the [Purdue model](https://en.wikipedia.org/wiki/Purdue_Enterprise_Reference_Architecture) describes, where the further down you reach, the older, stricter, and less forgiving the systems get. The agent lives in the upper layers, talks down through a gateway, and never holds a direct connection to a controller.

## What agents add to existing automation

Anyone who has walked a plant floor will raise the obvious objection early: manufacturing is already automated. A [PLC](https://en.wikipedia.org/wiki/Programmable_logic_controller) executes its ladder logic in milliseconds and has done so reliably since the 1970s. [SCADA](https://en.wikipedia.org/wiki/SCADA) systems supervise entire sites. [Lights-out factories](https://en.wikipedia.org/wiki/Lights_out_(manufacturing)) machine parts with nobody in the building. What do agents add?

Classical automation, from relay logic to the most modern SCADA stack, executes situations a person enumerated in advance. Inside the enumerated space it's deterministic and close to unbeatable, and factory-grade agents should leave that space alone.

The unenumerated cases are where agents can help. The exception cascade, the customer order that breaks the schedule, the material lot that arrives out of spec, the coordination across four systems that were never designed to talk: today all of that escalates to a person who handles it by phone and across several systems. Agents can take on part of that work, the semi-structured judgment calls that were too variable to hard-code and too constant to staff generously.

The division of labor this implies also works as a test of vendors. Deterministic control belongs to deterministic systems. A millisecond safety interlock has no business waiting on a language model, and a vendor proposing to put one there should be ruled out. The agent's tier sits above the control layer, interpreting messy context and coordinating across systems, with the PLC's control loops untouched below it. Respecting that boundary is a large part of getting an agent accepted.

Manufacturing even has a precedent for the accountability half of the definition. Toyota gave every line worker the authority to stop production by pulling the [andon](https://en.wikipedia.org/wiki/Andon_(manufacturing)) cord, on the theory that stopping is cheap compared to shipping defects. A factory-grade agent should be held to the same standard. It gets the authority to stop a line only after it has shown, stage by stage, that it uses that authority correctly.

## Why now

Until recently the unenumerated work was out of reach because models couldn't hold the context. It runs on messy inputs: a maintenance note typed at 3 a.m., a supplier email, a schedule in one system contradicting inventory in another. Reading that is what large models became good at, and [the agent pattern](https://lilianweng.github.io/posts/2023-06-23-agent/) let a model act on what it read.

So the bottleneck moved. The scarce thing today is trust infrastructure: the limits, freshness proofs, staged rollouts, and audit trails that let a plant manager say yes. That's why I define the category by its constraints. The model is interchangeable. The infrastructure around it is the hard part.

Impressive demos were never the blocker. A model has been able to draft a plausible reschedule for a while now. The demo stops being convincing in the second meeting, when someone from operations asks what happens when the input data was wrong, who signs off, and how anyone would know afterwards. The three criteria are those questions, written down. Teams that can answer them get pilots.

## Beyond factories

Factories are where the criteria are most visible, but nothing in the definition mentions manufacturing. A warehouse robot fleet, a power grid dispatch assistant, a hospital pharmacy system: each one faces costly reversals, stale-able state, and the demand that autonomy be earned on the record. "Factory-grade" names a standard, the way "production-grade" does in software.

I expect most agents will never need this standard, in the same way most code never needs to survive production traffic. The ones that touch the physical world will clear it or they won't ship.

The follow-up question is how an agent gets from one stage to the next, and that has [a post of its own](https://siddharthksah.github.io/posts/2026/08/factory-agent-evals/): the promotion gates, the verifier stack, and why a falling override rate should worry you. This post defines the standard. That one is about meeting it.
