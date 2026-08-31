---
title: "Factory-grade agents: what changes when AI meets the physical world"
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

The model proposing to slow a filling line tonight is the same model that drafted your marketing email this morning. Same weights, same API, possibly the same context window. One of those two actions can be deleted.

Everything that matters about putting agents into factories lives inside that gap, and the gap deserves a name. I have started calling the systems on the far side of it factory-grade agents, and this essay is my attempt to pin down what the term means, where its edges are, and why the distinction earns its keep.

## The definition

> An agent is **factory-grade** when three things hold at once:
>
> 1. Its actions are costly to reverse, so correctness is enforced by machinery outside the model.
> 2. Its picture of the world can silently go stale, so every action must carry proof that its inputs were alive.
> 3. Its autonomy is earned in observable stages, and every action it takes can be attributed and audited afterward.

Each criterion is a test you can run from outside the system, without access to the weights or the prompts. That is deliberate. A definition you can only check from inside the vendor's codebase is marketing.

**Costly to reverse.** A wrong marketing email gets a correction. A wrong line-speed change has already consumed machine hours, moved material, and committed people by the time anyone reads the log. When undo stops being an option, the burden of correctness moves out of the model and into hard boundaries around it: envelopes on what values an action may take, contracts on what state it may touch, and tripwires that fire before consequences compound. I walk through that machinery in [my piece on evals for factory agents](https://siddharthksah.github.io/posts/2026/08/factory-agent-evals/); the definition only requires that it exists and sits outside the model's control.

Irreversibility also warps the error math. In chat, a bad answer and a missed good answer cost roughly the same: a shrug. In a plant, the two directions of error carry wildly different prices, and both are denominated in money. Halting a line on a false alarm burns measurable dollars a minute; waving through a real defect can cost a recall. Any single accuracy number an agent vendor quotes has averaged those two prices together, which is a tidy way of hiding the only figure the plant manager cares about.

**Stale-able world state.** A chat agent's world is the transcript, and the transcript never rots behind its back. A factory-grade agent reasons over sensors that freeze, an [MES](https://en.wikipedia.org/wiki/Manufacturing_execution_system) that lags the floor, and calibrations that drift between maintenance windows. The world it believes in and the world that exists diverge by default, which is why criterion two demands a freshness proof attached to every action.

**Earned, auditable autonomy.** Nobody grants a new hire the keys on day one, and the same applies to software that acts. Factory-grade agents climb a ladder: propose invisibly, then suggest, then act with approval, then act within measured fences. Every rung leaves records a person can inspect after the fact. An agent that cannot be audited fails the definition regardless of how clever it is, because in a plant, accountability outlives any single decision.

## What sits outside the line

Coding agents come closest, and the comparison is instructive. A coding agent's afternoon of mistakes reverses with `git revert`, its claims face interrogation by a test suite, and the blast radius of a bad day is a branch. Those are wonderful properties. They are also precisely the properties a factory refuses to provide.

Chat agents sit further out for the reason above: their world cannot go stale because their world is the conversation.

[Embodied agents](https://en.wikipedia.org/wiki/Embodied_agent), the robotics tradition, overlap but run on a different axis. There, the agent is the body, and the research centers on perception and motor control. A factory-grade agent typically commands infrastructure it does not inhabit: it is a mind with authority over machines, which is exactly why the authority needs fences.

The line blurs in interesting places, and the blur is a feature of defining by criteria instead of by industry. A coding agent with deploy rights to production infrastructure starts failing the reversibility test the moment its migrations touch customer data. An agent that books freight or moves money faces stale-able state and audit demands with no factory in sight. Run the three tests and let them decide; the answer matters more than the label on the building.

## What changes in the stack

| Dimension | Chat agent | Factory-grade agent |
|---|---|---|
| Undo | regenerate the answer | machine hours already spent |
| World state | the transcript | sensors that can lie |
| Ground truth | arrives with the next message | arrives hours later, priced in currency |
| Verification | a judge model or a reader | executable contracts and envelopes |
| Autonomy | full on day one | promoted rung by rung |
| Record | a chat log | an audit trail per action |

Each row reshapes a layer of the build. Perception stops meaning "call the API" and starts meaning reconciling an [OPC UA](https://en.wikipedia.org/wiki/OPC_Unified_Architecture) tag stream with what the MES claims and what the shift supervisor knows. Execution inherits the oldest lesson in distributed systems: actions must be [idempotent](https://en.wikipedia.org/wiki/Idempotence) or fenced, because retries against physical equipment have side effects.

Observability grows teeth too, because the audit criterion turns logging from a debugging aid into a legal requirement.

The architecture around all this also has to live inside a plant's existing hierarchy, the layered OT world the [Purdue model](https://en.wikipedia.org/wiki/Purdue_Enterprise_Reference_Architecture) describes, where the further down you reach, the older, stricter, and less forgiving the systems get. The agent is a guest in a building with load-bearing walls.

## Factories ran for fifty years without this

Anyone who has walked a plant floor will raise the fair objection early: manufacturing is already automated. A [PLC](https://en.wikipedia.org/wiki/Programmable_logic_controller) executes its ladder logic in milliseconds and has done so reliably since the 1970s. [SCADA](https://en.wikipedia.org/wiki/SCADA) systems supervise entire sites. [Lights-out factories](https://en.wikipedia.org/wiki/Lights_out_(manufacturing)) machine parts with nobody in the building. What, exactly, do agents add?

The honest answer starts with what classical automation is. All of it, from relay logic to the most modern SCADA stack, executes situations a person enumerated in advance. Inside the enumerated space it is deterministic and close to unbeatable, and factory-grade agents should leave that space alone.

The residue is the product opportunity. Everything unenumerated, the exception cascade, the customer order that breaks the schedule, the material lot that arrives out of spec, the coordination across four systems that were never designed to talk, today escalates to a human with a radio and twenty browser tabs. Agents are a bid to absorb part of that residue: the semi-structured judgment work that was too variable to hard-code and too constant to staff generously.

The division of labor this implies is worth stating plainly, because it doubles as a red-flag detector. Deterministic control belongs to deterministic systems forever; a millisecond safety interlock has no business waiting on a language model, and a vendor proposing to put one there has disqualified themselves. The agent's tier sits above the control layer, interpreting messy context and coordinating across systems, with the PLC's reflexes untouched below it. Respecting that boundary is half of what makes an agent welcome in the building.

Manufacturing even has a cultural precedent for the accountability half of the definition. Toyota gave every line worker the authority to stop production by pulling the [andon](https://en.wikipedia.org/wiki/Andon_(manufacturing)) cord, on the theory that stopping is cheap compared to shipping defects. A factory-grade agent enters the same social contract from the other side: it gets a hand near the cord only after it has proven, rung by rung, that it knows when to pull it.

## Why the bar is reachable now

Until recently the residue was out of reach because models could not hold the context. The unenumerated work runs on messy inputs: a maintenance note typed at 3 a.m., a supplier email, a schedule in one system contradicting inventory in another. Reading that residue is what large models became good at, and [the agent pattern](https://lilianweng.github.io/posts/2023-06-23-agent/) gave the reading hands.

So the bottleneck moved. The scarce thing today is trust infrastructure: the envelopes, freshness proofs, promotion ladders, and audit trails that let a plant manager say yes. That is why I define the category by its constraints. The model is the commodity in this story; the fences are the product.

Impressive demos were never the blocker, and anyone who has sat in the meeting knows it. A model has been able to draft a plausible reschedule for a while now. The demo dies in the second meeting, when someone from operations asks what happens when the input data was wrong, who signs off, and how anyone would know afterward. The three criteria are that meeting, written down. Teams that can answer them get pilots; teams that answer with model benchmarks get polite follow-up emails.

## The bar travels

Factories are where the criteria are most visible, but nothing in the definition mentions manufacturing. A warehouse robot fleet, a power grid dispatch assistant, a hospital pharmacy system: each one faces costly reversals, stale-able state, and the demand that autonomy be earned on the record. "Factory-grade" names the bar rather than the building, the way "production-grade" came to name a standard in software that long ago outgrew the server room.

I expect most agents will never need the bar, in the same way most code never needs to survive production traffic. The ones that touch the physical world will clear it or they will not ship, and the teams that internalize the three criteria early will be the ones a plant manager eventually trusts.

The follow-up question is how an agent actually earns its way up the ladder, and that discipline has [a piece of its own](https://siddharthksah.github.io/posts/2026/08/factory-agent-evals/): the promotion gates, the verifier stack, and the reasons a falling override rate should scare you. The two pieces are halves of one argument: this one names the bar, and that one is about clearing it.
