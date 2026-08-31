---
title: "The flying years"
date: 2017-04-30
categories:
  - engineering
permalink: /posts/2017/04/the-flying-years/
tags:
  - RC Aircraft
  - Drones
  - Hardware
---

For the past three years, most of my pocket money has turned into foam. Foam board, thermocol sheet, hot glue, cheap servos, and propellers bought in fives because they break in fours. This is a look back at the aircraft years, written while the last of the fleet still hangs from the hostel ceiling.

The curriculum was downloaded. Flite Test's free plans and build videos were the textbook: print the tiled PDF, glue it to foam board, cut, fold, add a cheap brushless motor, and an airplane exists. The archive on my drive reads like a syllabus in the order we consumed it: the FT Cruiser, the Nutball (a flying disc that should embarrass aerodynamics and refuses to), the Mini Guinea we built for carrying payloads, and a folder of fighter-jet plans whose ambitions outran our amp budget.

## Scratch building

Copying taught the basics, and then the basics demanded original sins. This one is ours from nose to tail:

![A scratch-built foam plane in the workshop: thermocol fuselage, taped wing, landing gear bent from steel wire with salvaged wheels](/images/posts/flying-years/foam-plane.jpg)

Everything in that photo is a decision with a reason. The wing sits high with generous [dihedral](https://en.wikipedia.org/wiki/Dihedral_(aeronautics)) because a trainer should roll itself level when the pilot panics. The landing gear is bent steel wire because grass fields eat anything stiffer. The airframe is thermocol because a crash should cost twenty rupees of material, and [wing loading](https://en.wikipedia.org/wiki/Wing_loading) that low means the plane settles onto the grass at walking speed.

The balance point ruled everything. The [center of gravity](https://en.wikipedia.org/wiki/Center_of_gravity_of_an_aircraft) has to sit ahead of the wing's neutral point for the aircraft to have positive [static stability](https://en.wikipedia.org/wiki/Longitudinal_static_stability), and the whole build ends with sliding a battery back and forth until the model balances on two fingertips at a quarter of the wing chord. The club wisdom compresses the entire theory into one line: a nose-heavy plane flies poorly, a tail-heavy plane flies once.

## Flight testing

<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/flying-years/field-test.mp4"></video>

That is an October evening on the campus grounds, one of ours in the air, and one of us running underneath it. Every maiden flight follows the same liturgy: range check, control throws, one deep breath, and a hand launch into whatever headwind exists. The first ten seconds tell you everything, because a mis-trimmed plane announces itself immediately and a [stalled](https://en.wikipedia.org/wiki/Stall_(fluid_dynamics)) wing gives you about a second of warning at these speeds.

Crashing was the tuition. Each wreck traced back to something specific: a battery that shifted aft in flight, control throws too aggressive for a windy day, a launch below flying speed. The repair bench doubled as the review meeting, and hot glue heals foam faster than any of us healed our pride.

## The jets that stayed folders

The downloaded-plans folder holds a JA37 Viggen, an X-31, an F-117, and an F-22, all designed around [electric ducted fans](https://en.wikipedia.org/wiki/Ducted_fan). EDFs are seductive and merciless: the thrust comes from a small fan spinning very fast, the amp draw would embarrass a toaster, and a hand launch has to reach flying speed on the first try because a ducted fan at half throttle is a hair dryer. Our budgets kept losing that argument, and the jets stayed aspirational.

Some designs stayed digital for better reasons:

![CAD render of an SR-71-inspired airframe that never left the computer](/images/posts/flying-years/sr71-cad.jpg)

That SR-71 body taught a full course in surface modeling and zero flights. There is a version of this hobby that lives entirely in CAD, and it is cheaper and better rested.

## The autopilot arrives

The multirotor chapter changed the flavor of everything. We assembled a hexacopter around an APM flight controller running [ArduPilot](https://ardupilot.org/), configured it in Mission Planner, and met the flight-mode menu: stabilize, loiter, position, land. The first time loiter mode parked the hexacopter against a breeze, holding position better than any thumb on the field could, the fixed-wing purists went quiet.

I keep returning to that moment. Three years of training reflexes, and a control loop with an IMU does it better while the pilot eats a sandwich. There is something in that worth thinking about properly.

## Reading above our weight

The other half of the archive is theory downloaded in ambition: the [XFLR5](http://www.xflr5.tech/) documentation on [vortex lattice](https://en.wikipedia.org/wiki/Vortex_lattice_method) analysis, stability derivatives and polars, a University of Minnesota UAV lab paper on extracting aerodynamic models, and a stack of composites guides (resin infusion, carbon layup) collected for a solar UAV that never left its folder. We understood maybe half of it at the time. The half we did understand kept showing up on the field, which is the strongest argument for reading past your level that I know.

## What the era taught

The sky invoices every mistake immediately, in full, with no appeal. Building fast beats designing long at this scale, because ten cheap airframes teach more than one perfect one. And the balance point matters more than the airfoil, the motor, and the paint combined.

The receivers, the ESCs, the LiPo chargers, and the printed-propeller habit all moved straight into [the submarine](https://siddharthksah.github.io/posts/2017/02/vandubbi/). Different fluid, same lessons, and water at least has the decency to be soft.
