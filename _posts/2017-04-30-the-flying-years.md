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

For the past three years, most of my pocket money has turned into foam. Foam board, [thermocol](https://en.wikipedia.org/wiki/Polystyrene) sheet, [hot glue](https://en.wikipedia.org/wiki/Hot-melt_adhesive), cheap servos, and propellers bought in fives because they break in fours. This is a look back at the aircraft years, written while the last of the fleet still hangs from the hostel ceiling.

The curriculum was downloaded. Flite Test's free plans and build videos were the textbook: print the tiled PDF, glue it to foam board, cut, fold, add a cheap [brushless motor](https://en.wikipedia.org/wiki/Brushless_DC_electric_motor), and an airplane exists. The archive on my drive reads like a syllabus in the order we consumed it: the FT Cruiser, the Nutball (a flying disc that should embarrass aerodynamics and refuses to), the Mini Guinea we built for carrying payloads, and a folder of fighter-jet plans whose ambitions outran our amp budget.

![The workshop floor: a long-winged white plane, a yellow trainer, a blue balsa build, boxes of propellers and LiPo packs, and a hovercraft where furniture should be](/images/posts/flying-years/workshop-fleet.jpg)

By the third year the workshop had stopped pretending to be a room. Wings leaned against every wall, a hovercraft sat where furniture should be, and one trainer flew with a soft-drink bottle for a fuselage because the bottle was the right diameter and free.

## Scratch building

Copying taught the basics, and then the basics demanded original sins. This one is ours from nose to tail:

![A scratch-built foam plane in the workshop: thermocol fuselage, taped wing, landing gear bent from steel wire with salvaged wheels](/images/posts/flying-years/foam-plane.jpg){: loading="lazy" decoding="async"}

Everything in that photo is a decision with a reason. The wing sits high with generous [dihedral](https://en.wikipedia.org/wiki/Dihedral_(aeronautics)) because a trainer should roll itself level when the pilot panics. The landing gear is bent steel wire because grass fields eat anything stiffer. The airframe is thermocol because a crash should cost twenty rupees of material, and [wing loading](https://en.wikipedia.org/wiki/Wing_loading) that low means the plane settles onto the grass at walking speed.

Once the copies flew, the designs drifted. This one has no tail at all:

![A yellow swept flying wing held up for inspection, with tip fins and the motor in a mid-wing cutout](/images/posts/flying-years/flying-wing.jpg){: loading="lazy" decoding="async"}

A [flying wing](https://en.wikipedia.org/wiki/Flying_wing) steers with two [elevons](https://en.wikipedia.org/wiki/Elevon) that mix pitch and roll into the same pair of surfaces, sweep stands in for the missing stabilizer, and the motor rides in a cutout in the wing itself. The whole aircraft is a few sheets of foam board and one argument about where the balance point goes.

The most ambitious scratch design got a name. Blue Thunder is a [biplane](https://en.wikipedia.org/wiki/Biplane) built for steady, slow aerial filming: two stacked wings buy low wing loading at a short span, the price is the wings interfering with each other's airflow, and [positive stagger](https://en.wikipedia.org/wiki/Stagger_(aeronautics)), the upper wing shifted forward, softens the interference. The upper airfoil is symmetric while the lower one is a flat-bottomed lifting section pressed into [camber](https://en.wikipedia.org/wiki/Camber_(aerodynamics)) over a wooden spar. Two carbon rods stiffen the top wing, bicycle spokes serve as pushrods, and an 1100 KV motor swings a 10x4.7 propeller on a 3S pack. All up it weighs about a kilogram, carries a bay for a camera or a second battery, and its [polars](https://en.wikipedia.org/wiki/Polar_curve_(aerodynamics)) ran through XFLR5 before any foam got cut.

Foam sheet is one school of construction. The other is built-up: a truss fuselage of balsa sticks, plywood formers at the load points, a [spar](https://en.wikipedia.org/wiki/Spar_(aeronautics)) carrying the wing's bending loads, and a skin doing almost nothing but keeping the air organized. We tried that school too:

![A built-up airframe leaning against the workshop wall: white molded wing, balsa truss fuselage, motor on a wooden pylon](/images/posts/flying-years/balsa-fuselage.jpg){: loading="lazy" decoding="async"}

A truss is a lecture in load paths. Every stick either carries something or gets cut, the structure weighs a fraction of a solid fuselage, and one bad landing turns a week of careful gluing back into sticks. Foam board forgives; balsa educates.

The balance point ruled everything. The [center of gravity](https://en.wikipedia.org/wiki/Center_of_gravity_of_an_aircraft) has to sit ahead of the wing's neutral point for the aircraft to have positive [static stability](https://en.wikipedia.org/wiki/Longitudinal_static_stability), and the whole build ends with sliding a battery back and forth until the model balances on two fingertips at a quarter of the wing chord. The club wisdom compresses the entire theory into one line: a nose-heavy plane flies poorly, a tail-heavy plane flies once.

## Flight testing

<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/flying-years/field-test.mp4"></video>

That is an October evening on the campus grounds, one of ours in the air, and one of us running underneath it. Every maiden flight follows the same liturgy: range check, control throws, one deep breath, and a hand launch into whatever headwind exists. The first ten seconds tell you everything, because a mis-trimmed plane announces itself immediately and a [stalled](https://en.wikipedia.org/wiki/Stall_(fluid_dynamics)) wing gives you about a second of warning at these speeds.

Crashing was the tuition. Each wreck traced back to something specific: a battery that shifted aft in flight, control throws too aggressive for a windy day, a launch below flying speed. The repair bench doubled as the review meeting, and hot glue heals foam faster than any of us healed our pride.

## The multirotor school

A multirotor is a different religion. A plane wants to fly and the pilot mostly negotiates; a multirotor is four motors arguing while a control board referees hundreds of times a second. On a [quadcopter](https://en.wikipedia.org/wiki/Quadcopter) adjacent propellers spin in opposite directions so their reaction torques cancel, and yaw comes from speeding up one diagonal pair. A tricopter cancels nothing, so its tail motor rides on a [servo](https://en.wikipedia.org/wiki/Servo_(radio_control)) that tilts it, vectoring thrust to hold the nose where the pilot left it.

![Our first quadcopter: wooden crossmember arms, yellow motor domes, and an Arduino wired in as the flight controller](/images/posts/flying-years/arduino-quad.jpg){: loading="lazy" decoding="async"}

That is the first one, wooden arms and an Arduino wired in as the flight controller. It looks like a school project, and every frame we built afterward borrowed parts from it.

![A Y-frame tricopter on the workshop floor mid-build: plywood center plate, power distribution wiring half soldered, the transmitter waiting in the corner](/images/posts/flying-years/tricopter-build.jpg){: loading="lazy" decoding="async"}

The build ritual barely changed between frames. Calibrate every [ESC](https://en.wikipedia.org/wiki/Electronic_speed_control) so all the motors agree on what full throttle means. Balance every propeller, because the flight controller's [IMU](https://en.wikipedia.org/wiki/Inertial_measurement_unit) reads vibration as motion and responds to blur with panic. Mount the board on foam tape for the same reason. Then spend an evening on [PID](https://en.wikipedia.org/wiki/PID_controller) gains: raise the proportional term until the frame oscillates, back it off, add derivative until the twitch smooths out, and resist the integral term until a breeze proves you need it.

The shrouds were our one deliberate aerodynamics experiment:

![A tricopter with hand-cut thermocol duct rings around all three rotors, hexagonal wooden center plate](/images/posts/flying-years/ducted-tricopter.jpg){: loading="lazy" decoding="async"}

A duct around a rotor promises free lift. It suppresses the [tip vortex](https://en.wikipedia.org/wiki/Wingtip_vortices) losses at the blade ends and, shaped well, pulls extra air through the disc for the same watts. The fine print is weight, drag in forward flight, and a shape that must be accurate to work at all. We cut our rings from thermocol to find out where the promise ends, which is the cheapest way anyone has ever audited a research paper.

Then a September evening pays for all of it:

![The quadcopter hovering low over a campus lawn, propellers blurred, football practice continuing in the background](/images/posts/flying-years/quad-hover.jpg){: loading="lazy" decoding="async"}

Props blurred, skids a hand's width off the grass, football practice ignoring us in the background. A hover that steady is the PID loop's diploma.

## The autopilot arrives

The APM flight controller changed the flavor of everything. We assembled a Y6, a coaxial hexacopter with its six motors stacked in pairs on three arms, around one running [ArduPilot](https://ardupilot.org/), configured it in [Mission Planner](https://ardupilot.org/planner/), and met the flight-mode menu: stabilize, loiter, position, land. The [FPV](https://en.wikipedia.org/wiki/First-person_view_(radio_control)) rig on it later grew a head tracker, so turning your head panned the camera, which is the closest thing to sitting inside the aircraft that this hobby sells. The first time loiter mode parked the hexacopter against a breeze, holding position better than any thumb on the field could, the fixed-wing purists went quiet.

I keep returning to that moment. Three years of training reflexes, and a control loop with an IMU does it better while the pilot eats a sandwich. There is something in that worth thinking about properly.

![The quadcopter silhouetted against a monsoon sky above the hostel rooftops](/images/posts/flying-years/quad-sky.jpg){: loading="lazy" decoding="async"}

Black against a monsoon sky, it looks less like a toy and more like where the next decade of this hobby is headed.

## The jet phase

The downloaded-plans folder holds a JA37 Viggen, an X-31, an F-117, and an F-22, all drawn around [electric ducted fans](https://en.wikipedia.org/wiki/Ducted_fan). EDF physics is merciless. The fan is small, so [disc loading](https://en.wikipedia.org/wiki/Disk_loading) runs high and thrust comes from flinging a thin stream of air backward very fast, which costs amps at a rate that would embarrass a toaster. The [LiPo](https://en.wikipedia.org/wiki/Lithium_polymer_battery) packs with discharge ratings that survive it cost more than an entire foam trainer, and a hand launch has to reach flying speed on the first try, because a ducted fan at half throttle is a hair dryer. We built our jets anyway:

![A grey-painted foam fighter jet on the workshop floor, February 2016, with a half-built hovercraft in the background](/images/posts/flying-years/edf-jet.jpg){: loading="lazy" decoding="async"}

Grey paint hides a remarkable amount of hot glue. That one shared the floor with the hovercraft, which belongs to another story.

The EDF ran its own syllabus. Datasheet thrust proved optimistic by half, and the 1.5x power margin we budgeted evaporated where the forums prescribed 2x. On an airframe this small the glue is a component, so hot glue lost its job to lighter adhesives, and the inlet lip mattered more than any of us expected, because a ducted fan breathes through a hole whose smoothness decides how much of the theory survives. The honest logbook entry is that the first EDF plane never flew. We kept the consolation: a crashing EDF jet hits the world with foam, and every propeller we owned is a spinning knife.

Some designs stayed digital for better reasons:

![CAD render of an SR-71-inspired airframe that never left the computer](/images/posts/flying-years/sr71-cad.jpg){: loading="lazy" decoding="async"}

That SR-71 body taught a full course in surface modeling and zero flights. There is a version of this hobby that lives entirely in CAD, and it is cheaper and better rested.

## One rocket

We also built a small [model rocket](https://en.wikipedia.org/wiki/Model_rocket) and named it Cohero. A rocket is the stability argument stripped to its bones: it flies straight only while the [center of pressure](https://en.wikipedia.org/wiki/Center_of_pressure_(fluid_mechanics)) sits behind the center of gravity, which is the same relationship an airplane hides inside its wing position. The nose cone came off the 3D printer, the fins came from the same foam stock as everything else, and the swing test, the whole vehicle spun on a string to check that it weathervanes into the airflow, stood in for a wind tunnel we did not have. The dimensions ran through [OpenRocket](https://openrocket.info/) and RockSim first, because fin and nose numbers are sensitive enough that one simulator's opinion is gossip.

The motor is the part that demands respect. Amateur rocketry's standard propellant is [rocket candy](https://en.wikipedia.org/wiki/Rocket_candy), [potassium nitrate](https://en.wikipedia.org/wiki/Potassium_nitrate) and sugar, and nobody sensible stands next to one at ignition, so the igniter fires electrically from a distance through a circuit we built ourselves, off a launch pad with an adjustable angle.

![The model rocket held up for inspection: shaped nose cone, pipe body, yellow foam fins](/images/posts/flying-years/rocket.jpg){: loading="lazy" decoding="async"}

A rocket compresses a semester of stability reading into two pencil marks on a pipe.

## Reading above our weight

The other half of the archive is theory downloaded in ambition: the [XFLR5](http://www.xflr5.tech/) documentation on [vortex lattice](https://en.wikipedia.org/wiki/Vortex_lattice_method) analysis, stability derivatives and polars, a University of Minnesota UAV lab paper on extracting aerodynamic models, and a stack of composites guides (resin infusion, carbon layup) collected for a solar UAV that never left its folder. We understood maybe half of it at the time. The half we did understand kept showing up on the field, which is the strongest argument for reading past your level that I know.

## What the era taught

1. The balance point outranks the airfoil, the motor, and the paint combined. A quarter-chord balance on two fingertips is the entire preflight that matters.
2. Datasheet thrust is marketing. Halve it for a duct, keep the 2x power margin the forums prescribe, and weigh the glue, because on a small airframe the adhesive is a component.
3. Balance the propeller before tuning the controller. The IMU reads vibration as motion, and no PID gain can tell a bent prop from a gust.
4. Ten cheap airframes out-teach one perfect one, and the sky invoices every mistake immediately, in full, with no appeal.
5. Crashes end arguments that whiteboards cannot. Every wreck traced to one specific cause, and the repair bench doubled as the only review meeting nobody skipped.
6. An autopilot outflying your thumbs is information, and pride is a bad reason to file it away.

![The workshop wall in September 2016, finished airframes hung in rows above the benches](/images/posts/flying-years/hangar-wall.jpg){: loading="lazy" decoding="async"}

By the end, the wall hung like a museum of the whole argument: every airframe a hypothesis, most of them disproven, all of them kept.

The receivers, the ESCs, the LiPo chargers, and the printed-propeller habit all moved straight into [the submarine](https://siddharthksah.github.io/posts/2017/02/vandubbi/). Different fluid, same lessons, and water at least has the decency to be soft.

<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/flying-years/court-hover.mp4"></video>

An April afternoon this month, a hover check on the court outside the workshop. The logbook is still open.
