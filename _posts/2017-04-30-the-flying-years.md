---
title: "Three years of RC planes and multirotors"
date: 2017-04-30
categories:
  - engineering
permalink: /posts/2017/04/the-flying-years/
tags:
  - RC Aircraft
  - Drones
  - Hardware
---

For the past three years most of my pocket money has gone into foam. Foam board, [thermocol](https://en.wikipedia.org/wiki/Polystyrene) sheet, [hot glue](https://en.wikipedia.org/wiki/Hot-melt_adhesive), cheap servos, and propellers bought in fives because they break in fours. This is a look back at the aircraft years, written while the last of the fleet still hangs from the hostel ceiling.

We learned from Flite Test. Their free plans and build videos were the textbook: print the tiled PDF, glue it to foam board, cut, fold, add a cheap [brushless motor](https://en.wikipedia.org/wiki/Brushless_DC_electric_motor), and you have an airplane. The archive on my drive is in the order we built them: the FT Cruiser, the Nutball (a flying disc that has no business flying and does anyway), the Mini Guinea we built for carrying payloads, and a folder of fighter-jet plans we couldn't afford the batteries for.

![The workshop floor: a long-winged white plane, a yellow trainer, a blue balsa build, boxes of propellers and LiPo packs, and a hovercraft where furniture should be](/images/posts/flying-years/workshop-fleet.jpg){: srcset="/images/posts/flying-years/workshop-fleet-480.jpg 480w, /images/posts/flying-years/workshop-fleet-960.jpg 960w, /images/posts/flying-years/workshop-fleet.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px"}

By the third year the workshop was full. Wings leaned against every wall, a hovercraft sat where furniture should be, and one trainer flew with a soft-drink bottle for a fuselage because the bottle was the right diameter and free.

## Scratch building

After a few copies we started designing our own. This one is ours from nose to tail:

![A scratch-built foam plane in the workshop: thermocol fuselage, taped wing, landing gear bent from steel wire with salvaged wheels](/images/posts/flying-years/foam-plane.jpg){: srcset="/images/posts/flying-years/foam-plane-480.jpg 480w, /images/posts/flying-years/foam-plane-960.jpg 960w, /images/posts/flying-years/foam-plane.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

Each part in that photo has a reason. The wing sits high with generous [dihedral](https://en.wikipedia.org/wiki/Dihedral_(aeronautics)) because a trainer should roll itself level when the pilot panics. The landing gear is bent steel wire because grass fields eat anything stiffer. The airframe is thermocol because a crash should cost twenty rupees of material, and [wing loading](https://en.wikipedia.org/wiki/Wing_loading) that low means the plane settles onto the grass at walking speed.

Once the copies flew, the designs drifted. This one has no tail at all:

![A yellow swept flying wing held up for inspection, with tip fins and the motor in a mid-wing cutout](/images/posts/flying-years/flying-wing.jpg){: srcset="/images/posts/flying-years/flying-wing-480.jpg 480w, /images/posts/flying-years/flying-wing-960.jpg 960w, /images/posts/flying-years/flying-wing.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

A [flying wing](https://en.wikipedia.org/wiki/Flying_wing) steers with two [elevons](https://en.wikipedia.org/wiki/Elevon) that mix pitch and roll into the same pair of surfaces. Sweep stands in for the missing stabilizer, and the motor rides in a cutout in the wing itself. The whole aircraft is a few sheets of foam board and one argument about where the balance point goes.

The most ambitious scratch design got a name. Blue Thunder is a [biplane](https://en.wikipedia.org/wiki/Biplane) built for steady, slow aerial filming. Two stacked wings give low wing loading at a short span. The cost is that the wings interfere with each other's airflow, and [positive stagger](https://en.wikipedia.org/wiki/Stagger_(aeronautics)), the upper wing shifted forward, reduces that. The upper airfoil is symmetric and the lower one is a flat-bottomed lifting section pressed into [camber](https://en.wikipedia.org/wiki/Camber_(aerodynamics)) over a wooden spar. Two carbon rods stiffen the top wing, bicycle spokes serve as pushrods, and an 1100 KV motor swings a 10x4.7 propeller on a 3S pack. All up it weighs about a kilogram, carries a bay for a camera or a second battery, and we ran its [polars](https://en.wikipedia.org/wiki/Polar_curve_(aerodynamics)) through XFLR5 before cutting any foam.

Foam sheet is one way to build. The other is built-up construction: a truss fuselage of balsa sticks, plywood formers at the load points, a [spar](https://en.wikipedia.org/wiki/Spar_(aeronautics)) carrying the wing's bending loads, and a skin that does little except keep the airflow attached. We tried that too:

![A built-up airframe leaning against the workshop wall: white molded wing, balsa truss fuselage, motor on a wooden pylon](/images/posts/flying-years/balsa-fuselage.jpg){: srcset="/images/posts/flying-years/balsa-fuselage-480.jpg 480w, /images/posts/flying-years/balsa-fuselage-960.jpg 960w, /images/posts/flying-years/balsa-fuselage.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

A truss teaches load paths. Every stick either carries something or gets cut, the structure weighs a fraction of a solid fuselage, and one bad landing turns a week of careful gluing back into sticks. Foam board is much easier to repair.

The balance point decided everything. The [center of gravity](https://en.wikipedia.org/wiki/Center_of_gravity_of_an_aircraft) has to sit ahead of the wing's neutral point for the aircraft to have positive [static stability](https://en.wikipedia.org/wiki/Longitudinal_static_stability), and every build ends with sliding a battery back and forth until the model balances on two fingertips at a quarter of the wing chord. The club saying is: a nose-heavy plane flies poorly, a tail-heavy plane flies once.

## Flight testing

<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/flying-years/field-test.mp4"></video>

This is an October evening on the campus grounds, one of ours in the air and one of us running underneath it. Every maiden flight follows the same steps: range check, control throws, one deep breath, and a hand launch into whatever headwind exists. The first ten seconds tell you everything. A mis-trimmed plane announces itself immediately, and a [stalled](https://en.wikipedia.org/wiki/Stall_(fluid_dynamics)) wing gives you about a second of warning at these speeds.

We crashed a lot. Each wreck traced back to something specific: a battery that shifted aft in flight, control throws too aggressive for a windy day, a launch below flying speed. The repair bench doubled as the review meeting.

## Multirotors

A multirotor is a different problem. A plane wants to fly and the pilot mostly negotiates. A multirotor is four motors that would fall out of the sky without a control board correcting them hundreds of times a second. On a [quadcopter](https://en.wikipedia.org/wiki/Quadcopter) adjacent propellers spin in opposite directions so their reaction torques cancel, and yaw comes from speeding up one diagonal pair. A tricopter cancels nothing, so its tail motor rides on a [servo](https://en.wikipedia.org/wiki/Servo_(radio_control)) that tilts it, vectoring thrust to hold the nose where the pilot left it.

![Our first quadcopter: wooden crossmember arms, yellow motor domes, and an Arduino wired in as the flight controller](/images/posts/flying-years/arduino-quad.jpg){: srcset="/images/posts/flying-years/arduino-quad-480.jpg 480w, /images/posts/flying-years/arduino-quad-960.jpg 960w, /images/posts/flying-years/arduino-quad.jpg 1069w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

This is the first one, wooden arms and an Arduino wired in as the flight controller. It looks like a school project, and every frame we built afterward borrowed parts from it.

![A Y-frame tricopter on the workshop floor mid-build: plywood center plate, power distribution wiring half soldered, the transmitter waiting in the corner](/images/posts/flying-years/tricopter-build.jpg){: srcset="/images/posts/flying-years/tricopter-build-480.jpg 480w, /images/posts/flying-years/tricopter-build-960.jpg 960w, /images/posts/flying-years/tricopter-build.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

The build routine barely changed between frames. Calibrate every [ESC](https://en.wikipedia.org/wiki/Electronic_speed_control) so all the motors agree on what full throttle means. Balance every propeller, because the flight controller's [IMU](https://en.wikipedia.org/wiki/Inertial_measurement_unit) reads vibration as motion. Mount the board on foam tape for the same reason. Then spend an evening on [PID](https://en.wikipedia.org/wiki/PID_controller) gains: raise the proportional term until the frame oscillates, back it off, add derivative until the twitch smooths out, and hold off on the integral term until a breeze proves you need it.

The shrouds were our one deliberate aerodynamics experiment:

![A tricopter with hand-cut thermocol duct rings around all three rotors, hexagonal wooden center plate](/images/posts/flying-years/ducted-tricopter.jpg){: srcset="/images/posts/flying-years/ducted-tricopter-480.jpg 480w, /images/posts/flying-years/ducted-tricopter-960.jpg 960w, /images/posts/flying-years/ducted-tricopter.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

A duct around a rotor promises free lift. It suppresses the [tip vortex](https://en.wikipedia.org/wiki/Wingtip_vortices) losses at the blade ends and, shaped well, pulls extra air through the disc for the same watts. The fine print is weight, drag in forward flight, and a shape that has to be accurate to work at all. We cut our rings from thermocol to find out where the promise ends.

This is from a September evening:

![The quadcopter hovering low over a campus lawn, propellers blurred, football practice continuing in the background](/images/posts/flying-years/quad-hover.jpg){: srcset="/images/posts/flying-years/quad-hover-480.jpg 480w, /images/posts/flying-years/quad-hover-960.jpg 960w, /images/posts/flying-years/quad-hover.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

Props blurred, skids a hand's width off the grass, football practice going on behind. A hover that steady means the PID gains are right.

## The APM autopilot

The APM flight controller changed everything. We assembled a Y6, a coaxial hexacopter with its six motors stacked in pairs on three arms, around one running [ArduPilot](https://ardupilot.org/), configured it in [Mission Planner](https://ardupilot.org/planner/), and got a flight-mode menu: stabilize, loiter, position, land. The [FPV](https://en.wikipedia.org/wiki/First-person_view_(radio_control)) rig on it later grew a head tracker, so turning your head panned the camera. That's the closest thing to sitting inside the aircraft this hobby offers. The first time loiter mode parked the hexacopter against a breeze, holding position better than any of us could by hand, the fixed-wing people in the club stopped arguing.

I keep thinking about that. Three years of training reflexes, and a control loop with an IMU does it better while the pilot eats a sandwich.

![The quadcopter silhouetted against a monsoon sky above the hostel rooftops](/images/posts/flying-years/quad-sky.jpg){: srcset="/images/posts/flying-years/quad-sky-480.jpg 480w, /images/posts/flying-years/quad-sky-960.jpg 960w, /images/posts/flying-years/quad-sky.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

Black against a monsoon sky above the hostel rooftops.

## EDF jets

The downloaded-plans folder holds a JA37 Viggen, an X-31, an F-117, and an F-22, all drawn around [electric ducted fans](https://en.wikipedia.org/wiki/Ducted_fan). EDFs are hard. The fan is small, so [disc loading](https://en.wikipedia.org/wiki/Disk_loading) is high and thrust comes from throwing a thin stream of air backward very fast, which draws a lot of current. The [LiPo](https://en.wikipedia.org/wiki/Lithium_polymer_battery) packs with discharge ratings that survive it cost more than an entire foam trainer, and a hand launch has to reach flying speed on the first try, because a ducted fan at half throttle is a hair dryer. We built our jets anyway:

![A grey-painted foam fighter jet on the workshop floor, February 2016, with a half-built hovercraft in the background](/images/posts/flying-years/edf-jet.jpg){: srcset="/images/posts/flying-years/edf-jet-480.jpg 480w, /images/posts/flying-years/edf-jet-960.jpg 960w, /images/posts/flying-years/edf-jet.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

Grey paint hides a lot of hot glue. That one shared the floor with the hovercraft, which belongs to another story.

We learned a few things specific to EDFs. Datasheet thrust was optimistic by about half, and the 1.5x power margin we budgeted evaporated where the forums say 2x. On an airframe this small the glue is a component, so hot glue lost its job to lighter adhesives. The inlet lip mattered more than any of us expected, because a ducted fan breathes through a hole whose smoothness decides how much of the theory survives. The first EDF plane never flew. One consolation: a crashing EDF jet hits the world with foam, and every propeller we owned is a spinning knife.

Some designs stayed digital:

![CAD render of an SR-71-inspired airframe that never left the computer](/images/posts/flying-years/sr71-cad.jpg){: srcset="/images/posts/flying-years/sr71-cad-480.jpg 480w, /images/posts/flying-years/sr71-cad-960.jpg 960w, /images/posts/flying-years/sr71-cad.jpg 1200w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

That SR-71 body taught us surface modeling and never flew. There's a version of this hobby that lives entirely in CAD, and it's cheaper and better rested.

## The rocket

We also built a small [model rocket](https://en.wikipedia.org/wiki/Model_rocket) and named it Cohero. A rocket has the same stability requirement as a plane with nothing else in the way. It flies straight only while the [center of pressure](https://en.wikipedia.org/wiki/Center_of_pressure_(fluid_mechanics)) sits behind the center of gravity, which is the relationship an airplane hides inside its wing position. The nose cone came off the 3D printer, the fins came from the same foam stock as everything else, and the swing test (the whole vehicle spun on a string to check that it weathervanes into the airflow) stood in for a wind tunnel we didn't have. The dimensions went through [OpenRocket](https://openrocket.info/) and RockSim first, because fin and nose numbers are sensitive enough that we wanted two simulators to agree.

The motor is the part that demands respect. Amateur rocketry's standard propellant is [rocket candy](https://en.wikipedia.org/wiki/Rocket_candy), [potassium nitrate](https://en.wikipedia.org/wiki/Potassium_nitrate) and sugar, and nobody sensible stands next to one at ignition. So the igniter fires electrically from a distance through a circuit we built ourselves, off a launch pad with an adjustable angle.

![The model rocket held up for inspection: shaped nose cone, pipe body, yellow foam fins](/images/posts/flying-years/rocket.jpg){: srcset="/images/posts/flying-years/rocket-480.jpg 480w, /images/posts/flying-years/rocket.jpg 699w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

## Theory we read

The other half of the archive is theory we downloaded and half understood: the [XFLR5](http://www.xflr5.tech/) documentation on [vortex lattice](https://en.wikipedia.org/wiki/Vortex_lattice_method) analysis, stability derivatives and polars, a University of Minnesota UAV lab paper on extracting aerodynamic models, and a stack of composites guides (resin infusion, carbon layup) collected for a solar UAV that never left its folder. The half we did understand kept showing up on the field. That's the best argument I know for reading past your level.

## What we learned

1. The balance point matters more than the airfoil, the motor, and the paint combined. A quarter-chord balance on two fingertips is the only preflight check that matters.
2. Datasheet thrust is marketing. Halve it for a duct, keep the 2x power margin the forums recommend, and weigh the glue, because on a small airframe the adhesive is a component.
3. Balance the propeller before tuning the controller. The IMU reads vibration as motion, and no PID gain can tell a bent prop from a gust.
4. Ten cheap airframes taught us more than one careful one would have. Crashes are cheap when the airframe cost twenty rupees.
5. Crashes settle arguments. Every wreck traced to one specific cause, and the repair bench was the only review meeting nobody skipped.
6. An autopilot outflying your thumbs is information. Pride is a bad reason to ignore it.

![The workshop wall in September 2016, finished airframes hung in rows above the benches](/images/posts/flying-years/hangar-wall.jpg){: srcset="/images/posts/flying-years/hangar-wall-480.jpg 480w, /images/posts/flying-years/hangar-wall-960.jpg 960w, /images/posts/flying-years/hangar-wall.jpg 1600w" sizes="(max-width: 800px) 92vw, 770px" loading="lazy" decoding="async"}

By the end, the wall was full of finished airframes, most of which had crashed at least once and all of which we kept.

The receivers, the ESCs, the LiPo chargers, and the printed-propeller habit all moved straight into [the submarine](https://siddharthksah.github.io/posts/2017/02/vandubbi/). Different fluid, same lessons, and water is softer.

<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/flying-years/court-hover.mp4"></video>

An April afternoon this month, a hover check on the court outside the workshop.
