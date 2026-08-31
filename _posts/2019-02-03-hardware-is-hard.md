---
title: "Hardware is hard"
date: 2019-02-03
categories:
  - engineering
permalink: /posts/2019/02/hardware-is-hard/
tags:
  - Bioprinting
  - Hardware
  - 3D Printing
---

For the past year, a team of ten of us has been building a 3D bioprinter during our undergrad. I lead the project. It is called BioP, it extrudes soft materials instead of plastic, and it has taught me more engineering than every course I have taken combined. This post is a plain account of what building it actually involves.

![The BioP printer: laser-cut enclosure with UV and hot-surface warnings, syringe extruder over the heated bed, and the control software running in front](/images/posts/hardware-is-hard/machine.jpg)

That is the machine. The motion platform began life as a Prusa i3-class frame from the [RepRap](https://en.wikipedia.org/wiki/RepRap) family, and almost everything above the bed has since been replaced: the extruder, the enclosure, the electronics, and all of the software. The warning decals on the front are earned, since the enclosure carries a UV curing lamp and a heated bed, and both bite.

The project has picked up awards along the way, including selection as a [World Summit Award](https://wsa-global.org/) Young Innovators winner. Mostly, though, the year has looked like the unglamorous work below.

A normal desktop 3D printer is a solved problem. The RepRap lineage worked out thermoplastics years ago: melt a filament, push it through a hot nozzle, and the plastic freezes obediently where you put it. The control loop is forgiving because the material cooperates.

[Bioprinting](https://en.wikipedia.org/wiki/3D_bioprinting) replaces that obedient plastic with [hydrogels](https://en.wikipedia.org/wiki/Hydrogel), soft water-based materials that living cells can survive in. We work with [alginate](https://en.wikipedia.org/wiki/Sodium_alginate) crosslinked with calcium, with Laponite, and with [GelMA](https://en.wikipedia.org/wiki/Gelatin_methacryloyl), a gelatin derivative that cures under UV. Each one is a compromise between printing well and being worth printing: the gels that hold shape best are the ones cells like least.

## The parameter grid

Hydrogels flow when pushed and stiffen when still, a property called [shear thinning](https://en.wikipedia.org/wiki/Shear_thinning), and the balance between pressure, temperature, and speed decides whether a print holds its shape or collapses into a puddle. The window where everything works is narrow, it moves with room temperature, and it is different for every ink batch.

So the daily work is sweeping parameters and writing the results on the plate in marker, because the plate is the lab notebook that cannot get lost.

![Two test extrusions of GelMA ink at 20 and 40 psi, printed at 100 mm/s and 22.5 °C, with the parameters written on the plate in marker](/images/posts/hardware-is-hard/parameter-sweep.jpg)

The [G-code](https://en.wikipedia.org/wiki/G-code) for both of these prints describes a clean cross. The material printed what it wanted to print. At 20 psi the ink under-extrudes and the arms thin out; at 40 psi it swells past the toolpath and pools at the junction. Same geometry file, same nozzle, same day. The file describes an intention, and the ink negotiates.

## Curing is a race against slump

Shape only survives if it sets. Alginate [crosslinks](https://en.wikipedia.org/wiki/Cross-link) ionically, wherever calcium ions reach it, so those prints get misted and stiffen from the outside in. GelMA cures under the UV lamp through a [photoinitiator](https://en.wikipedia.org/wiki/Photoinitiator) mixed into the ink, Irgacure 2959 in our case, and the dose is its own parameter: under-cured layers slump before the next pass lands, over-cured ones go brittle, and the lamp's heat dries the print while it works. Every layer is a small race between deposition and collapse, and the UV timing decides who wins.

## The extruder is the machine

A syringe of hydrogel needs gentle, precise displacement that a filament drive was never designed to deliver, so we designed our own: a stepper-driven carriage that presses a standard syringe through a leadscrew, printed in parts on the same class of machine it now improves.

![The custom syringe extruder: a NEMA stepper driving a leadscrew through a 3D-printed carriage that presses a standard syringe](/images/posts/hardware-is-hard/extruder.jpg)

This is where most of the design iterations went. Too much backlash and the ink keeps flowing after the move ends; too much friction and the stepper skips exactly one step, which you discover three layers later.

The nozzle is a blunt dispensing needle on a [Luer](https://en.wikipedia.org/wiki/Luer_taper) fitting, sized in [gauge](https://en.wikipedia.org/wiki/Birmingham_gauge) like anything in a clinic, and the arithmetic is volumetric: bore area times plunger travel equals ink on the plate, so the same millimeter of plunger is a fat line from a wide syringe and a hairline from a narrow one. The syringe also behaves like a spring. Seals flex and the gel compresses, so flow lags the command at both ends of a move, and the plunger retracts a hair before travel moves to fight the ooze. Stopping extrusion is a request; the material honors it late.

Here is fourteen seconds of it printing into a dish under the UV lamp:

<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/hardware-is-hard/printing-clip.mp4"></video>

## The motion platform is a blind robot

Everything moves on [stepper motors](https://en.wikipedia.org/wiki/Stepper_motor), and steppers shape the whole machine because they are open loop. No encoder reports where an axis actually is. The controller counts microsteps and trusts [dead reckoning](https://en.wikipedia.org/wiki/Dead_reckoning); demand too much acceleration and a motor skips silently, and from that moment the count is a lie. So every move runs a trapezoid, accelerate, cruise, decelerate, and every session starts by driving each axis into its [limit switch](https://en.wikipedia.org/wiki/Limit_switch), because the home switches are the only ground truth the machine owns.

The other ritual is calibration. Steps-per-millimeter on each axis gets checked with calipers against a commanded move, and the difference between [repeatability and accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) matters here: a gantry that lands 0.1 mm off the same way every time is fixable in software, and one that lands somewhere new each pass is scrap. First layers get trammed with a sheet of paper under the needle, the oldest instrument in 3D printing.

## The electronics are held together by learning

Motion and extrusion run on [Arduino](https://www.arduino.cc/)-based control. We wrote the firmware, built a G-code conversion path for our toolhead, and rebuilt both more times than I want to count.

![The control electronics mid-development: toggle switches, an indicator LED board, and more jumper wires than any diagram admits to](/images/posts/hardware-is-hard/electronics.jpg)

This photo is what prototyping hardware really looks like. The schematic in the design file is clean. The bench is not, because every wire in that tangle exists to answer a question the schematic did not know to ask: whether the endstop chatters, or the pressure line leaks at exactly the moment a print gets interesting.

Software forgives. You recompile and the old mistake is gone. Hardware keeps a ledger: the stripped thread stays stripped, the burnt driver stays burnt, and the print that failed at 2 a.m. consumed real ink from a batch that took a day to prepare. Debugging with a multimeter costs hours where debugging with a print statement costs seconds.

## The software knows about syringes

Slicers assume filament, so we wrote our own control software, BioApp. It runs the printer over serial: jog controls, layer height, speed, a print-time calculator, and a start button that we trust because we know exactly what it does.

![BioApp: live temperature, humidity, and smoke readings with sensor status, stepper controls, and a camera view of the print bed](/images/posts/hardware-is-hard/bioapp.jpg)

Half the interface is environment monitoring, because hydrogel prints care about the room. Two temperature sensors and two humidity sensors feed running means, a smoke sensor watches the electronics with an alarm wired to it, and a camera stares at the bed so a print can be babysat from across the lab.

<video controls preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/hardware-is-hard/software-teaser.mp4"></video>

## Closing the loop on layer height

Bioprinting has a control problem that plastic printing mostly ignores. The G-code assumes every layer lands exactly one layer-height tall, so the nozzle climbs by a fixed step per layer, open loop. Hydrogels break that assumption from both directions: ink swells as it leaves the nozzle, a rheology effect called [die swell](https://en.wikipedia.org/wiki/Die_swell), and then slumps as it settles before curing. The error per layer is small and it compounds, so by layer twenty the nozzle is either ploughing through the print or extruding into air.

The fix is to treat the printer as a robot rather than a player piano: perceive, estimate, correct. The bed camera does more than babysitting duty. After [camera calibration](https://en.wikipedia.org/wiki/Camera_resectioning), a checkerboard on the bed gives the [homography](https://en.wikipedia.org/wiki/Homography_(computer_vision)) that turns pixels into millimeters, and each deposited pass gets segmented with classical computer vision, [OpenCV](https://en.wikipedia.org/wiki/OpenCV) thresholding and [Canny edges](https://en.wikipedia.org/wiki/Canny_edge_detector), to estimate the real height of the material near the nozzle. A [closed-loop controller](https://en.wikipedia.org/wiki/Closed-loop_controller) then applies a small, clamped Z correction before the next layer. Dynamic Z compensation, in the plainest sense: measure what the material did, adjust what the machine does next.

The honest status is that it works on opaque inks in good light. A wet transparent gel under a UV lamp is a miserable computer-vision subject, all glare and low contrast, and the classical pipeline needs re-tuning for every new ink. That frustration is currently doing more to steer my reading list than any course.

## Straight from the CAD

The full enclosure, rendered as a turntable from the design files, back when rendering it was easier than building it:

<video autoplay loop muted playsinline preload="metadata" style="width:100%;border-radius:6px;" src="/images/posts/hardware-is-hard/cad-turntable.mp4"></video>

## What a year of this teaches

The lessons are unglamorous, which I now suspect is the mark of the real ones.

1. Change one variable per print. A plate with two changed variables teaches nothing, and the grid grows faster than the ink budget.
2. Write the parameters on the plate, photograph the plate, file the photo. Memory is not an instrument.
3. The material always wins. A control loop can nudge physics; it cannot outvote it.
4. Cheap components are expensive. Every rupee saved on a stepper driver was paid back in debugging evenings, with interest.
5. If the machine can measure something, stop assuming it. We assumed layer height for months. The camera disagreed.
6. When a print fails, believe the print. The G-code, the firmware, and the datasheet are all testimony; the failed print is evidence.
7. A ten-person build runs on writing things down. A design that lives in one head is a single point of failure, and exam season is coming for that head.

A semester of thesis work in a bioprinting lab in Boston taught me the formal versions of these rules. Building our own machine from scratch taught me why they exist.

## What is next

The part of this work I keep drifting toward lately sits on the imaging side. Checking a printed structure means imaging it, and reconstructing useful 3D information from those images is its own problem. I have started working with machine learning for image reconstruction there, and it is the most interesting thing I have touched in months: the models find structure in data I would have called noise.

The printer is not finished. Printers like this are never finished. But it prints, the ink listens more often than it used to, and the plate photos are slowly filling a hard drive, which is what progress looks like in hardware.
