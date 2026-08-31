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

For the past two years I have been building a 3D bioprinter. It is called BioP, it extrudes soft materials instead of plastic, and it has taught me more engineering than every course I have taken combined. This post is a plain account of what building it actually involves.

A normal desktop 3D printer is a solved problem. The [RepRap](https://en.wikipedia.org/wiki/RepRap) lineage worked out thermoplastics years ago: melt a filament, push it through a hot nozzle, and the plastic freezes obediently where you put it. The control loop is forgiving because the material cooperates.

[Bioprinting](https://en.wikipedia.org/wiki/3D_bioprinting) replaces that obedient plastic with [hydrogels](https://en.wikipedia.org/wiki/Hydrogel), soft water-based materials that living cells can survive in. We work with [alginate](https://en.wikipedia.org/wiki/Sodium_alginate) crosslinked with calcium, with Laponite, and with [GelMA](https://en.wikipedia.org/wiki/Gelatin_methacryloyl), a gelatin derivative that cures under UV. Each one is a compromise between printing well and being worth printing: the gels that hold shape best are the ones cells like least.

## The parameter grid

Hydrogels flow when pushed and stiffen when still, a property called [shear thinning](https://en.wikipedia.org/wiki/Shear_thinning), and the balance between pressure, temperature, and speed decides whether a print holds its shape or collapses into a puddle. The window where everything works is narrow, it moves with room temperature, and it is different for every ink batch.

So the daily work is sweeping parameters and writing the results on the plate in marker, because the plate is the lab notebook that cannot get lost.

![Two test extrusions of GelMA ink at 20 and 40 psi, printed at 100 mm/s and 22.5 °C, with the parameters written on the plate in marker](/images/posts/hardware-is-hard/parameter-sweep.jpg)

The [G-code](https://en.wikipedia.org/wiki/G-code) for both of these prints describes a clean cross. The material printed what it wanted to print. At 20 psi the ink under-extrudes and the arms thin out; at 40 psi it swells past the toolpath and pools at the junction. Same geometry file, same nozzle, same day. The file describes an intention, and the ink negotiates.

## The electronics are held together by learning

BioP's motion and extrusion run on [Arduino](https://www.arduino.cc/)-based control with custom syringe extruders, because a syringe of hydrogel needs gentle, precise pressure that a filament drive was never designed to deliver. We wrote the firmware, built a G-code conversion path for our toolhead, and rebuilt both more times than I want to count.

![The control electronics of the printer mid-development: toggle switches, an indicator LED board, and more jumper wires than any diagram admits to](/images/posts/hardware-is-hard/electronics.jpg)

This photo is what prototyping hardware really looks like. The schematic in the design file is clean. The bench is not, because every wire in that tangle exists to answer a question the schematic did not know to ask: whether the endstop chatters, or the pressure line leaks at exactly the moment a print gets interesting.

![An earlier state of the same electronics, photographed in a moment of honesty](/images/posts/hardware-is-hard/wiring-bw.jpg)

Software forgives. You recompile and the old mistake is gone. Hardware keeps a ledger: the stripped thread stays stripped, the burnt driver stays burnt, and the print that failed at 2 a.m. consumed real ink from a batch that took a day to prepare. Debugging with a multimeter costs hours where debugging with a print statement costs seconds.

## What two years of this teaches

The lessons are unglamorous, which I now suspect is the mark of the real ones.

1. Change one variable per print. The grid grows fast, and a plate with two changed variables teaches nothing.
2. Write the parameters on the plate, photograph the plate, and file the photo. Memory is not an instrument.
3. The material always wins. A control loop can nudge physics; it cannot outvote it.
4. Cheap components are expensive. Every rupee saved on a stepper driver was paid back in debugging evenings, with interest.
5. When a print fails, believe the print. The G-code, the firmware, and the datasheet are all testimony; the failed print is evidence.

A semester of thesis work in a bioprinting lab in Boston taught me the formal versions of these rules. Building our own machine from scratch taught me why they exist.

## What is next

The part of this work I keep drifting toward lately sits on the imaging side. Checking a printed structure means imaging it, and reconstructing useful 3D information from those images is its own problem. I have started working with machine learning for image reconstruction there, and it is the most interesting thing I have touched in months: the models find structure in data I would have called noise.

The printer is not finished. Printers like this are never finished. But it prints, the ink listens more often than it used to, and the plate photos are slowly filling a hard drive, which is what progress looks like in hardware.
