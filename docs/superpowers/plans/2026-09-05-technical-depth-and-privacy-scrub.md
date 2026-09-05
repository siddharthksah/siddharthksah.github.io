# Technical depth and privacy scrub for all posts

> **For agentic workers:** content plan, executed inline in this session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen every published post with real engineering detail (equations, numbers, code, failure mechanics) while removing anything a stranger shouldn't learn about the owner's home network or habits.

**Approach:** One task per post. Each task lists (a) what to cut for privacy, (b) the technical sections to add with the actual numbers and formulas, (c) checks. Prose stays in the plain voice from `docs/writing-guide.md`. Owner license (2026-09-05): invented technical detail is fine if it is deeply technical, fits the post, and is physically and computationally correct. Existing facts, links, media, tables, and permalinks stay.

**Tools:** `scripts/prose-tells.py` for voice regression, `bundle exec jekyll build` for rendering, a before/after count of images, videos, and external links per post.

---

## Ground rules for all tasks

- Every added number must be derivable from a stated formula or a stated measurement. Show the derivation once.
- Every added mechanism must be the standard way the thing works (Marlin, pf, ACME, ONNX Runtime, etc.), so a reader can verify it against public docs.
- No new hostnames, IPs, subnets, schedules, or lists of exposed services. When a fact was sensitive, generalize it; don't invent a fake specific.
- The SnapOtter post links to a real repo. Add mechanism-level explanation only; no new file paths or issue numbers.
- The two factory posts carry an "illustrative" disclaimer, so invented scenarios and code are fine there.
- Keep the voice: short declarative sentences, no aphorism closers, no personified hardware, no em dashes.

## Privacy scrub list (all in `_posts/2026-09-01-homelab.md`)

| Cut | Replace with |
|---|---|
| "static IP from the ISP" | a public address the DNS record points at |
| DuckDNS by name, `*.duckdns.org` pattern, the tools-table row | "a dynamic DNS provider with an ACME DNS-01 hook" |
| Three routers, one per bedroom, same SSID | drop entirely |
| "main router forwards exactly two ports" | drop the count; 80 and 443 stay because the nginx section needs them |
| Docker daemon pinned to `172.16.0.0/12` | "pinned to a range that doesn't overlap the LAN" |
| Pi-hole "plugged straight into the main router" | "on the Raspberry Pi" |
| Immich named as the public service for relatives abroad | "a few services are reachable from outside so family can use them" |
| Sunday 04:30 restart | "a weekly restart in the early hours" |
| Kiosk dashboard "reachable from anywhere" with basic auth | "behind an access list" |
| Enumerated nine public services and four LAN-only tools; stale proxy-host rows paragraph | one sentence: admin tools never get a public proxy host; drop the enumeration and the stale-rows detail |
| "the age key lives in the password manager" | "the age key is stored off the machine" |
| Backblaze exact retention cost, git vault 90 MB ceiling, 772 MB dump size | keep; not identifying |

Other posts: nothing to cut. Campus projects and open-source work are already public.

---

### Task 1: Bono (`_posts/2017-02-12-bono.md`)

- [x] Add "The kinematics" with explicit equations. Motors at (0,0) and (W,0), y down. Inverse: L1 = sqrt(x^2 + y^2), L2 = sqrt((W-x)^2 + y^2). Forward: x = (L1^2 - L2^2 + W^2) / 2W, y = sqrt(L1^2 - x^2). Segment length 2 mm; chord error on a 100 mm radius arc is 2^2/(8*100) = 0.005 mm.
- [x] Add "Steps and microsteps": 200 steps/rev, 1/16 microstepping, printed sprocket circumference 60 mm gives 3200/60 = 53.3 microsteps/mm, 18.75 um per microstep. Incremental torque of one 1/16 microstep is sin(90 deg/16) = 9.8% of holding torque, so microsteps get lost under load. A4988 current limit I = Vref / (8 * Rs), Rs = 0.1 ohm, Vref 0.8 V gives 1.0 A. L293D was capped at 600 mA with no current limiting.
- [x] Add tension and conditioning math to the workspace section. Vertical equilibrium T1 sin(phi1) + T2 sin(phi2) = mg; near the top edge phi is small and T = mg / (2 sin phi), at 10 deg that is 2.9 mg per cord and vertical stiffness collapses. Near a top corner the far cord tension falls to mg cos(phi_near), about 0.17 mg, and the counterweight is all that keeps it taut. Deep below the motors the cords go parallel and the Jacobian condition number cot-style grows; give the three numbers (top centre 5.4, middle 1.6, deep bottom 3.6).
- [x] Add pendulum numbers: f = (1/2pi) sqrt(g/L), L about 0.6 m gives 0.64 Hz; acceleration 150 mm/s^2 to 60 mm/s cruise means a 0.4 s ramp, under half a period, so the swing isn't pumped.
- [x] Add shading algorithm: pixel darkness maps to scribble amplitude; pen tip 2 mm sets minimum pixel; 400-pixel-wide images on a 900 mm board.
- [x] Verify: scan, media/link counts unchanged.

### Task 2: Vandubbi (`_posts/2017-02-26-vandubbi.md`)

- [x] Control chain: `pulseIn` waits for the pulse edge then measures, up to ~22 ms per channel with signal present, so three channels cost up to ~66 ms per loop; the default 1 s `pulseIn` timeout would stall the loop 3 s on signal loss, so the sketch passes a 25 ms timeout and treats zero as failsafe. V2: pin-change interrupts on a PPM sum stream, one ISR, sub-millisecond latency.
- [x] ESC: sensorless commutation needs back-EMF; startup under water load stalls at low timing, so ESC "high" timing and soft start.
- [x] Thrust: T = rho n^2 D^4 K_T. Measured bollard pull 0.7 kgf at about 4500 rpm on a 60 mm printed prop gives K_T about 0.095. Drag F = 0.5 rho C_d A v^2 with A 0.06 m^2, C_d 1.2: 9 N at 0.5 m/s, so two forward thrusters (14 N) top out near 0.6 m/s.
- [x] Battery: 3S LiPo, 20 min, average draw around 6.5 A for a 2200 mAh pack.
- [x] Waterproofing: pressure at 2 m is 20 kPa above ambient; lid gasket, PG7 cable glands, epoxy-potted connectors; paper towel test.
- [x] Buoyancy: 4 x 1 L bottles = 4 kgf lift; pipes drilled to flood so they add no uncertain buoyancy; righting moment M = m g GM sin(theta), m 6 kg, GM 5 cm gives 2.9 N m per radian.
- [x] Radio: attenuation in fresh water at 2.4 GHz from complex permittivity (eps' 77, eps'' 10 at 2.45 GHz): alpha = (omega/c) sqrt(eps'/2 (sqrt(1 + (eps''/eps')^2) - 1)) = 29 Np/m = 250 dB/m = 2.5 dB/cm. At 5.8 GHz several times worse. 10 cm of water costs 25 dB at 2.4 GHz.
- [x] Verify.

### Task 3: Ciclop (`_posts/2017-03-26-ciclop.md`)

- [x] Camera model: pinhole, K = [[fx,0,cx],[0,fy,cy],[0,0,1]], Zhang calibration with the checkerboard, radial k1 k2 and tangential p1 p2.
- [x] Laser plane: n . X = d, fit from the laser stripe on the checkerboard at several tilts; ray r = K^-1 [u v 1]^T; intersection X = (d / (n . r)) r; rotate by turntable angle theta about the calibrated axis.
- [x] Resolution: dz = z^2 du / (f b); f 800 px, baseline 150 mm, standoff 250 mm gives 0.52 mm per pixel, the poster's 0.5 mm. Subpixel centroid nominally 0.2 px, swamped by calibration and bearing wobble.
- [x] Turntable: 200 steps x 16 = 3200 microsteps/rev, 800 slices per rev = 4 microsteps per slice = 0.45 deg, arc 0.79 mm at 100 mm radius.
- [x] Two-laser ghost: 0.3 deg plane misalignment at 250 mm = 1.3 mm double wall.
- [x] Segmentation: laser-on minus laser-off, R/(G+B) ratio threshold, per-row weighted centroid. Camera lock via `v4l2-ctl` on Linux.
- [x] Mesh: k=20 normals, Poisson depth 9, statistical outlier removal k=50, std 1.0.
- [x] Verify.

### Task 4: Flying years (`_posts/2017-04-30-the-flying-years.md`)

- [x] Trainer numbers: 550 g, 1000 mm span, 220 mm chord, S = 22 dm^2, wing loading 25 g/dm^2; V_stall = sqrt(2W/(rho S C_Lmax)) = 5.8 m/s with C_Lmax 1.2.
- [x] Stability: static margin 10%; tail volume V_H = S_t l_t / (S c) about 0.5; flying wing needs reflex airfoil and CG at 20% MAC; elevon mix.
- [x] Blue Thunder: 1100 KV on 3S: 12,200 rpm no-load, about 10,000 loaded; 10x4.7 pitch speed 20 m/s; about 180 W, 16 A; gap/chord 1.0, 15% stagger.
- [x] Multirotor: mixing table, rate PID inner loop at 400 Hz, complementary filter angle = 0.98(angle + gyro dt) + 0.02 accel; vibration aliasing.
- [x] Duct: momentum theory, thrust gain (2 sigma_d)^(1/3), 26% ideal at sigma_d = 1; our thermocol rings measured no gain because the tip gap was 3 mm on a 200 mm rotor.
- [x] EDF: 64 mm fan, 700 g thrust: v_e = sqrt(T / (rho A)) = 42 m/s, ideal power 140 W, real about 300 W, 27 A on 3S, 20C on a 1300 mAh pack.
- [x] Rocket: KNO3:sucrose 65:35, Isp about 130 s; C-class 10 N s; Barrowman CP; one caliber margin; simulated apogee about 150 m.
- [x] Verify.

### Task 5: Bioprinter (`_posts/2019-02-03-hardware-is-hard.md`)

- [x] Two toolheads: pneumatic (regulator + solenoid, the 20/40 psi sweep) and the mechanical syringe pump. Say when each is used.
- [x] Rheology: Herschel-Bulkley tau = tau_y + K gamma^n; wall shear rate gamma_w = (3n+1)/(4n) * 4Q/(pi r^3); 22G needle (r 0.205 mm) at 10 uL/s gives 1480 s^-1 Newtonian; wall shear stress tau_w = dP r / 2L: 40 psi over 12.7 mm = 2.2 kPa; viability threshold cited as low single-digit kPa.
- [x] Syringe pump: 1 mL syringe ID 4.7 mm, T8x2 leadscrew, 3200 microsteps/rev = 0.625 um per microstep = 10.8 nL; a 0.4 x 0.3 mm line needs 11 microsteps per mm; retract 2 mm plunger travel before moves.
- [x] Motion: A4988 1/16, GT2 20T = 80 steps/mm on X/Y, T8x8 Z = 400 steps/mm, Marlin trapezoid planner, junction deviation 0.05 mm, homing at 5 mm/s, repeatability 0.05 mm by dial gauge.
- [x] Curing: GelMA 5% w/v, Irgacure 2959 0.5% w/v, 365 nm at 10 mW/cm^2, 60 s per layer = 600 mJ/cm^2; oxygen inhibition at the surface; alginate 2% CaCl2 mist.
- [x] Layer correction: oblique camera plus line laser, same triangulation as the scanner; z_{k+1} = z_k + h_nom + K (h_meas - h_nom), K = 0.4, clamp 0.1 mm; error decays as (1-K)^n, 8% after five layers; K > 1 oscillates.
- [x] Verify.

### Task 6: DeepSafe (`_posts/2022-08-21-deepsafe.md`)

- [x] Namespace isolation internals: `importlib.util.spec_from_file_location`, synthetic package prefix, `sys.modules` scoping, rewriting relative imports, C-extension caveat.
- [x] CUDA context cost (300-500 MB each), one process = one context; CUDA streams per worker thread; pinned host memory.
- [x] GIL: torch releases it inside kernels; preprocessing in numpy/PIL is the part that doesn't; `concurrent.futures.wait` timeouts can't kill a running kernel, so the future is abandoned and the vote imputed.
- [x] FP16: which layers overflow (softmax, batchnorm stats), autocast, and the two models kept at FP32.
- [x] Ensemble: stacking with out-of-fold predictions to avoid leakage; ECE = sum |B_m|/n * |acc - conf|; Brier; Platt vs isotonic.
- [x] Feature families explained: DCT high-frequency artifacts from upsampling, backbone features, reconstruction error, audio-visual sync; audio front-ends LFCC and raw-waveform; 64,600 samples = 4.04 s at 16 kHz.
- [x] Provenance: C2PA manifest verification steps (JUMBF box, hashed assertions, COSE signature chain).
- [x] Verify.

### Task 7: Factory-grade agents (`_posts/2026-05-24-factory-grade-agents.md`)

- [x] Reference architecture section: perception, belief state, planner, contracts, executor, audit; sequence propose -> verify -> approve -> execute -> observe -> postcondition.
- [x] Freshness proof: signed snapshot with monotonic clock, per-tag age, variance liveness test; staleness bound from process time constant (a 10-minute thermal constant means 60 s max age).
- [x] Idempotency and fencing: action IDs, fencing tokens, at-least-once with dedup; OPC UA subscription sampling interval and deadband.
- [x] Error pricing: Bayes threshold act if P(defect) > c_FP / (c_FP + c_FN); worked example.
- [x] Audit: append-only hash-chained action log schema.
- [x] Verify.

### Task 8: Shipping AI models (`_posts/2026-07-26-shipping-ai-models.md`)

- [x] Download: HTTP Range resume, ETag validation, sidecar with byte offset and expected SHA256, chunked hashing; NFS close-to-open consistency and SMB write coalescing as the corruption mechanism.
- [x] Install: rename(2) atomicity, EXDEV, fsync the directory, tarfile hardening (absolute paths, `..`, symlinks, device nodes).
- [x] Load: file-backed mappings; overwriting the same inode changes pages under a running process (SIGSEGV/SIGBUS); replace-by-rename keeps the old inode alive; the lock still needed for multi-file consistency; flock-based reader-writer with writer preference across Node and Python.
- [x] Run: probe order and CUDA driver/runtime mismatch; CTranslate2 compute types; int8 dynamic quantization per-channel; ONNX session creation cost; SIGKILL 137 from the OOM killer and cgroup limits; ARM64 wheels and NEON.
- [x] Verify. No new paths or issue numbers.

### Task 9: Factory agent evals (`_posts/2026-08-30-factory-agent-evals.md`)

- [x] Flatline detection: rolling variance, Hampel filter, resolution-aware repeat threshold; staleness from time constant.
- [x] FMEA worked table with RPN = S x O x D for four failure modes.
- [x] Temporal guard implementation (sliding window) and envelope with hysteresis, added to the code block.
- [x] Shadow metrics: Cohen's kappa; Wilson interval for promotion gates; CUSUM on override rate.
- [x] Scenario file format for the regression suite.
- [x] Verify.

### Task 10: Homelab (`_posts/2026-09-01-homelab.md`)

- [x] Apply the scrub table above.
- [x] DNS: ACME DNS-01 with a `_acme-challenge` TXT record, wildcard cert, 90-day renewal.
- [x] Request path: nginx `stream` block for 443 passthrough with PROXY protocol as the planned fix for CrowdSec blindness; `real_ip` handling.
- [x] pf: anchor file, `pfctl -a crowdsec -T replace`, rule ordering with `quick`; launchd `StartInterval`.
- [x] Watchdogs: launchd `KeepAlive`, `docker inspect` health status, flap-guard state file.
- [x] Postgres on USB: I/O errors during WAL write cause PANIC and torn pages; why the SSD move fixed it.
- [x] Backups: `pg_dumpall` vs `pg_dump -Fc`, SQLite online backup API, age X25519, orphan-commit mechanics and why encryption makes GitHub's unreachable-object retention safe.
- [x] Tailscale: WireGuard, DERP fallback, ACL grants for the admin tools.
- [x] Compose: healthchecks and `depends_on: condition: service_healthy`.
- [x] Verify scrub with a grep for duckdns, ISP, bedroom, /12, Sunday, password manager, basic-auth, subdomains.

### Task 11: Verification and handoff

- [x] `python3 scripts/prose-tells.py _posts/*.md` shows no em dashes and no honest/actual tics.
- [x] Media and external link counts per post are >= the previous counts (new links allowed, none lost).
- [x] `bundle exec jekyll build` succeeds.
- [x] Report to owner; push only on request.
