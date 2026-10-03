# LinkedIn post — DepthWizard

Fill in the two links marked `[...]` before posting.

---

## Post

I spent a few weeks building a system that turns **one flat satellite photo into a 3D terrain you can fly through** — and measures how tall everything in it is.

It was built for Smart India Hackathon 2026, Problem Statement 26175, posed by **ISRO**. We didn't make the cut. The engineering stands on its own, so I finished it and put it online.

**🔗 Live demo:** [GITHUB_PAGES_URL]
**💻 Code:** https://github.com/prayasPradzzy/depth-wizard3D

---

**The problem**

Elevation models underpin flood modelling, landslide risk and urban planning. Getting them normally needs stereo image pairs or airborne LiDAR — expensive, and they have to be planned in advance.

Doing it from a single image runs into two walls:

→ Depth-estimation models are trained on ordinary ground-level photographs. Look straight down from orbit and the cues they rely on simply aren't there.

→ They output *relative* depth. "Taller than that one" — never "30 metres."

---

**What I did about it**

Rather than assume the domain gap, I measured it. I benchmarked against **GAMUS** — satellite imagery published with LiDAR-measured ground truth — across 240 held-out tiles and 252 million pixels.

Off-the-shelf: **6.22 m RMSE.**

Then I fine-tuned Depth Anything V2 against those LiDAR heights, retargeting the head from relative disparity to metres above ground.

After fine-tuning: **4.60 m RMSE** — 26% better, and improved on every land-cover class and all three cities.

The result I'm most pleased with: the trained model now emits **real metres with no calibration step at all**, and at 5.48 m it still beats the baseline that was *given* free scale alignment.

51 minutes of training on a 6 GB laptop GPU.

---

**Beyond the model**

→ **3D flythrough** — WASD navigation at 60–130 FPS on a million-vertex mesh, in the browser, nothing to install

→ **Flood screening for evacuation** — raise a water level and it shows what goes under. Crucially it follows where water can actually *flow*, so it won't tell you to evacuate a neighbourhood that's safe behind high ground. Set a rise rate and it turns elevation into time: *"this spot floods in 1.8 hours."*

→ **Standard geospatial output** — DSM as GeoTIFF with the coordinate system intact, plus GLB mesh export

---

**What it can't do**

Training data is three flat US cities, so the 4.60 m holds for urban sub-metre imagery and nothing more — I can't yet claim it for the Himalaya.

And it can't see through cloud. I tested it on Nepal flood imagery and it failed. Shadow length is how it judges height, and cloud erases shadows — which matters, because monsoon flood imagery is cloudy by definition.

So the architecture routes around it: build terrain beforehand from clear imagery, and take water level from radar during the event. ISRO already flies RISAT for exactly that reason.

---

Thanks to **ISRO** and **Smart India Hackathon** for a problem statement genuinely worth the time. The constraint that elevation data is expensive, and the archive of single-view imagery is enormous, is a real one.

#RemoteSensing #ComputerVision #ISRO #GIS #DeepLearning #SIH2026 #Geospatial #DisasterManagement

---

## Notes before you post

**Two links to fill in**
- `[GITHUB_PAGES_URL]` → from Settings → Pages (see DEPLOY.md)
- Optionally add the Hugging Face Space once it's up

**On tone** — the honest version is the stronger version here. The audience worth reaching is engineers who read comments, and specific measured claims survive scrutiny in a way that superlatives don't. The limitations section is doing real work: it's what makes the 4.60 m believable.

**Never** phrase it as "built for ISRO" or imply endorsement. "Problem statement posed by ISRO for SIH 2026" is accurate and safe.

**Media** — post 3–4 images as a carousel, or a 30–45 s screen recording. Video gets materially more reach than images on LinkedIn.

Suggested recording, no narration needed:
1. Orbit around the Bangkok scene (5 s)
2. Toggle **Sharp Buildings** off and on — the ramps-to-blocks change is the most visually obvious win (5 s)
3. Enter flythrough, fly between buildings with WASD (10 s)
4. Open **Flood Risk**, drag the water level up (10 s)
5. Hover to show "floods in X hours" (5 s)
6. End on the Accuracy panel with 6.22 → 4.60 (5 s)

**Timing** — Tuesday to Thursday, 9–11am IST tends to perform best.

**First comment** — put extra links there rather than in the post body; LinkedIn suppresses reach on posts with many outbound links.
