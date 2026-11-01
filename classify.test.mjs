// Run: node classify.test.mjs
import assert from "node:assert/strict";
import { classify } from "./classify.js";

// Build 33 landmarks from a few named points (pixel-ish units, scaled to 0..1). Both sides identical.
function body({ nose, shoulder, hip, knee, ankle, wrist = [5, 50] }, visibility = 1) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility }));
  const s = (pt) => ({ x: (pt[0] + 100) / 300, y: pt[1] / 300, visibility });
  lm[0] = s(nose);
  [11, 12].forEach((i) => (lm[i] = s(shoulder)));
  [15, 16].forEach((i) => (lm[i] = s(wrist)));
  [23, 24].forEach((i) => (lm[i] = s(hip)));
  [25, 26].forEach((i) => (lm[i] = s(knee)));
  [27, 28].forEach((i) => (lm[i] = s(ankle)));
  return lm;
}

const cases = [
  ["qiyam", body({ nose: [0, 0], shoulder: [0, 10], hip: [0, 50], knee: [0, 90], ankle: [0, 130] })],
  ["qunoot", body({ nose: [0, 0], shoulder: [0, 10], hip: [0, 50], knee: [0, 90], ankle: [0, 130], wrist: [-15, 2] })],
  ["ruku", body({ nose: [-50, 52], shoulder: [-40, 50], hip: [0, 50], knee: [0, 90], ankle: [0, 130] })],
  ["sajda", body({ nose: [-55, 128], shoulder: [-30, 115], hip: [10, 95], knee: [0, 130], ankle: [40, 130] })],
  ["jalsa", body({ nose: [38, 60], shoulder: [35, 75], hip: [35, 115], knee: [0, 130], ankle: [40, 130] })],
  ["transition", body({ nose: [-30, 5], shoulder: [-26, 19], hip: [0, 50], knee: [0, 90], ankle: [0, 130] })],
  ["unknown", body({ nose: [0, 0], shoulder: [0, 10], hip: [0, 50], knee: [0, 90], ankle: [0, 130] }, 0.2)],
];

for (const [want, lm] of cases) {
  const got = classify(lm);
  assert.equal(got.pose, want, `expected ${want}, got ${got.pose} ${JSON.stringify(got.metrics)}`);
}
assert.equal(classify(cases[2][1]).flags.backFlat, true);
console.log(`ok — ${cases.length} poses classified`);

// Ja'fari sequence notes: rakat 1 skips jalsat al-istiraha, rakat 2 has a takbir but no real qunoot.
import { annotate } from "./classify.js";
let t = 0;
const seg = (pose, ms, extra = {}) => ({ pose, start: t, end: (t += ms), stillMs: ms, rukuFrames: 0, backFlat: 0, kneesStraight: 0, ...extra });
const notesOf = (rows, lbl, nth = 0) => rows.filter((r) => r.label === lbl)[nth].notes.join("; ");
const run = annotate([
  seg("qiyam", 5000), seg("ruku", 3000), seg("qiyam", 1500), seg("sajda", 3000), seg("jalsa", 2000), seg("sajda", 3000),
  seg("transition", 600), seg("qiyam", 5000), seg("qunoot", 800), seg("qiyam", 3000), seg("ruku", 3000),
]);
assert.match(notesOf(run, "qiyam", 2), /no jalsat al-istiraha/);
assert.equal(notesOf(run, "takbir"), "");
assert.match(notesOf(run, "ruku", 1), /qunoot not detected/);
const withQunoot = annotate([seg("ruku", 3000), seg("qunoot", 6000), seg("ruku", 3000)]);
assert.doesNotMatch(notesOf(withQunoot, "ruku", 1), /qunoot/);
console.log("ok — Ja'fari sequence notes");

// Ruku/sajda when the knee is hidden (folded body, loose kameez): must still classify, not "unknown".
const hideKnees = (lm) => { [25, 26, 27, 28].forEach((i) => (lm[i] = { ...lm[i], visibility: 0.1 })); return lm; };
assert.equal(classify(hideKnees(cases[2][1])).pose, "ruku");
assert.equal(classify(hideKnees(cases[3][1])).pose, "sajda");
const standRef = { hipY: 50 / 300, torso: 40 / 300 }; // hip height and torso length from cases[0]
assert.equal(classify(hideKnees(cases[0][1])).pose, "transition", "no reference yet: must not guess");
assert.equal(classify(cases[0][1], 1, { ref: standRef }).pose, "qiyam");
assert.equal(classify(undefined).pose, "nobody");
console.log("ok — hidden knees and no person");

// FRONT camera (phone facing the person). Same 2D image for standing and ruku-ish, told apart by 3D.
import { standingRef } from "./classify.js";
const P = (x, y, z = 0, visibility = 0.9) => ({ x, y, z, visibility });
function frontView({ noseY, shY, hipY, kneeVis = 0.9 }) {
  const lm = Array.from({ length: 33 }, () => P(0.5, 0.5, 0, 0.1));
  lm[0] = P(0.5, noseY);
  lm[11] = P(0.42, shY); lm[12] = P(0.58, shY);
  lm[15] = P(0.42, hipY + 0.05); lm[16] = P(0.58, hipY + 0.05); // hands low (at sides / on knees)
  lm[23] = P(0.45, hipY); lm[24] = P(0.55, hipY);
  lm[25] = P(0.45, hipY + 0.2, 0, kneeVis); lm[26] = P(0.55, hipY + 0.2, 0, kneeVis);
  lm[27] = P(0.45, hipY + 0.4, 0, kneeVis); lm[28] = P(0.55, hipY + 0.4, 0, kneeVis);
  return lm;
}
// World landmarks in metres, hip-centred, y down, -z toward the camera.
const world = (shoulder, nose) => {
  const w = Array.from({ length: 33 }, () => P(0, 0, 0));
  w[0] = P(...nose);
  w[11] = P(-0.18, ...shoulder.slice(1)); w[12] = P(0.18, ...shoulder.slice(1));
  w[23] = P(-0.1, 0, 0); w[24] = P(0.1, 0, 0);
  w[25] = P(-0.1, 0.45, 0); w[26] = P(0.1, 0.45, 0);
  w[27] = P(-0.1, 0.9, 0); w[28] = P(0.1, 0.9, 0);
  return w;
};
const standFront = frontView({ noseY: 0.15, shY: 0.25, hipY: 0.5 });
const standRes = classify(standFront, 1, { world: world([0, -0.5, 0], [0, -0.65, 0]) });
assert.equal(standRes.pose, "qiyam");
const fref = standingRef(undefined, standFront, 1, standRes);
assert.ok(fref && Math.abs(fref.hipY - 0.5) < 1e-9);

// Ruku from the front: in 2D the torso just looks short; 3D shows it pitched toward the camera.
const rukuFront = frontView({ noseY: 0.42, shY: 0.4, hipY: 0.5 });
assert.notEqual(classify(rukuFront, 1).pose, "ruku", "2D alone can't see front ruku (the bug)");
assert.equal(classify(rukuFront, 1, { world: world([0, -0.05, -0.48], [0, 0.02, -0.62]), ref: fref }).pose, "ruku");

// Jalsa from the front with legs folded under the kameez: knees hidden, hips well below standing height.
const sitFront = frontView({ noseY: 0.5, shY: 0.6, hipY: 0.85, kneeVis: 0.1 });
assert.equal(classify(sitFront, 1, { world: world([0, -0.5, 0], [0, -0.65, 0]), ref: fref }).pose, "jalsa");

// Sajda from the front: head on the floor, hips raised behind.
const sajdaFront = frontView({ noseY: 0.9, shY: 0.8, hipY: 0.6, kneeVis: 0.2 });
assert.equal(classify(sajdaFront, 1, { world: world([0, 0.25, -0.45], [0, 0.4, -0.6]), ref: fref }).pose, "sajda");
console.log("ok — front camera: qiyam, ruku, jalsa, sajda");

// Practice guide follows a 2-rakat (Fajr) prayer through every step.
import { guideStep } from "./classify.js";
t = 0;
const fajr = [
  ["qiyam", "qiyam"], ["ruku", "ruku"], ["qiyam", "qawmah"], ["sajda", "sajda"], ["jalsa", "betweenSajdas"],
  ["sajda", "sajda"], ["jalsa", "istiraha"], ["qiyam", "qiyam"], ["qunoot", "qunoot"], ["qiyam", "qiyam"],
  ["ruku", "ruku"], ["qiyam", "qawmah"], ["sajda", "sajda"], ["jalsa", "betweenSajdas"], ["sajda", "sajda"],
  ["jalsa", "tashahhudSalam"],
];
const segsSoFar = [];
for (const [pose, want] of fajr) {
  segsSoFar.push(seg(pose, 3000));
  segsSoFar.push(seg("transition", 500));
  assert.equal(guideStep(segsSoFar, 2).key, want, `step ${segsSoFar.length / 2} (${pose})`);
}
// Rakat 3 of a 4-rakat prayer: standing means tasbihat, and the last sitting is tashahhud + salam.
t = 0;
const z = [];
const push = (p) => { z.push(seg(p, 3000)); };
for (let r = 1; r <= 4; r++) { push("qiyam"); push("ruku"); push("qiyam"); push("sajda"); push("jalsa"); push("sajda"); push("jalsa"); }
assert.equal(guideStep(z.slice(0, 15), 4).key, "qiyamTasbihat");
assert.equal(guideStep(z.slice(0, 14), 4).key, "tashahhud");
assert.equal(guideStep(z, 4).key, "tashahhudSalam");
console.log("ok — practice guide steps");

// Framing hints for setting the phone up alone.
import { framing } from "./classify.js";
const frame = (noseY, ankleY, x = 0.5) => {
  const lm = Array.from({ length: 33 }, () => ({ x, y: 0.5, visibility: 0.9 }));
  lm[0] = { x, y: noseY, visibility: noseY < 0 ? 0.1 : 0.9 };
  lm[27] = lm[28] = { x, y: ankleY, visibility: ankleY > 1 ? 0.1 : 0.9 };
  return lm;
};
assert.equal(framing(undefined), "nobody");
assert.equal(framing(frame(0.1, 0.9)), "ok");
assert.equal(framing(frame(0.1, 1.2)), "feet");
assert.equal(framing(frame(-0.1, 0.9)), "head");
assert.equal(framing(frame(-0.1, 1.2)), "farther");
assert.equal(framing(frame(0.4, 0.7)), "closer");
assert.equal(framing(frame(0.1, 0.9, 0.1)), "centre");
console.log("ok — framing hints");
