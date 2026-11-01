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
assert.equal(classify(hideKnees(cases[0][1])).pose, "qiyam");
assert.equal(classify(undefined).pose, "nobody");
console.log("ok — hidden knees and no person");
