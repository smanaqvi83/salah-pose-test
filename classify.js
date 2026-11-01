// Pure pose classifier for a SIDE-VIEW camera. No DOM, no MediaPipe — testable with node.
// Input: 33 MediaPipe landmarks {x, y, visibility} (normalized 0..1), plus aspect = videoWidth / videoHeight.
// Image y grows downward.

const NOSE = 0;
const LEFT = { shoulder: 11, wrist: 15, hip: 23, knee: 25, ankle: 27 };
const RIGHT = { shoulder: 12, wrist: 16, hip: 24, knee: 26, ankle: 28 };
export const MIN_VIS = 0.5;

const deg = (r) => (r * 180) / Math.PI;

// Angle at b formed by a-b-c, in degrees.
export function jointAngle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const len = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
  if (len === 0) return 0;
  const cos = Math.max(-1, Math.min(1, (v1.x * v2.x + v1.y * v2.y) / len));
  return deg(Math.acos(cos));
}

// Torso tilt from vertical: 0 = upright, 90 = horizontal, >90 = shoulders below hips.
export function torsoTilt(shoulder, hip) {
  const len = Math.hypot(shoulder.x - hip.x, shoulder.y - hip.y);
  if (len === 0) return 0;
  return deg(Math.acos(Math.max(-1, Math.min(1, (hip.y - shoulder.y) / len))));
}

// Camera sees one side best; use whichever side is more visible.
function pickSide(lm) {
  const v = (s) => Object.values(s).reduce((t, i) => t + (lm[i].visibility ?? 1), 0);
  return v(LEFT) >= v(RIGHT) ? LEFT : RIGHT;
}

// ponytail: hand-tuned thresholds from geometry, not from data. Ceiling: will misfire on unusual
// bodies/camera angles. Upgrade path: log real sessions (export CSV), tune, or train a tiny classifier.
export function classify(lm, aspect = 1) {
  const side = pickSide(lm);
  const p = (i) => ({ x: lm[i].x * aspect, y: lm[i].y, v: lm[i].visibility ?? 1 });
  const nose = p(NOSE), wrist = p(side.wrist), sh = p(side.shoulder), hip = p(side.hip), knee = p(side.knee), ankle = p(side.ankle);

  const vis = Math.min(sh.v, hip.v, knee.v);
  const metrics = {
    knee: jointAngle(hip, knee, ankle),
    hip: jointAngle(sh, hip, knee),
    tilt: torsoTilt(sh, hip),
    headBelowHip: nose.y > hip.y,
    vis,
  };
  if (vis < MIN_VIS) return { pose: "unknown", metrics, flags: {} };

  const { knee: k, tilt } = metrics;
  let pose = "transition";
  if (k < 120 && metrics.headBelowHip) pose = "sajda";
  else if (k < 120 && tilt < 45) pose = "jalsa";
  // Qunoot (Ja'fari, 2nd rakat): standing with hands raised to face level. Brief raises are takbir —
  // the caller tells them apart by duration.
  else if (k >= 140 && tilt < 25 && wrist.v >= MIN_VIS && wrist.y < sh.y) pose = "qunoot";
  else if (k >= 140 && tilt < 25) pose = "qiyam";
  else if (k >= 130 && tilt >= 55 && tilt <= 120) pose = "ruku";

  const flags = {};
  if (pose === "ruku") {
    flags.backFlat = Math.abs(tilt - 90) <= 15;
    flags.kneesStraight = k >= 160;
  }
  return { pose, metrics, flags };
}

const QUNOOT_MS = 2000; // a hands-raised hold shorter than this is a takbir, longer is qunoot
export const dur = (s) => s.end - s.start;
export const label = (s) => (s.pose === "qunoot" && dur(s) < QUNOOT_MS ? "takbir" : s.pose);

// Segments: {pose, start, end, stillMs, rukuFrames, backFlat, kneesStraight} in time order (ms).
// Walk segments in order so Ja'fari sequence checks can see what came before.
// ponytail: assumes one prayer per session and ignores tashahhud/salam. Upgrade path: full per-prayer state machine in the MVP.
export function annotate(segments, { minSegmentMs = 400, tumaninahMs = 1000 } = {}) {
  const out = [];
  let prev = null, rukus = 0, qunootSeen = false;
  for (const s of segments) {
    if (dur(s) < minSegmentMs || s.pose === "unknown") continue;
    const n = [];
    if (["qiyam", "ruku", "sajda", "jalsa"].includes(s.pose) && s.stillMs < tumaninahMs) n.push("rushed (not still ≥1s)");
    if (s.pose === "ruku") {
      rukus++;
      if (s.rukuFrames) {
        if (s.backFlat / s.rukuFrames < 0.6) n.push("back not flat");
        if (s.kneesStraight / s.rukuFrames < 0.6) n.push("knees bent");
      }
      if (rukus === 2 && !qunootSeen) n.push("qunoot not detected before this ruku (mustahab)");
    }
    if (label(s) === "qunoot") qunootSeen = true;
    if (s.pose === "qiyam" && prev?.pose === "sajda") n.push("stood straight from sajda — no jalsat al-istiraha (mustahab)");
    if (s.pose !== "transition") prev = s;
    out.push({ s, label: label(s), notes: n });
  }
  return out;
}

