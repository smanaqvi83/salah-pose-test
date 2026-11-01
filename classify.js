// Pure pose classifier for a SIDE-VIEW camera. No DOM, no MediaPipe — testable with node.
// Input: 33 MediaPipe landmarks {x, y, visibility} (normalized 0..1), plus aspect = videoWidth / videoHeight.
// Image y grows downward.


const deg = (r) => (r * 180) / Math.PI;

// Angle at b formed by a-b-c, in degrees. Works in 2D or 3D (z defaults to 0).
export function jointAngle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
  const len = Math.hypot(v1.x, v1.y, v1.z) * Math.hypot(v2.x, v2.y, v2.z);
  if (len === 0) return 0;
  const cos = Math.max(-1, Math.min(1, (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / len));
  return deg(Math.acos(cos));
}

// Torso tilt from vertical: 0 = upright, 90 = horizontal, >90 = shoulders below hips. y points down.
export function torsoTilt(shoulder, hip) {
  const len = Math.hypot(shoulder.x - hip.x, shoulder.y - hip.y, (shoulder.z ?? 0) - (hip.z ?? 0));
  if (len === 0) return 0;
  return deg(Math.acos(Math.max(-1, Math.min(1, (hip.y - shoulder.y) / len))));
}

// Combine left and right into one point per joint, weighted by visibility.
function joint(pts, vis, l, r, aspect = 1) {
  const a = pts[l], b = pts[r];
  const va = vis[l].visibility ?? 1, vb = vis[r].visibility ?? 1, w = va + vb || 1;
  return {
    x: ((a.x * va + b.x * vb) / w) * aspect,
    y: (a.y * va + b.y * vb) / w,
    z: ((a.z ?? 0) * va + (b.z ?? 0) * vb) / w,
    v: Math.max(va, vb),
  };
}

// ponytail: hand-tuned thresholds from geometry, not from data. Ceiling: will misfire on unusual
// bodies/camera angles. Upgrade path: log real sessions (Export frames CSV), tune, or train a tiny classifier.
export const TORSO_VIS = 0.3; // shoulders + hips must be at least this visible
export const KNEE_VIS = 0.35; // below this the knee angle is ignored, not trusted
export const SIT_DROP = 0.45; // hips this many torso-lengths below standing height = sitting

// lm: 2D image landmarks (0..1). world: MediaPipe 3D world landmarks in metres (optional) — they make
// torso tilt work from ANY camera angle, incl. a phone in front of the person, where ruku in 2D just
// looks like a shorter torso. ref: { hipY, torso } measured while clearly standing (see standingRef);
// needed to tell sitting from standing when folded legs hide the knees.
export function classify(lm, aspect = 1, { world, ref } = {}) {
  if (!lm) return { pose: "nobody", metrics: null, flags: {} };
  const J = (l, r) => joint(lm, lm, l, r, aspect);
  const W = (l, r) => joint(world, lm, l, r);
  const nose = { x: lm[0].x * aspect, y: lm[0].y };
  const sh = J(11, 12), wrist = J(15, 16), hip = J(23, 24), knee = J(25, 26), ankle = J(27, 28);
  const torso2d = Math.hypot(sh.x - hip.x, sh.y - hip.y) || 1e-6;
  const kneeOk = knee.v >= KNEE_VIS && ankle.v >= KNEE_VIS * 0.7;

  // Prefer 3D for angles: correct from the front and the side. Fall back to 2D when absent.
  const g = world ? { sh: W(11, 12), hip: W(23, 24), knee: W(25, 26), ankle: W(27, 28) } : { sh, hip, knee, ankle };
  const tilt = world ? Math.max(torsoTilt(sh, hip), torsoTilt(g.sh, g.hip)) : torsoTilt(sh, hip);
  const scale = ref?.torso ?? torso2d; // standing torso length; the live one shrinks when bending toward the camera
  const metrics = {
    knee: kneeOk ? jointAngle(g.hip, g.knee, g.ankle) : null,
    hip: knee.v >= KNEE_VIS ? jointAngle(g.sh, g.hip, g.knee) : null,
    tilt,
    headDrop: (nose.y - hip.y) / scale, // >0: head below hips (sajda)
    hipDrop: ref ? (hip.y - ref.hipY) / scale : null, // >0: hips lower than when standing
    vis: Math.min(sh.v, hip.v),
    kneeVis: knee.v,
  };
  if (metrics.vis < TORSO_VIS) return { pose: "unknown", metrics, flags: {} };

  const { knee: k, headDrop, hipDrop } = metrics;
  const sitting = hipDrop !== null && hipDrop > SIT_DROP;
  const bent = (k !== null && k < 120) || (k === null && sitting);
  // With hidden knees and no standing reference yet we can't tell standing from sitting: don't guess.
  const straight = k !== null ? k >= 140 : hipDrop !== null && !sitting;
  let pose = "transition";
  if (headDrop > 0.25 && tilt > 100 && (k === null || k < 130)) pose = "sajda";
  else if (bent && headDrop > 0) pose = "sajda";
  else if (bent && tilt < 45) pose = "jalsa";
  // Qunoot (Ja'fari, 2nd rakat): standing with hands raised to face level. Brief raises are takbir —
  // the caller tells them apart by duration.
  else if (straight && tilt < 25 && wrist.v >= 0.5 && wrist.y < sh.y) pose = "qunoot";
  else if (straight && tilt < 25) pose = "qiyam";
  else if ((k === null ? !sitting : k >= 130) && tilt >= 55 && tilt <= 115) pose = "ruku";

  const flags = {};
  if (pose === "ruku") {
    flags.backFlat = Math.abs(tilt - 90) <= 15;
    flags.kneesStraight = k === null ? null : k >= 160;
  }
  return { pose, metrics, flags };
}

// Update the standing reference from a frame where the person is clearly standing with knees visible.
// Exponential average so it follows small camera/position changes. Returns the new ref (or the old one).
export function standingRef(ref, lm, aspect, result) {
  if (!lm || result.pose !== "qiyam" || result.metrics.knee === null || result.metrics.knee < 160) return ref;
  const J = (l, r) => joint(lm, lm, l, r, aspect);
  const sh = J(11, 12), hip = J(23, 24);
  const torso = Math.hypot(sh.x - hip.x, sh.y - hip.y);
  if (!ref) return { hipY: hip.y, torso };
  return { hipY: ref.hipY * 0.9 + hip.y * 0.1, torso: ref.torso * 0.9 + torso * 0.1 };
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
    if (dur(s) < minSegmentMs || s.pose === "unknown" || s.pose === "nobody") continue;
    const n = [];
    if (["qiyam", "ruku", "sajda", "jalsa"].includes(s.pose) && s.stillMs < tumaninahMs) n.push("rushed (not still ≥1s)");
    if (s.pose === "ruku") {
      rukus++;
      if (s.rukuFrames) {
        if (s.backFlat / s.rukuFrames < 0.6) n.push("back not flat");
        if (s.kneesSeen && s.kneesStraight / s.kneesSeen < 0.6) n.push("knees bent");
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


// ---- Practice guide: which step of the prayer the person is on (Fiqh Ja'fari) --------------------
// Walks the detected positions to know the rakat, whether this standing is qawmah (after ruku), and
// whether a sitting is between the two sajdas, jalsat al-istiraha, or tashahhud.
// ponytail: trusts detection; a missed sajda shifts the count until the next ruku. Upgrade path: let the
// user tap "next" to correct, and use rakat-aware expectations in the MVP state machine.
export function guideStep(segments, totalRakats = 4, { minSegmentMs = 400 } = {}) {
  let rakat = 1, rukuDone = false, sajdas = 0, step = null;
  for (const s of segments) {
    if (dur(s) < minSegmentMs || ["unknown", "nobody", "transition"].includes(s.pose)) continue;
    const lbl = label(s);
    if (lbl === "takbir") continue;
    if (lbl === "qiyam" && sajdas >= 2) { rakat++; rukuDone = false; sajdas = 0; }
    if (lbl === "ruku") { rukuDone = true; sajdas = 0; }
    // A new sajda segment only counts once; sajda -> jalsa -> sajda is two.
    if (lbl === "sajda" && step?.pose !== "sajda") sajdas++;

    let key = lbl;
    if (lbl === "qiyam") key = rukuDone ? "qawmah" : rakat <= 2 ? "qiyam" : "qiyamTasbihat";
    if (lbl === "jalsa") {
      if (sajdas <= 1) key = "betweenSajdas";
      else if (rakat === totalRakats) key = "tashahhudSalam";
      else if (rakat === 2) key = "tashahhud";
      else key = "istiraha";
    }
    step = { pose: lbl, key, rakat, seg: s };
  }
  return step;
}

// ---- Framing check for setting up alone: is the whole body in the picture? ----------------------
// lm: 2D landmarks (0..1). Returns a key the page turns into a spoken hint.
export function framing(lm) {
  if (!lm) return "nobody";
  const v = (i) => lm[i].visibility ?? 1;
  const inFrame = (i) => v(i) >= 0.5 && lm[i].x > 0.01 && lm[i].x < 0.99 && lm[i].y > 0.01 && lm[i].y < 0.99;
  const headIn = inFrame(0);
  const feetIn = inFrame(27) || inFrame(28);
  if (!headIn && !feetIn) return "farther";
  if (!headIn) return "head";
  if (!feetIn) return "feet";
  const midX = (lm[11].x + lm[12].x + lm[23].x + lm[24].x) / 4;
  if (midX < 0.2 || midX > 0.8) return "centre";
  const height = Math.max(lm[27].y, lm[28].y) - lm[0].y;
  if (height < 0.45) return "closer";
  return "ok";
}
