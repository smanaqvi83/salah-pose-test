// Pure pose classifier for a SIDE-VIEW camera. No DOM, no MediaPipe — testable with node.
// Input: 33 MediaPipe landmarks {x, y, visibility} (normalized 0..1), plus aspect = videoWidth / videoHeight.
// Image y grows downward.


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

// Combine left and right into one point per joint, weighted by visibility. From the side the far
// limb is hidden, but MediaPipe still estimates it close to the near one, so the blend stays usable.
function joint(lm, l, r, aspect) {
  const a = lm[l], b = lm[r];
  const va = a.visibility ?? 1, vb = b.visibility ?? 1, w = va + vb || 1;
  return { x: ((a.x * va + b.x * vb) / w) * aspect, y: (a.y * va + b.y * vb) / w, v: Math.max(va, vb) };
}

// ponytail: hand-tuned thresholds from geometry, not from data. Ceiling: will misfire on unusual
// bodies/camera angles. Upgrade path: log real sessions (Export frames CSV), tune, or train a tiny classifier.
export const TORSO_VIS = 0.3; // shoulders + hips must be at least this visible
export const KNEE_VIS = 0.35; // below this the knee angle is ignored, not trusted

export function classify(lm, aspect = 1) {
  if (!lm) return { pose: "nobody", metrics: null, flags: {} };
  const J = (l, r) => joint(lm, l, r, aspect);
  const nose = { x: lm[0].x * aspect, y: lm[0].y, v: lm[0].visibility ?? 1 };
  const sh = J(11, 12), wrist = J(15, 16), hip = J(23, 24), knee = J(25, 26), ankle = J(27, 28);
  const torsoLen = Math.hypot(sh.x - hip.x, sh.y - hip.y) || 1e-6;
  const kneeOk = knee.v >= KNEE_VIS && ankle.v >= KNEE_VIS * 0.7;

  const metrics = {
    knee: kneeOk ? jointAngle(hip, knee, ankle) : null,
    hip: knee.v >= KNEE_VIS ? jointAngle(sh, hip, knee) : null,
    tilt: torsoTilt(sh, hip),
    headDrop: (nose.y - hip.y) / torsoLen, // >0: head below hips (sajda)
    vis: Math.min(sh.v, hip.v),
    kneeVis: knee.v,
  };
  if (metrics.vis < TORSO_VIS) return { pose: "unknown", metrics, flags: {} };

  const { knee: k, tilt, headDrop } = metrics;
  const bent = k !== null && k < 120, straight = k === null || k >= 140;
  let pose = "transition";
  // Sajda: head clearly below hips with the torso tipped past horizontal. Doesn't need the knee.
  if (headDrop > 0.25 && tilt > 100 && (k === null || k < 130)) pose = "sajda";
  else if (bent && headDrop > 0) pose = "sajda";
  else if (bent && tilt < 45) pose = "jalsa";
  // Qunoot (Ja'fari, 2nd rakat): standing with hands raised to face level. Brief raises are takbir —
  // the caller tells them apart by duration.
  else if (straight && tilt < 25 && wrist.v >= TORSO_VIS && wrist.y < sh.y) pose = "qunoot";
  else if (straight && tilt < 25) pose = "qiyam";
  // Ruku: torso roughly horizontal with legs straight (or legs hidden). Sitting with knee hidden can't
  // reach this tilt band, so a hidden knee is accepted.
  else if ((k === null || k >= 130) && tilt >= 55 && tilt <= 115) pose = "ruku";

  const flags = {};
  if (pose === "ruku") {
    flags.backFlat = Math.abs(tilt - 90) <= 15;
    flags.kneesStraight = k === null ? null : k >= 160;
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

