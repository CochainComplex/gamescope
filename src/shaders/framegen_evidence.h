// Confidence-raise evidence shared by the learned refiner's inference
// (cs_framegen_motion_net.comp) and its online training
// (cs_framegen_motion_net_train.comp), so serving and training can never
// disagree about when a raise is allowed. Mirrored by evidence_gate() in
// scripts/framegen-net-train.py.

// Luma residual normalised by the brighter side (floored at 1 code value).
float framegenLumaResid(float a, float b) {
    return abs(a - b) / max(1.0f, max(abs(a), abs(b)));
}

// A raise is allowed only where the refined vector predicts the pair
// (residual fades out over 0.03..0.10) and only as far as it explains the
// pair better than holding (zero motion): on static content any small vector
// "predicts" the pair, and a raise there would revive unconfident vectors as
// drag. The 0.03 floor is the same "explained" level.
float framegenEvidenceGate(float refinedResid, float heldResid) {
    float predicts = 1.0f - smoothstep(0.03f, 0.10f, refinedResid);
    float beatsHolding = clamp((heldResid - refinedResid) / max(heldResid, 0.03f), 0.0f, 1.0f);
    return predicts * beatsHolding;
}
