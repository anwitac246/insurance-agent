# Insurance MAS — Live Demo Inputs
> Run `python demo.py --manual` for each scenario below and paste the inputs when prompted.
> Or use `python demo.py --preset <name>` for the quick one-liner version.

---

## Demo 1 — Clean Approval (Normal Claim)
**Story to tell supervisors:** *"A standard rear-end collision. No fraud signals. The system should approve it and calculate the correct payout."*

```
python demo.py --preset normal
```

**Or manually:**
```
Incident type:     Rear-end
Narrative:         While stopped at a red light on Main Street, the claimant's
                   vehicle was struck from behind by a silver sedan travelling at
                   approximately 30 mph. The other driver admitted fault at the
                   scene and a police report was filed. The rear bumper, trunk
                   lid, and exhaust system require full replacement.
Estimated loss:    8500
Repair shop:       Sunrise Auto Body
OCR estimate:      (leave blank — same as loss)
```

**Expected output:**
- `APPROVED`
- Fraud risk: `Low`
- No exclusions triggered
- Payout ≈ `$8,500 - deductible`

---

## Demo 2 — Staged Accident (Fraud Detection)
**Story to tell supervisors:** *"The claimant describes a catastrophic 14-car pileup — but the repair estimate from the garage is only $200. The system detects this narrative-vs-evidence mismatch."*

```
python demo.py --preset staged_accident
```

**Or manually:**
```
Incident type:     Rear-end
Narrative:         A massive multi-vehicle pileup occurred on the interstate
                   involving at least fourteen vehicles including three semi-trucks.
                   Multiple lanes were blocked for six hours. Emergency services,
                   fire department, and two ambulances were on scene. The claimant's
                   vehicle was caught in the centre of the pile-up and sustained
                   extensive structural damage throughout the chassis and engine bay.
Estimated loss:    18000
Repair shop:       Highway Collision Repairs
OCR estimate:      200
```

**Expected output:**
- `DENIED`
- Fraud risk: `High`
- Staging flag: `True`
- Anomaly: *"narrative severity 9/10 but OCR repair estimate only $200.00"*

---

## Demo 3 — Collusion Ring (Repair Shop Fraud)
**Story to tell supervisors:** *"The claim looks legitimate on the surface — minor vandalism, reasonable amount. But the repair shop is flagged as part of a known fraud network. The system cross-references it and denies."*

```
python demo.py --preset collusion_ring
```

**Or manually:**
```
Incident type:     Vandalism
Narrative:         Claimant returned to the car park after shopping to find the
                   driver-side door heavily dented and the wing mirror broken off.
                   No witnesses were present and no CCTV footage is available from
                   the car park.
Estimated loss:    3200
Repair shop:       Apex AutoBody & Collision
OCR estimate:      (leave blank — same as loss)
```

**Expected output:**
- `DENIED`
- Fraud risk: `High`
- Collusion flag: `True`
- Anomaly: *"Repair shop matches flagged collusion network"*

---

## Demo 4 — Semantic Exclusion (Policy Reasoning via RAG)
**Story to tell supervisors:** *"This is the hardest test. The policy says 'Street Racing' is excluded — but the narrative never uses those words. It describes a 'closed circuit track event'. The system uses semantic vector matching to link the two and deny the claim."*

```
python demo.py --preset semantic_exclusion
```

**Or manually:**
```
Incident type:     Total Loss
Narrative:         The claimant was driving at high speed on a closed circuit track
                   during a private performance driving event. The vehicle lost
                   traction on a hairpin corner at approximately 110 mph and struck
                   the armco barrier. The chassis is bent and the engine bay
                   sustained fire damage. The vehicle is a total loss.
Estimated loss:    24000
Repair shop:       Motorsport Engineering Ltd
OCR estimate:      (leave blank — same as loss)
```

**Expected output:**
- `DENIED`
- Exclusion triggered: `True`
- Exclusion reason: quotes the exact racing exclusion clause from the policy
- Payout: `$0.00`

---

## Demo 5 — Borderline / Ambiguous (Good for Q&A)
**Story to tell supervisors:** *"This one is deliberately ambiguous — a high-value theft claim from a customer with a slightly risky profile. Watch how the system weighs the signals and reasons through it."*

```
Incident type:     Theft
Narrative:         The claimant states the vehicle was stolen overnight from the
                   driveway of their home. The alarm did not trigger despite being
                   active. A police report was filed the following morning. The
                   vehicle has not been recovered. The claimant is requesting a
                   full replacement payout.
Estimated loss:    19500
Repair shop:       N/A - Total Loss
OCR estimate:      (leave blank — same as loss)
```

**Expected output:** This will vary based on the borrowed customer profile.
- If the customer has a clean history → likely `APPROVED` with Medium risk
- If the customer has prior denied claims → may escalate to `DENIED`
- Good talking point: *"The system's decision depends on the customer's history pulled live from MongoDB"*

---

## Demo 6 — Extreme Overstatement (Another Staging Variant)
**Story to tell supervisors:** *"Similar to Demo 2 but different framing — a catastrophic hit-and-run with a suspiciously low garage bill. Good for showing the system isn't just keyword matching."*

```
Incident type:     Hit and Run
Narrative:         A large SUV ran a red light and struck the insured vehicle
                   at full speed in the middle of a busy intersection during
                   morning rush hour. The impact was severe enough to deploy all
                   airbags and spin the vehicle 180 degrees. Multiple bystanders
                   witnessed the collision. The at-fault driver fled immediately.
                   The claimant was taken to hospital by ambulance.
Estimated loss:    15000
Repair shop:       QuickFix Garage
OCR estimate:      150
```

**Expected output:**
- `DENIED`
- Fraud risk: `High`
- Staging flag: `True`
- Narrative severity score: `8-9/10`

---

## Suggested Demo Order for Supervisors

| Order | Demo | Why |
|-------|------|-----|
| 1st | Demo 1 — Normal | Establish the baseline. Show what approval looks like. |
| 2nd | Demo 2 — Staged Accident | Most visually dramatic. High impact opener for fraud. |
| 3rd | Demo 3 — Collusion Ring | Shows cross-claim intelligence (shop network detection). |
| 4th | Demo 4 — Semantic Exclusion | Most technically impressive. RAG + semantic reasoning. |
| 5th | Demo 5 — Borderline | Best for Q&A — shows the system isn't just rule-based. |

---

## Quick Reference — Preset One-Liners

```bash
python demo.py --preset normal
python demo.py --preset staged_accident
python demo.py --preset collusion_ring
python demo.py --preset semantic_exclusion
python demo.py --preset fraud_high_risk
python demo.py --list        # see all 50 seeded claims by scenario
```

---

## Tips for the Live Demo

- Run `python demo.py --preset staged_accident` first at home to make sure Ollama is running and the model is warm. The first call takes ~10s longer.
- If a claim takes >60s, Ollama is probably cold-starting — run one dummy claim beforehand.
- For Demo 5 (borderline), run `python demo.py --list` beforehand and pick a `normal` scenario claim ID with `python demo.py --id <id>` — you'll get a real seeded customer with actual history.
- Keep the terminal font size large (18pt+) so supervisors can read the output.