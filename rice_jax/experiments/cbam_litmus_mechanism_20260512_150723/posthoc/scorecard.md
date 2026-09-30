## Litmus Results Summary (auto-generated)

### Pass Table

| Test | Family | Question | Value | Pass? | Strength |
|------|--------|----------|-------|-------|----------|
| M1 | Fixed diff | Does diversion exist? | 30.3% rel drop | PASS | Strong |
| M2 | Fixed diff | Costless mitigation when pinned? | μ = 0.548 | PASS | Strong |
| M3 | Fixed diff | Costly mitigation survives when pinned? | μ = 0.492 | PASS | Moderate |
| M4 | Fixed diff | Diversion crowds out mitigation? | μ_M4 < μ_M3 by 0.024 | PASS | Strong |
| N1 | Null | Mechanical tariff-relief? | cost_high < cost_low | PASS | Mechanical |

### Anomaly Flags

- ⚠ M1 per-region inversion (diff > ctrl): India

### One-Sentence Summaries

> **Fixed-differential:** The fixed-differential litmus suite shows that both diversion and mitigation are viable responses to CBAM, but when both margins are open, exporters shift toward diversion and away from costly mitigation.

### Layer 2: Network Introspection

**m1/agent_ctrl/r1**

- Top Jacobian groups: utility (1.466), cbam_cost (0.997), dest_alloc (0.648)
- Counterfactual (zero CBAM obs): L1 logit shift = 0.0000

**m1/agent_ctrl/r2**

- Top Jacobian groups: utility (1.191), timestep (0.727), cbam_cost (0.551)
- Counterfactual (zero CBAM obs): L1 logit shift = 0.0000

**m1/agent_ctrl/r4**

- Top Jacobian groups: timestep (0.792), gross_output (0.340), utility (0.323)
- Counterfactual (zero CBAM obs): L1 logit shift = 0.0000

**m1/agent_ctrl/r5**

- Top Jacobian groups: cbam_tariff_rate (1.116), timestep (0.952), cbam_cost (0.787)
- Counterfactual (zero CBAM obs): L1 logit shift = 0.0000

**m1/agent_ctrl/r6**

- Top Jacobian groups: transfer_rcvd (0.446), gross_output (0.422), timestep (0.412)
- Counterfactual (zero CBAM obs): L1 logit shift = 0.0000

**m1/agent_ctrl/r7**

- Top Jacobian groups: utility (0.675), cbam_lambda (0.486), trade_flows (0.177)
- Counterfactual (zero CBAM obs): L1 logit shift = 0.0000

**m1/agent_ctrl/r8**

- Top Jacobian groups: revenue_share (0.541), transfer_rcvd (0.390), cbam_lambda (0.348)
- Counterfactual (zero CBAM obs): L1 logit shift = 0.0000

**m1/agent_diff/r1**

- Top Jacobian groups: cbam_tariff_rate (0.837), trade_flows (0.474), dest_alloc (0.322)
- Counterfactual (zero CBAM obs): L1 logit shift = 78142.2109

**m1/agent_diff/r2**

- Top Jacobian groups: transfer_rcvd (0.654), utility (0.511), revenue_share (0.366)
- Counterfactual (zero CBAM obs): L1 logit shift = 121568.1797

**m1/agent_diff/r4**

- Top Jacobian groups: transfer_rcvd (0.236), cbam_lambda (0.211), cbam_revenue (0.195)
- Counterfactual (zero CBAM obs): L1 logit shift = 64112.6719

**m1/agent_diff/r5**

- Top Jacobian groups: timestep (1.166), cbam_cost (0.415), cbam_revenue (0.369)
- Counterfactual (zero CBAM obs): L1 logit shift = 119797.0703

**m1/agent_diff/r6**

- Top Jacobian groups: timestep (0.314), cbam_lambda (0.167), trade_flows (0.156)
- Counterfactual (zero CBAM obs): L1 logit shift = 71855.8516

**m1/agent_diff/r7**

- Top Jacobian groups: timestep (8.319), revenue_share (3.733), dest_alloc (3.341)
- Counterfactual (zero CBAM obs): L1 logit shift = 252753.6250

**m1/agent_diff/r8**

- Top Jacobian groups: revenue_share (0.640), timestep (0.552), gross_output (0.382)
- Counterfactual (zero CBAM obs): L1 logit shift = 137579.3125

**m2/agent/r1**

- Top Jacobian groups: cbam_cost (13.244), dest_alloc (12.143), cbam_tariff_rate (6.796)
- Counterfactual (zero CBAM obs): L1 logit shift = 147856.8750

**m2/agent/r2**

- Top Jacobian groups: timestep (37.180), utility (21.564), dest_alloc (14.649)
- Counterfactual (zero CBAM obs): L1 logit shift = 229125.4531

**m2/agent/r4**

- Top Jacobian groups: dest_alloc (9.478), cbam_cost (8.777), timestep (8.388)
- Counterfactual (zero CBAM obs): L1 logit shift = 159844.0000

**m2/agent/r5**

- Top Jacobian groups: utility (3.598), cbam_tariff_rate (3.508), gross_output (3.301)
- Counterfactual (zero CBAM obs): L1 logit shift = 145216.0156

**m2/agent/r6**

- Top Jacobian groups: utility (8.545), timestep (6.837), gross_output (3.514)
- Counterfactual (zero CBAM obs): L1 logit shift = 205501.9531

**m2/agent/r7**

- Top Jacobian groups: utility (5.748), dest_alloc (4.007), timestep (3.078)
- Counterfactual (zero CBAM obs): L1 logit shift = 149012.3125

**m2/agent/r8**

- Top Jacobian groups: utility (2.727), cbam_cost (2.103), timestep (1.564)
- Counterfactual (zero CBAM obs): L1 logit shift = 133592.1406

**m3/agent/r1**

- Top Jacobian groups: timestep (28.470), dest_alloc (17.314), revenue_share (7.457)
- Counterfactual (zero CBAM obs): L1 logit shift = 200300.2969

**m3/agent/r2**

- Top Jacobian groups: cbam_cost (6.943), gross_output (5.625), dest_alloc (5.372)
- Counterfactual (zero CBAM obs): L1 logit shift = 164581.0781

**m3/agent/r4**

- Top Jacobian groups: cbam_cost (13.455), dest_alloc (8.896), revenue_share (5.692)
- Counterfactual (zero CBAM obs): L1 logit shift = 100719.3984

**m3/agent/r5**

- Top Jacobian groups: dest_alloc (2.452), gross_output (2.226), cbam_cost (1.681)
- Counterfactual (zero CBAM obs): L1 logit shift = 182846.0000

**m3/agent/r6**

- Top Jacobian groups: utility (26.191), timestep (22.923), dest_alloc (12.262)
- Counterfactual (zero CBAM obs): L1 logit shift = 166488.5469

**m3/agent/r7**

- Top Jacobian groups: utility (8.484), timestep (2.809), cbam_tariff_rate (2.778)
- Counterfactual (zero CBAM obs): L1 logit shift = 108912.6719

**m3/agent/r8**

- Top Jacobian groups: utility (31.001), cbam_cost (16.237), dest_alloc (14.326)
- Counterfactual (zero CBAM obs): L1 logit shift = 249451.4375

**m4/agent/r1**

- Top Jacobian groups: gross_output (0.509), trade_flows (0.403), cbam_revenue (0.400)
- Counterfactual (zero CBAM obs): L1 logit shift = 99052.5625

**m4/agent/r2**

- Top Jacobian groups: timestep (0.537), dest_alloc (0.426), trade_flows (0.370)
- Counterfactual (zero CBAM obs): L1 logit shift = 87734.2188

**m4/agent/r4**

- Top Jacobian groups: timestep (2.293), utility (0.986), gross_output (0.927)
- Counterfactual (zero CBAM obs): L1 logit shift = 66751.2969

**m4/agent/r5**

- Top Jacobian groups: cbam_lambda (0.497), cbam_cost (0.432), utility (0.374)
- Counterfactual (zero CBAM obs): L1 logit shift = 86799.6719

**m4/agent/r6**

- Top Jacobian groups: utility (0.544), cbam_cost (0.492), cbam_lambda (0.294)
- Counterfactual (zero CBAM obs): L1 logit shift = 99089.6406

**m4/agent/r7**

- Top Jacobian groups: utility (6.881), gross_output (3.633), dest_alloc (2.505)
- Counterfactual (zero CBAM obs): L1 logit shift = 202523.2812

**m4/agent/r8**

- Top Jacobian groups: cbam_cost (1.100), timestep (0.826), revenue_share (0.351)
- Counterfactual (zero CBAM obs): L1 logit shift = 127352.1172
