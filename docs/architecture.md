# Architecture and design decisions

```text
                     ┌──────────────────────────┐
                     │ Six provided sample CSVs │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │ Loader + quality checks  │
                     └─────┬─────────┬──────────┘
                           │         │
            ┌──────────────┘         └────────────────┐
            ▼                                         ▼
┌────────────────────────┐             ┌───────────────────────────┐
│ Material property model│             │ Financial feature pipeline│
│ ExtraTrees baseline    │             │ DS2 + DS4 date/asset join  │
└────────────┬───────────┘             └───────────┬───────────────┘
             │                                     │
             ▼                                     ▼
┌────────────────────────┐             ┌───────────────────────────┐
│ Physics checks         │             │ Commodity time-split model│
│ K/G/nu algebraic check │             │ technical vs cross-domain │
└────────────┬───────────┘             └───────────┬───────────────┘
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │ Bridge deficiency model  │
                    │ PD proxy × LGD × EAD     │
                    └────────────┬─────────────┘
                                 ▼
         ┌────────────────────────────────────────────────┐
         │ Stress engine • ESG calculator • material      │
         │ screening • MatRisk Lab decision game          │
         └───────────────────────┬────────────────────────┘
                                 ▼
                   ┌──────────────────────────────┐
                   │ Streamlit UI + FastAPI       │
                   └──────────────────────────────┘
```

## Explicit decisions

1. **Use a tabular baseline first.** It is appropriate for a short prototype with descriptors but no full crystal coordinates. Calling this a CGNN would be inaccurate.
2. **Hold out chemical systems.** Materials with similar element sets may share patterns. Grouping formulas by elemental system provides a more demanding test than a random row split, although it does not eliminate every possible leakage path.
3. **Use a chronological commodity split.** Future-day return is the target, and the training cut-off precedes the test period.
4. **Keep model risk separate from financial assumptions.** Structural-deficiency probability is not the same as time-to-failure or loan default. The conversion to a financial loss proxy is visibly labelled.
5. **Expose scenario assumptions.** The Monte Carlo engine samples uncertainty around declared rates so the UI is interactive; these rates are not calibrated facts.
6. **Use a screening baseline for inverse design.** Generated alloys need a trained predictor and chemistry-aware validation. Ranking known materials is a lower-risk prototype that can be tested with the supplied data.
