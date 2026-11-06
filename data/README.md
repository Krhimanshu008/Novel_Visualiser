# Novel Visualiser — Data Directory Reference

This folder is the **single canonical source of truth** for all input texts, character turnaround references, and research dossiers.

```
data/
├── novels/
│   └── Shadow Slave - Volume 1 To 7 Collection.epub   # Source novel eBooks & texts
│
├── references/                                        # Reference images for ComfyUI IP-Adapter Plus
│   ├── Nephis/                                        # Official illustrations & turnaround sheets
│   └── Sunny/                                         # Character face & body reference plates
│
└── research_exports/                                  # Exported character intelligence dossiers
    ├── Nephis_dossier.json                            # Serialized machine-readable research data
    ├── Nephis_dossier.md                              # Human-readable markdown research dossier
    ├── Sunny_dossier.json
    └── Sunny_dossier.md
```

For full documentation on how data is harvested, spoiler-gated, and persisted to SQLite, see [RESEARCH_AGENT_GUIDE.md](../RESEARCH_AGENT_GUIDE.md).
