import json
import os
from pathlib import Path

help_dir = Path(r"c:\Users\5480\Desktop\Automation\Novel Visualiser\AI Studio Help")

for f in sorted(help_dir.glob("*.json")):
    data = json.loads(f.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    print(f"\n=======================================================")
    print(f" Workflow: {f.name} ({len(nodes)} nodes)")
    print(f"=======================================================")
    for n in nodes:
        nid = n.get("id")
        ntype = n.get("type")
        title = n.get("title", ntype)
        widgets = n.get("widgets_values", [])
        print(f"  Node {nid:2d} | Type: {ntype:<32} | Title: '{title}'")
        if widgets:
            # show preview of widget values
            w_str = str(widgets)
            if len(w_str) > 80:
                w_str = w_str[:77] + "..."
            print(f"          Widgets: {w_str}")

