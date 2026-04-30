"""Combine 1_a/b/c/d.drawio into a single-canvas 2x2 figure_1.drawio."""
import xml.etree.ElementTree as ET

PANELS = [
    ("a", "1_a.drawio", 0,    0),
    ("b", "1_b.drawio", 1380, 0),
    ("c", "1_c.drawio", 0,    980),
    ("d", "1_d.drawio", 1940, 980),
]

PAGE_W, PAGE_H = 3800, 1960


def add_offset(value: str, offset: int) -> str:
    v = float(value) + offset
    return str(int(v)) if v == int(v) else str(v)


def offset_point(elem, dx, dy):
    if elem.get("x") is not None:
        elem.set("x", add_offset(elem.get("x"), dx))
    if elem.get("y") is not None:
        elem.set("y", add_offset(elem.get("y"), dy))


all_cells = []

for letter, fname, dx, dy in PANELS:
    tree = ET.parse(fname)
    root_elem = tree.getroot().find("diagram").find("mxGraphModel").find("root")
    for cell in root_elem.findall("mxCell"):
        cid = cell.get("id")
        if cid in ("0", "1"):
            continue
        cell.set("id", f"{letter}_{cid}")
        for attr in ("parent", "source", "target"):
            v = cell.get(attr)
            if v and v not in ("0", "1"):
                cell.set(attr, f"{letter}_{v}")
        geom = cell.find("mxGeometry")
        if geom is not None:
            offset_point(geom, dx, dy)
            for pt in geom.iter("mxPoint"):
                offset_point(pt, dx, dy)
        all_cells.append(cell)

out_root = ET.Element("mxfile", host="Electron", agent="Claude")
diag = ET.SubElement(out_root, "diagram", id="figure_1_combined", name="Figure 1")
model = ET.SubElement(
    diag, "mxGraphModel",
    dx="2000", dy="1500", grid="0", gridSize="10", guides="1", tooltips="1",
    connect="1", arrows="1", fold="1", page="1", pageScale="1",
    pageWidth=str(PAGE_W), pageHeight=str(PAGE_H),
    math="0", shadow="0", adaptiveColors="auto",
)
rt = ET.SubElement(model, "root")
ET.SubElement(rt, "mxCell", id="0")
ET.SubElement(rt, "mxCell", id="1", parent="0")
for c in all_cells:
    rt.append(c)

ET.ElementTree(out_root).write("figure_1.drawio", encoding="unicode", xml_declaration=False)
print(f"wrote figure_1.drawio with {len(all_cells)} cells, page {PAGE_W}x{PAGE_H}")
