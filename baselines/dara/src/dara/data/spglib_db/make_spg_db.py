import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# the glide for a, b, c to e
# different notation for the same space group
new2old = {
    "Aem2": "Abm2",
    "Aea2": "Aba2",
    "Cmce": "Cmca",
    "Cmme": "Cmma",
    "Ccce": "Ccca",
    "Cc2e": "Cc2a",
    "Bbe2": "Bba2",
    "B2em": "B2cm",
    "Bme2": "Bma2",
    "Ae2a": "Ac2a",
    "B2eb": "B2cb",
    "C2ce": "C2cb",
    "Cm2e": "Cm2a",
    "Ae2m": "Ac2m",
    "C2me": "C2mb",
    "B2/m2/e2/m": ["B2/m2/c2/m", "B2/m2/a2/m"],
    "P4/m2_1/nc": "P4/m2_1/n2/c",
    "P4/m2_1/bm": "P4/m2_1/b2/m",
    "C2/m2/c2_1/e": "C2/m2/c2_1/a",
    "I2/b2/m2/m": "I2_1/b2_1/m2_1/m",
    "P2_1/m2/a2/m": "P2/m2/a2_1/m",
    "A2/e2/a2/a": ["A2/b2/a2/a", "A2/c2/a2/a"],
    "A2_1/e2/a2/m": "A2_1/c2/a2/m",
    "C2/c2/m2_1/e": "C2/c2/m2_1/b",
    "C2/c2/m2_1/m": "C2/c2/m2_1m",
    "P2_1/n2_1/a2/b": "P2_1/n2/a2_1/b",
    "P2/c2/a2_1/a": "P2/c2_1/a2/a",
    "B2/b2/e2/b": ["B2/b2/a2/b", "B2/b2/c2/b"],
    "P4/n2_1/mm": "P4/n2_1/m2/m",
    "C2/m2/m2/e": ["C2/m2/m2/a", "C2/m2/m2/b"],
    "P2/c2_1/a2_1/n": "P2_1/c2/a2_1/n",
    "A2/e2/m2/m": ["A2/c2/m2/m", "A2/b2/m2/m"],
    "I2/m2/c2/m": "I2_1/m2_1/c2_1/m",
    "C2/c2/c2/e": ["C2/c2/c2/a", "C2/c2/c2/b"],
    "P2/m2_1/m2/b": "P2_1/m2/m2/b",
    "P2_1/c2_1/n2/b": "P2/c2_1/n2_1/b",
    "P2_1/m2_1/a2/b": "P2_1/m2/a2_1/b",
    "P2/n2/a2_1/n": "P2_1/n2/a2/n",
    "P2_1/m2/c2_1/a": "P2_1/m2/c2_1a",
    "P2_1/c2/a2_1/m": "P2/c2_1/a2_1/m",
    "I2/b2/c2/a": "I2_1/b2_1/c2_1/a",
    "I2/m2/a2/m": "I2_1/m2_1/a2_1/m",
    "I2/m2/m2/a": "I2_1/m2_1/m2_1/a",
    "I2/m2/m2/b": "I2_1/m2_1/m2_1/b",
    "I2/c2/m2/m": "I2_1/c2_1/m2_1/m",
    "I2/c2/a2/b": "I2_1/c2_1/a2_1/b",
    "A2_1/e2/m2/a": "A2_1/b2/m2/a",
    "B2/b2_1/e2/m": "B2/b2_1/c2/m",
    "P4/n2_1/cc": "P4/n2_1/c2/c",
    "P2/c2_1/c2/b": "P2_1/c2/c2/b",
    "P2_1/n2/n2/b": "P2/n2_1/n2/b",
    "P2_1/b2/a2/b": "P2/b2/a2_1/b",
    "P2/c2/m2_1/m": "P2/c2_1/m2/m",
    "P2/c2_1/m2_1/b": "P2_1/c2_1/m2/b",
    "P2_1/n2/a2_1/b": "P2_1/n2_1/a2/b",
    "P2/c2_1/n2/n": "P2/c2/n2_1/n",
    "B2/m2_1/e2/b": "B2/m2_1/a2/b",
    "P2_1/m2/n2_1/m": ["P2_1/m2/n2_1/m", "P2_1/m2/n2_1m"],
    "P1": ["P1", "A1", "B1", "C1", "F1", "I1"],
    "P-1": ["P-1", "A-1", "B-1", "C-1", "F-1", "I-1"],
}

# a sanity check on the definition of new2old
for k, v in new2old.items():
    if isinstance(v, str):
        v = [v]

    for v_ in v:
        if re.match(r"[ABCFI]-?1", v_) is not None:
            continue

        k_new = re.sub(r"[\d_/]+", "", k)
        k_new = re.sub(r"e", "[abc]", k_new)
        v_new = re.sub(r"[\d_/]+", "", v_)

        if re.match(k_new, v_new) is None:
            print(f"{k} {v_}")


def xml2dict_sp(xml_file):
    """
    Convert xml file to dict.

    :param xml_file
    :return:
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = {}
    for child in root:
        for sub_child in child:
            if sub_child.tag == "setting":
                settings = sub_child.attrib
                if "Number" in settings:
                    settings["Setting"] = settings.pop("Number")

                wycs = {}
                for wyckoff in sub_child:
                    wycoff_letter = wyckoff.attrib["Symbol"]
                    wycs[wycoff_letter] = {
                        **wyckoff.attrib,
                        "std_notations": [
                            elem.text
                            for elem in wyckoff
                            if elem.attrib["Standard"] == "1"
                        ],
                    }

                    for i in range(len(wycs[wycoff_letter]["std_notations"])):
                        notation = wycs[wycoff_letter]["std_notations"][i]
                        if re.match(r"-x x (\w+)", notation):
                            wycs[wycoff_letter]["std_notations"][i] = re.sub(
                                r"-x x (\w+)", r"x -x \1", notation
                            )

                data.setdefault(sub_child.get("HermannMauguin"), []).append(
                    {**settings, "SpacegroupNo": child.get("Number"), "Wyckoffs": wycs}
                )
    return data


def csv2dict_sp(csv_file):
    """
    Convert csv file to dict.

    :param csv_file:
    :return:
    """
    data = {}
    keys = {
        "Hall": 0,
        "group_number": 1,
        "spacegroup": 4,
        "hall_symb": 6,
        "international_full": 8,
    }
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            hall_number = row[keys["Hall"]]
            group_number = row[keys["group_number"]]
            spacegroup = row[keys["spacegroup"]]
            international_short = row[keys["hall_symb"]]
            international_full = row[keys["international_full"]].replace(" ", "")

            data[hall_number] = {
                "group_number": group_number,
                "spacegroup": spacegroup,
                "hall_symb": international_short,
                "international_full": international_full,
            }

    return data


if __name__ == "__main__":
    data1 = xml2dict_sp(Path("spacegrp.xml"))
    data2 = csv2dict_sp(Path("spg.csv"))

    xml_international = set(data1.keys())
    csv_international = set(
        sum(
            [
                v_name
                if isinstance(
                    (
                        v_name := new2old.get(
                            v["international_full"], v["international_full"]
                        )
                    ),
                    list,
                )
                else [v_name]
                for v in data2.values()
            ],
            [],
        )
    )

    print(f"These are in xml but not in csv: {xml_international - csv_international}")
    print(f"These are in csv but not in xml: {csv_international - xml_international}")
    # merge two table
    data = {}

    for k_new, v_new in data2.items():
        hm = v_new["international_full"]
        hm = new2old.get(hm, hm)
        if (isinstance(hm, list) and any(hm_ in data1 for hm_ in hm)) or hm in data1:
            settings = (
                sum([data1[hm_] for hm_ in hm if hm_ in data1], [])
                if isinstance(hm, list)
                else data1[hm]
            )
            settings_list = []
            for setting in settings:
                setting = {**setting}
                wyckoffs = setting.pop("Wyckoffs")
                settings_list.append({"setting": setting, "wyckoffs": wyckoffs})
            data[k_new] = {**v_new, "settings": settings_list}
        else:
            data[k_new] = {
                **v_new,
                "settings": [
                    {
                        "setting": {
                            "HermannMauguin": hm,
                            "SpacegroupNo": v_new["spacegroup"],
                        },
                        "wyckoffs": {},
                    }
                ],
            }

    with (Path("__file__").parent / "spg.json").open("w") as f:
        json.dump(data, f, indent=1)
