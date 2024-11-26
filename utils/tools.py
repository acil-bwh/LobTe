"""
File: tools.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""


def get_emph_subtype_weights():
    weights = {
        'NORMALPARENCHYMA': 0.2,
        'MILDCENTRILOBULAREMPHYSEMA': 0.8,
        'MODERATECENTRILOBULAREMPHYSEMA': 1.0,
        'SEVERECENTRILOBULAREMPHYSEMA': 0.3,
        'PARASEPTALEMPHYSEMA': 0.4,
        'PANLOBULAREMPHYSEMA': 0.2
    }

    import os
    import xml.etree.ElementTree as ET
    types = list(weights)
    fname = os.path.join('ChestConventions.xml')
    tree = ET.parse(fname)
    codes = tree.getroot().find('ChestTypes')

    types_code = {}
    for t in codes:
        val = t.find('Code').text
        key = int(t.find('Id').text) << 8
        if val in types:
            types_code[key] = val

    return {k: weights[v] for k, v in codes.items()}


def get_lobe_codes():
    import os
    import xml.etree.ElementTree as ET
    types = [
        'RIGHTSUPERIORLOBE', 'RIGHTMIDDLELOBE', 'RIGHTINFERIORLOBE',
        'LEFTSUPERIORLOBE', 'LEFTINFERIORLOBE'
    ]

    fname = os.path.join('ChestConventions.xml')
    tree = ET.parse(fname)
    codes = tree.getroot().find('ChestRegions')

    lobe_codes = {}
    for t in codes:
        val = t.find('Code').text
        if val in types:
            key = int(t.find('Id').text)
            name = t.find('Name').text

            lobe_codes[key] = {
                'Code': val,
                'Name': name,
            }

    return lobe_codes
