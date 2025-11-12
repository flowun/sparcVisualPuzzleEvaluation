def get_type_from_abbreviation(abbreviation):
    translation_dict = {
            'S': 'Start',
            'E': 'End',
            '+': 'Walkable',
            'N': 'Empty Rule',
            'G': 'Gap',
            '.': 'Dot',
            'o': 'Stone',
            '*': 'Star',
            'A': 'Triangle',
            'B': 'Triangle',
            'C': 'Triangle',
            'D': 'Triangle',
            'P': 'Positive Polyshape',
            'Y': 'Negative Polyshape'
            }
    return translation_dict[abbreviation]

class Tile:
    def __init__(self, abbreviation):
        self.abbreviation = abbreviation
        self.is_walkable = abbreviation in ['S', 'E', '+', '.']
        self.is_rule_cell = abbreviation.startswith(('N', 'o', '*', 'A', 'B', 'C', 'D', 'P', 'Y'))
        self.type = get_type_from_abbreviation(abbreviation[0])
        self.color = None
        self.shape_id = None
        if '-' in abbreviation:
            self.color = abbreviation.split('-')[1]
        if self.abbreviation.startswith(('P', 'Y')):
            parts = abbreviation.split('-')
            self.shape_id = parts[2]
        if self.abbreviation.startswith(('A', 'B', 'C', 'D')):
            self.triangle_count = {'A': 1, 'B': 2, 'C': 3, 'D': 4}[abbreviation[0]]
        # print(f"Created Tile: {self.name}, Walkable: {self.is_walkable}")
