import re

class TabParser:
    """
    Takes the user edited tab and parses it to be structured to convert to correct format
    """
    def __init__(self):
        self.strings = []
        self.string_names = ['E', 'B', 'G', 'D', 'A', 'E']
    
    def parse(self, ascii_tab):
        """
        Parses ASCII tab into structured data format
        """
        lines = ascii_tab.split('\n')
        group = self._group_tab(lines)
        all_notes = []
        time_offset = 0
        
        for bar in group:
            if not bar:
                continue
            line_length = max(len(s) for s in bar)
            for x in range(line_length):
                for string_idx, string_line in enumerate(bar):
                    
                    # Safety check incase user removes a dash
                    if x >= len(string_line):
                        continue
                    
                    char = string_line[x]
                    
                    if char.isdigit():
                        if x > 0 and string_line[x-1].isdigit():
                            continue
                        fret_str = char
                        
                        # Safety check for index out of string crash prevention
                        if x + 1 < line_length and string_line[x + 1].isdigit():
                            fret_str += string_line[x + 1]
                        fret = int(fret_str)
                        
                        all_notes.append({
                            'string_idx': string_idx,
                            'fret': fret,
                            'relative_pos': time_offset + x
                        }) 
                time_offset += line_length
        return all_notes
    
    def _group_tab(self, lines):
        """
        Finds the group of 6 lines
        """
        group = []
        current = []
        for line in lines:
            cleaned_line = line.strip()
            
            if "|" in cleaned_line or cleaned_line.startswith(("E", "B", "G", "D", "A", "e")):
                current.append(line)
            if len(current) == 6:
                group.append(current)
                current = []
        return group
                