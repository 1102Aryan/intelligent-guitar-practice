# Creating tab data structure
class Tab:
    def __init__(self, string, fret, time, duration):
        self.string = string
        self.fret = fret
        self.time = time
        self.duration = duration
        
    def get_string(self):
        return self.string
    
    def get_fret(self):
        return self.fret
    
    def get_time(self):
        return self.time
    
    def get_duration(self):
        return self.duration
     
    # | /  slide up
    # | \  slide down
    # | h  hammer-on
    # | p  pull-off
    # | ~  vibrato
    # | +  harmonic
    # | x  Mute note   
            
    @staticmethod
    def display_ascii_tab(note_data, time_signature=4):
        output_lines = []
        
        tab = [["E|"],
               ["B|"],
               ["G|"],
               ["D|"],
               ["A|"],
               ["e|"] 
        ]
        steps = 0 
        time_sig_count = 0
        max_line = 8 * time_signature
        for group in note_data:
            string_to_fret = {}
            for note in group:
                (string, fret) = note[1]
                string_to_fret[string] = fret
            for guitar_string in range(len(tab)):
                string_number = 1 + guitar_string
                if string_number in string_to_fret:
                    fret_string = str(string_to_fret[string_number])
                    tab[guitar_string].append(fret_string.rjust(2))
                else:
                    tab[guitar_string].append(" -")
            time_sig_count += 1
            if time_sig_count == time_signature * 2:
                for tab_line in tab:
                    tab_line.append("|")
                time_sig_count = 0
                
            steps += 1
            if steps == max_line:
                for tab_line in tab: 
                    tab_line.append("|")
                    output_lines.append(' '.join(tab_line))
                output_lines.append("")
                output_lines.append("")
                
                tab = [["E|"], ["B|"], ["G|"], ["D|"], ["A|"], ["e|"]]
                steps = 0
        if steps > 0:
            for tab_line in tab:
                tab_line.append(" |")
            for tab_line in tab:
                tab_line.append("|")
                output_lines.append(' '.join(tab_line))
            output_lines.append("")
        return "\n".join(output_lines)


