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
        tab = [["E|"],
               ["B|"],
               ["G|"],
               ["D|"],
               ["A|"],
               ["e|"] 
        ]
        beat_line = ["  "]
        steps = 0 
        time_sig_count = 0
        beat_counter = 0
        beat_step = 1
        max_line = 8 * time_signature
        for group in note_data:
            if beat_counter % 2 == 0:
                time_label = str(beat_step)
                beat_step += 1
                if beat_step > time_signature:
                    beat_step = 1
            else:
                time_label = "+"
            beat_line.append(time_label.rjust(2))
            beat_counter += 1
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
                beat_line.append("|")
                for tab_line in tab:
                    tab_line.append("|")
                time_sig_count = 0
            steps += 1
            if steps == max_line:
                print(' '.join(beat_line))
                for tab_line in tab: 
                    tab_line.append("|")
                    print(' '.join(tab_line))
                    print()
                print()
                print()
                tab = [["E|"], ["B|"], ["G|"], ["D|"], ["A|"], ["e|"]]
                beat_line = ["  "]
                steps = 0
        if steps > 0:
            beat_line.append(" |")
            for tab_line in tab:
                tab_line.append(" |")
            print(' '.join(beat_line))
            for tab_line in tab:
                tab_line.append("|")
                print(' '.join(tab_line))
            print()


