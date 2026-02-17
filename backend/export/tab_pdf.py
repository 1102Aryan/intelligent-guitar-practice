from fpdf import FPDF

class TabPDF(FPDF):
    def header(self):
        self.set_font('Courier', "B", 16)
        self.cell(0, 5, 'Guitar Tab Transcription', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Courier', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
def split_tabs(text, max_chars=110):
    """
    Splits long tabs into blocks so it fits into the page width
    """
    all_lines = text.split('\n')
    processed_lines = []
    
    x = 0
    while x < len(all_lines):
        if x + 6 > len(all_lines):
            processed_lines.extend(all_lines[x:])
            break
        
        chunks = all_lines[x : x+6]
        
        is_tab = (len(chunks) == 6 and all('|' in i for i in chunks))
        if is_tab:
            max_len_in_block = max(len(line) for line in chunks)
            chunks = [line.ljust(max_len_in_block, '-') for line in chunks]
            line_len = len(chunks[0])
            
            if line_len > max_chars:
                start = 0
                
                while start < line_len:
                    end = start + max_chars
        
                    for single_line in chunks:
                        processed_lines.append(single_line[start:end])    
                    processed_lines.append("")
                    
                    start += max_chars 
            else:
                processed_lines.extend(chunks)
            x += 6
        else:
            processed_lines.append(all_lines[x])
            x += 1
    return processed_lines   
        
def export_to_pdf(text, output_file="output.pdf"):
    """
    Converts tab parser text to be saved as PDF
    """
    pdf = TabPDF()
    pdf.set_margins(8, 20, 8)
    pdf.add_page()
    
    pdf.set_font("Courier", size=8.5)
    wrapped = split_tabs(text, max_chars=110)
    for all_lines in wrapped:
        pdf.cell(0, 5, text=all_lines, new_x='LMARGIN', new_y="NEXT")
    pdf.output(output_file)
    return output_file

