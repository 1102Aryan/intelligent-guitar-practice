import sys
import os


from PySide6.QtCore import Qt, QThread, Signal, QObject, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel,
                               QFileDialog, QFrame, QSizePolicy, QComboBox, QSlider,
                               QTextEdit)
from PySide6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor

current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)              
sys.path.append(root_dir)                          

from backend.export.exporter import *
from backend.export.midi_exporter import *
from backend.export.exporter import export_to_gp5
from backend.export.tab_pdf import export_to_pdf
from backend.main import automatic_music_transcription
from backend.tablature.tab_generator import Tab
from backend.analysis.note_filters import group_notes

class TabEditor(QTextEdit):
    """
    Custom editor for guitar tabs.
    """
    def __init__(self):
        super().__init__()
        font = QFont("Consolas", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.document().setDocumentMargin(4)

    def setPlainText(self, text):
        super().setPlainText(text)
        self._apply_center()

    def _apply_center(self):
        self.selectAll()
        self.setAlignment(Qt.AlignHCenter)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)


class TranscriptionWorker(QThread):
    finished = Signal(str, list, float)

    def __init__(self, file_path, selected_model, use_demucs=False):
        super().__init__()
        self.file_path = file_path
        self.selected_model = selected_model
        self.use_demucs = use_demucs

    def run(self):
        try:
            result, mapped_notes, bpm = automatic_music_transcription(
                self.file_path, None, self.selected_model, use_demucs=self.use_demucs
            )
            self.finished.emit(result, mapped_notes, bpm)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}", [], 120)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Guitar Practice Tool")
        self.resize(1000, 700)
        self.file_name = None
        self.mapped_notes = []
        self.note_index = None
        self.selected_string = None
        self.bpm = 120.0
        self.position_map = {}
        self.current_row = None
        self.current_slot = None

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 20, 10, 20)

        self.top_spacer = QWidget()
        self.top_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.top_spacer)
        
        # Model selection
        self.lbl_model_select = QLabel("Select Model:")
        self.dropdown_model = QComboBox()
        
        self.dropdown_model.addItem("Sequential Architecture (SynthDataset)")
        self.dropdown_model.addItem("End-to-End Architecture (GOAT Dataset)")
        self.dropdown_model.addItem("Heuristic (Baseline)")
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.lbl_model_select)
        model_layout.addWidget(self.dropdown_model)

        # Control panel
        self.control_panel = QFrame()
        self.control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(20, 15, 20, 15)
        

        # Audio setup
        self.audio_path = None
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        self.btn_original_audio = QPushButton("▶")
        self.seek_slider_original = QSlider(Qt.Horizontal)
        self.seek_slider_original.setRange(0, 0)
        self.lbl_time_original = QLabel("00:00 / 00:00")
        
        self.player_midi = QMediaPlayer()
        self.midi_audio_output = QAudioOutput()
        self.player_midi.setAudioOutput(self.midi_audio_output)
        
        self.slider_midi = QSlider(Qt.Horizontal)
        self.slider_midi.setRange(0, 0)
        self.lbl_time_midi = QLabel("00:00 / 00:00")
        
        
        self.btn_midi_audio = QPushButton("▶")
        self.btn_midi_audio.clicked.connect(self.play_midi)
        
        self.btn_original_audio.clicked.connect(self.play_audio)
        self.seek_slider_original.sliderMoved.connect(self.set_audio_position)
        self.slider_midi.sliderMoved.connect(self.set_midi_position)

        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_duration)
        self.player_midi.positionChanged.connect(self.update_slider_midi)
        self.player_midi.durationChanged.connect(self.update_slider_midi_duration)
        
        # Header
        lbl_title = QLabel("TRANSCRIBE AUDIO")
        lbl_title.setObjectName("PanelTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(lbl_title)
        
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.lbl_model_select)
        model_layout.addWidget(self.dropdown_model)
        control_layout.addLayout(model_layout)

        # Demucs
        self.toggle_demucs = QPushButton("Demucs  OFF")
        self.toggle_demucs.setObjectName("DemucsToggle")
        self.toggle_demucs.setCheckable(True)
        self.toggle_demucs.setChecked(False)
        self.toggle_demucs.toggled.connect(self._on_demucs_toggled)

        demucs_layout = QHBoxLayout()
        lbl_demucs = QLabel("Source Separation:")
        demucs_layout.addWidget(lbl_demucs)
        demucs_layout.addWidget(self.toggle_demucs)
        demucs_layout.addStretch()
        control_layout.addLayout(demucs_layout)
        
        lbl_original = QLabel("Original Audio")
        lbl_original.setObjectName("PlayerLabel")
        
        lbl_midi = QLabel("Midi Tab Audio")
        lbl_midi.setObjectName("PlayerLabel")
        
        # Upload button
        self.btn_upload = QPushButton("Upload Audio File")
        self.btn_upload.setMinimumHeight(38)
        self.btn_upload.clicked.connect(self.upload_file)
        control_layout.addWidget(self.btn_upload)
        
        # Audio Player
        self.audio_controls = QWidget()
        audio_inner = QHBoxLayout(self.audio_controls)
        audio_inner.setContentsMargins(0, 0, 0, 0)
        audio_inner.addWidget(lbl_original)
        audio_inner.addWidget(self.btn_original_audio)
        audio_inner.addWidget(self.seek_slider_original)
        audio_inner.addWidget(self.lbl_time_original)
        self.audio_controls.setVisible(False)
        control_layout.addWidget(self.audio_controls)
        
        # MIDI player
        self.midi_controls = QWidget()
        midi_inner = QHBoxLayout(self.midi_controls)
        midi_inner.setContentsMargins(0, 0, 0, 0)
        midi_inner.addWidget(lbl_midi)
        midi_inner.addWidget(self.btn_midi_audio)
        midi_inner.addWidget(self.slider_midi)
        midi_inner.addWidget(self.lbl_time_midi)
        self.midi_controls.setVisible(False)
        control_layout.addWidget(self.midi_controls)

        # Transcribe button
        self.btn_transcribe = QPushButton("TRANSCRIBE")
        self.btn_transcribe.setObjectName("BtnTranscribe")
        self.btn_transcribe.setMinimumHeight(42)
        self.btn_transcribe.setCursor(Qt.PointingHandCursor)
        self.btn_transcribe.clicked.connect(self.run_transcription)
        control_layout.addWidget(self.btn_transcribe)

        self.main_layout.addWidget(self.control_panel)
        # Toolbar editor button
        self.edit_actions = QWidget()
        edit_layout = QHBoxLayout(self.edit_actions)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_fret_up = QPushButton("+ Fret")
        self.btn_fret_up.setObjectName("SecondaryButton")
        self.btn_fret_down = QPushButton("- Fret")
        self.btn_fret_down.setObjectName("SecondaryButton")
        self.btn_delete_note = QPushButton("Delete Note")
        self.btn_delete_note.setObjectName("SecondaryButton")
        edit_layout.addStretch()
        edit_layout.addWidget(self.btn_fret_up)
        edit_layout.addWidget(self.btn_fret_down)
        edit_layout.addWidget(self.btn_delete_note)
        self.btn_add_note = QPushButton("Add Note")
        self.btn_add_note.setObjectName("SecondaryButton")
        edit_layout.addWidget(self.btn_add_note)
        edit_layout.addStretch()
        self.edit_actions.setVisible(False)
        self.main_layout.addWidget(self.edit_actions)

        # Editor section
        self.editor = TabEditor()
        self.editor.setPlaceholderText("Generated tabs will appear here...")
        self.editor.setVisible(False)
        self.editor.setMinimumHeight(300)
        self.editor.setReadOnly(True)
        self.editor.cursorPositionChanged.connect(self.on_cursor_moved)
        self.main_layout.addWidget(self.editor, stretch=3)
        

        # Bottom section
        self.bottom_actions = QWidget()
        action_layout = QHBoxLayout(self.bottom_actions)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_gp5 = QPushButton("Export as gp5")
        self.btn_gp5.setObjectName("SecondaryButton")
        self.btn_gp5.setMinimumHeight(40)
        self.btn_save = QPushButton("Save Tab as PDF")
        self.btn_save.setObjectName("SecondaryButton")
        self.btn_save.setMinimumHeight(40)
        action_layout.addStretch() 
        action_layout.addWidget(self.btn_gp5)
        action_layout.addWidget(self.btn_save)
        self.bottom_actions.setVisible(False) 
        self.main_layout.addWidget(self.bottom_actions)
        self.bottom_spacer = QWidget()
        self.bottom_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.bottom_spacer)
        

        

        self.synth_theme()
        self.connection()
        
    def refresh_tab(self):
        if not self.mapped_notes:
            return

        grouped_notes = group_notes(self.mapped_notes, bpm=self.bpm, subdivisions=16)
        full_tab = Tab.display_ascii_tab(grouped_notes, time_signature=4, subdivisions=16)
        scroll_v = self.editor.verticalScrollBar().value()
        scroll_h = self.editor.horizontalScrollBar().value()
        self.editor.setPlainText(full_tab)

        # Build position_map from grouped_notes
        self.position_map = {}
        for rendered_slot_idx, group in enumerate(grouped_notes):
            row = rendered_slot_idx // 16
            slot_in_row = rendered_slot_idx % 16
            for note in group:
                string_num = note[1][0]
                try:
                    idx = self.mapped_notes.index(note)
                except ValueError:
                    continue
                self.position_map[(row, slot_in_row, string_num)] = idx

        print(f"Position map built: {len(self.position_map)} entries")
        print(f"Sample keys: {list(self.position_map.keys())[:10]}")
        
        if self.dropdown_model.currentText() != "Heuristic (Baseline)":
            self.colour_confidence()

        self.editor.verticalScrollBar().setValue(scroll_v)
        self.editor.horizontalScrollBar().setValue(scroll_h)

    def on_cursor_moved(self):
        cursor = self.editor.textCursor()
        block_number = cursor.blockNumber()
        col = cursor.positionInBlock()

        # 6 string lines + 2 blank lines = 8 lines per row
        lines_per_row = 8
        row = block_number // lines_per_row
        line_in_row = block_number % lines_per_row

        # Lines 6 and 7 are blank separators
        if line_in_row >= 6:
            return

        PREFIX_WIDTH = 3
        if col < PREFIX_WIDTH:
            return

        col_after_prefix = col - PREFIX_WIDTH

        # Each slot = 3 chars
        # Each beat separator = 2 chars
        # Beat block
        SLOT_WIDTH = 3
        SLOTS_PER_BEAT = 4
        BEAT_BLOCK_WIDTH = SLOTS_PER_BEAT * SLOT_WIDTH + 2  # 14

        beat = col_after_prefix // BEAT_BLOCK_WIDTH
        col_in_beat = col_after_prefix % BEAT_BLOCK_WIDTH

        # Past the 4 slots in this beat
        if col_in_beat >= SLOTS_PER_BEAT * SLOT_WIDTH:
            return

        slot_in_beat = col_in_beat // SLOT_WIDTH
        slot_in_bar = (beat * SLOTS_PER_BEAT) + slot_in_beat

        if slot_in_bar >= 16:
            return

        string_num = line_in_row + 1
        self.selected_string = string_num
        
        self.current_row = row
        self.current_slot = slot_in_bar

        key = (row, slot_in_bar, string_num)
        print(f"Key: {key}, Map size: {len(self.position_map)}")

        if key in self.position_map:
            self.note_index = self.position_map[key]
            print(f"HIT: index={self.note_index}")
        else:
            self.note_index = None
            print(f"MISS")
            
    def fret_up(self):
        if self.note_index is None:
            return
        note_event = self.mapped_notes[self.note_index][0]
        position = self.mapped_notes[self.note_index][1]
        string_num, fret = position[0], position[1]
        confidence = position[2] if len(position) >= 3 else None
        if string_num == self.selected_string and fret < 24:
            new_pos = (string_num, fret + 1, confidence) if confidence is not None else (string_num, fret + 1)
            self.mapped_notes[self.note_index] = (note_event, new_pos)
        self.refresh_tab()

    def fret_down(self):
        if self.note_index is None:
            return
        note_event = self.mapped_notes[self.note_index][0]
        position = self.mapped_notes[self.note_index][1]
        string_num, fret = position[0], position[1]
        confidence = position[2] if len(position) >= 3 else None
        if string_num == self.selected_string and fret > 0:
            new_pos = (string_num, fret - 1, confidence) if confidence is not None else (string_num, fret - 1)
            self.mapped_notes[self.note_index] = (note_event, new_pos)
        self.refresh_tab()

    def delete_note(self):
        if self.note_index is None:
            return
        string_num = self.mapped_notes[self.note_index][1][0]
        if string_num == self.selected_string:
            self.mapped_notes.pop(self.note_index)
            self.note_index = None
        self.refresh_tab()

    def add_note(self):
        if self.selected_string is None or self.current_row is None:
            return

        onset = None
        for (r, s, _), idx in self.position_map.items():
            if r == self.current_row and s == self.current_slot:
                onset = self.mapped_notes[idx][0][0]
                break

        if onset is None:
            seconds_per_slot = (60.0 / self.bpm * 4) / 16
            onset = (self.current_row * 16 + self.current_slot) * seconds_per_slot

        STRING_OPEN = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
        midi_val = STRING_OPEN[self.selected_string]
        self.mapped_notes.append(((onset, onset + 0.2, midi_val, 1.0), (self.selected_string, 0)))
        self.refresh_tab()
                

    def _on_demucs_toggled(self, checked):
        self.toggle_demucs.setText("Demucs   ON" if checked else "Demucs  OFF")

    def connection(self):
        # Connects the export button
        self.btn_gp5.clicked.connect(self.handle_export)
        self.btn_save.clicked.connect(self.handle_pdf)
        self.btn_fret_down.clicked.connect(self.fret_down)
        self.btn_fret_up.clicked.connect(self.fret_up)
        self.btn_delete_note.clicked.connect(self.delete_note)
        self.btn_add_note.clicked.connect(self.add_note)

    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Add Audio File", "", "(*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma)")
        if file_name:
            self.btn_upload.setText(f" {file_name.split('/')[-1]}")
            self.file_name = file_name
            self.load_audio_file(self.file_name)
            self.audio_controls.setVisible(True)
            # Reset view 
            # self.reset_view()
            
    def play_audio(self):
        """
        Plays Original loaded audio file 
        """
        if not self.audio_path:
            print("No Audio File Loaded")
            return
        state = self.media_player.playbackState()
        print(f"Audio state: {state}, name: {state.name}")
        if self.media_player.playbackState().name == "PlayingState":
            self.media_player.pause()
            self.btn_original_audio.setText("▶")
        else:
            self.media_player.play()
            self.btn_original_audio.setText("||")
            
    def load_midi_file(self):
        tab_text = self.editor.toPlainText() 
        if not tab_text.strip():
            print("Editor is empty.")
            return

        wav_path = export_to_audio(tab_text, "temp_midi.wav")
        
        if wav_path and os.path.exists(wav_path):
            # Load into the Midi Player
            url = QUrl.fromLocalFile(os.path.abspath(wav_path))
            self.player_midi.setSource(url)
            self.player_midi.mediaStatusChanged.connect(self.on_midi_loaded)
            self.midi_controls.setVisible(True)
            print("MIDI audio ready")
        else:
            self.btn_generate_midi.setText("Error Generating")
            
    def on_midi_loaded(self, status):
        if status.value == 3:
            print("MIDI loaded and ready to play")
            self.player_midi.mediaStatusChanged.disconnect(self.on_midi_loaded)
    
    def play_midi(self):
        if self.player_midi.playbackState().name == "PlayingState":
            self.player_midi.pause()
            self.btn_midi_audio.setText("▶")
        else:
            self.player_midi.play()
            self.btn_midi_audio.setText("||")
            
    def load_audio_file(self, audio_file):
        """
        Runs once audio file selected
        """
        self.audio_path = audio_file
        url = QUrl.fromLocalFile(audio_file)
        self.media_player.setSource(url)
        self.btn_original_audio.setText("▶")
            
    def update_slider_duration(self, duration):
        """
        Updates the position of slide to the new file
        """
        self.seek_slider_original.setRange(0, duration)
        self.update_time_label()
        
    def update_slider_position(self, position):
        """
        Updates when the slider position is changed
        """
        self.seek_slider_original.setValue(position)
        self.update_time_label()
    
    def set_audio_position(self, position):
        self.media_player.setPosition(position)

    def set_midi_position(self, position):
        self.player_midi.setPosition(position)

    def update_slider_midi_duration(self, duration):
        self.slider_midi.setRange(0, duration)
        self.update_time_label_midi()

    def update_slider_midi(self, position):
        self.slider_midi.setValue(position)
        self.update_time_label_midi()

    def update_time_label_midi(self):
        current = self.player_midi.position()
        total = self.player_midi.duration()
        current_fmt = f"{int(current // 60000):02}:{int((current % 60000) // 1000):02}"
        total_fmt = f"{int(total // 60000):02}:{int((total % 60000) // 1000):02}"
        self.lbl_time_midi.setText(f"{current_fmt} / {total_fmt}")
    
    def update_time_label(self):
        """
        Displays the time duration in mm:ss
        """
        current = self.media_player.position()
        total = self.media_player.duration()

        # Convert milliseconds to minutes and seconds
        current_fmt = f"{int(current // 60000):02}:{int((current % 60000) // 1000):02}"
        total_fmt = f"{int(total // 60000):02}:{int((total % 60000) // 1000):02}"
        
        self.lbl_time_original.setText(f"{current_fmt} / {total_fmt}")
        

    def run_transcription(self):
        """
        Simulates the transcription process.
        In the real app, this would be called after the ML model finishes.
        """
        if not self.file_name:
            self.btn_upload.setText("Please add a Audio File first.")
            return
        
        self.worker = TranscriptionWorker(self.file_name, self.dropdown_model.currentText(), self.toggle_demucs.isChecked())
        self.btn_transcribe.setText("Transcribing...")
        self.btn_transcribe.setEnabled(False)
        self.worker.finished.connect(self.handle_result)
        self.worker.start()  

        self.top_spacer.setVisible(False)
        self.bottom_spacer.setVisible(False)
        self.editor.setVisible(True)
        self.bottom_actions.setVisible(True)

    def handle_export(self):
        """
        Generates MusicXML format of the tab
        """     
        edited_text = self.editor.toPlainText()
        
        if not edited_text.strip():
            self.btn_gp5.setText("Generate a Tab.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Guitar Pro", "", "Guitar Pro 5 (*.gp5)")
        
        if file_path:
            if not file_path.lower().endswith('.gp5'):
                file_path += '.gp5'
            try:
                export_to_gp5(edited_text, file_path)
                self.btn_gp5.setText("Exported!")
            except Exception as e:
                print(e)
                self.btn_gp5.setText("Error")
                
    def handle_pdf(self):
        """
        Generates PDF formation of the tab
        """
        edited_text = self.editor.toPlainText()
        
        if not edited_text.strip():
            self.btn_save.setText("Generate a Tab.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save as PDF", "", "PDF Files (*.pdf)")
        
        if file_path:
            try:
                export_to_pdf(edited_text, file_path)
                self.btn_save.setText("Saved!")
            except Exception as e:
                print(e)
                self.btn_save.setText("Error")
                
                
    def colour_confidence(self):
        """
        Designed for the End-to-End system where, red: low, amber: med, green: high
        """
        if not self.position_map:
            return

        cursor = QTextCursor(self.editor.document())
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.setCharFormat(QTextCharFormat())

        for (row, slot, string_num), idx in self.position_map.items():
            note_event = self.mapped_notes[idx][0]
            position = self.mapped_notes[idx][1]
            fret = position[1]

            model = self.dropdown_model.currentText()
            if len(position) >= 3:
                confidence = position[2]
            elif model == "End-to-End Architecture (GOAT Dataset)" and len(note_event) > 3:
                confidence = float(note_event[3])
            else:
                continue
            
            if confidence >= 0.6:
                color = QColor("#00e676")
            elif confidence >= 0.35:
                color = QColor("#ffab00")  
            else:
                color = QColor("#ff1744")  

            lines_per_row = 8
            line_number = row * lines_per_row + (string_num - 1)

            PREFIX_WIDTH = 3
            SLOT_WIDTH = 3
            SLOTS_PER_BEAT = 4
            BEAT_BLOCK_WIDTH = SLOTS_PER_BEAT * SLOT_WIDTH + 2

            beat = slot // SLOTS_PER_BEAT
            slot_in_beat = slot % SLOTS_PER_BEAT
            col = PREFIX_WIDTH + beat * BEAT_BLOCK_WIDTH + slot_in_beat * SLOT_WIDTH

            # Find the block line and set cursor position
            block = self.editor.document().findBlockByLineNumber(line_number)
            if not block.isValid():
                continue

            pos = block.position() + col
            fret_str = str(fret).rjust(2)

            # Tab editing 
            cursor.setPosition(pos)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 2)

            # Apply colour
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            cursor.mergeCharFormat(fmt)
        
           
        

    def handle_result(self, result_text, mapped_notes, bpm): 
        """
        Runs when the worker thread finishes
        """
        self.bpm = bpm
        self.mapped_notes = mapped_notes
        # Re-enable button
        self.btn_transcribe.setText("TRANSCRIBE")
        self.btn_transcribe.setEnabled(True)

        self.top_spacer.setVisible(False)
        self.bottom_spacer.setVisible(False)
        self.editor.setVisible(True)
        self.midi_controls.setVisible(True)
        self.bottom_actions.setVisible(True)
        self.edit_actions.setVisible(True)
        
        # Set the text
        self.refresh_tab()
        self.load_midi_file()

    def synth_theme(self):
        """
        Synth theme
        """
        self.setStyleSheet("""
            /* BACKGROUND */
            QMainWindow {
                background-color: #000000;
            }
            QMainWindow > QWidget {
                background-color: #000000;
                color: #e0e0e0;
                font-family: 'Segoe UI', Roboto, Helvetica;
                font-size: 14px;
            }

            /* CONTROL PANEL */
            QFrame#ControlPanel {
                background-color: #241b35;
                border: 2px solid #4a148c;
                border-radius: 15px;
            }
            
            QSlider {
                background-color: transparent;
            }
            
            QLabel#PanelTitle {
                color: #d500f9;
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 2px;
                background-color: transparent;
                margin-bottom: 10px;
            }
            
            QPushButton#PlayButton {
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 2px;
                padding: 0px;
            }

            /* STANDARD BUTTONS */
            QPushButton {
                background-color: #3b1e52;
                border: 1px solid #7b1fa2;
                border-radius: 8px;
                padding: 10px 15px;
                color: #ffffff;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #4a148c;
                border-color: #d500f9;
            }

            /* UPLOAD BUTTON CHECKED STATE */
            QPushButton:checked { 
                background-color: #4a148c;
                border: 1px solid #00e5ff;
            }

            /* TRANSCRIBE BUTTON */
            QPushButton#BtnTranscribe {
                background-color: #aa00ff;
                border: none;
                font-weight: bold;
                font-size: 15px;
                letter-spacing: 1px;
            }
            QPushButton#BtnTranscribe:hover {
                background-color: #d500f9;
            }
            QPushButton#BtnTranscribe:pressed {
                background-color: #7b1fa2;
            }
            
            /* MODEL LABEL */
            QLabel {
                background-color: transparent;
                color: #e0e0e0;
            }

            /* DROPDOWN */
            QComboBox {
                background-color: #3b1e52;
                border: 1px solid #7b1fa2;
                border-radius: 8px;
                padding: 8px 12px;
                color: #ffffff;
            }
            QComboBox:hover {
                border-color: #d500f9;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #241b35;
                border: 1px solid #7b1fa2;
                color: #ffffff;
                selection-background-color: #4a148c;
            }

            QSlider::groove:horizontal {
                background-color: #3b1e52;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #d500f9;
                width: 12px;
                height: 12px;
                border-radius: 6px;
                margin: -4px 0;
            }
            QSlider::sub-page:horizontal {
                background-color: #7b1fa2;
                border-radius: 2px;
            }
            
            /* EDITOR */
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                background-color: #1a1025;
                border: 1px solid #4a148c;
                border-radius: 8px;
                padding: 8px 4px;
                color: #e040fb; /* Neon Pink Text */
                selection-background-color: #d500f9;
                selection-color: #ffffff;
            }

            /* DEMUCS TOGGLE OPTION */
            QPushButton#DemucsToggle {
                background-color: #1a1025;
                border: 2px solid #4a148c;
                border-radius: 12px;
                padding: 6px 16px;
                color: #888888;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
                min-width: 110px;
            }
            QPushButton#DemucsToggle:checked {
                background-color: #4a148c;
                border-color: #d500f9;
                color: #d500f9;
            }
            QPushButton#DemucsToggle:hover {
                border-color: #7b1fa2;
            }

            /* SECONDARY BUTTONS (Export) */
            QPushButton#SecondaryButton {
                background-color: transparent;
                border: 1px solid #7b1fa2;
            }
            QPushButton#SecondaryButton:hover {
                background-color: #3b1e52;
                border-color: #d500f9;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())