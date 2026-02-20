import sys
import os


from PySide6.QtCore import Qt, QThread, Signal, QObject, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QPlainTextEdit, 
                               QFileDialog, QFrame, QSizePolicy, QComboBox, QSlider)
from PySide6.QtGui import QFont

current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)              
sys.path.append(root_dir)                          

from backend.export.exporter import *
from backend.export.midi_exporter import *
from backend.export.exporter import export_to_gp5
from backend.export.tab_pdf import export_to_pdf
from backend.main import automatic_music_transcription

class TabEditor(QPlainTextEdit):
    """
    Custom editor for guitar tabs.
    """
    def __init__(self):
        super().__init__()
        font = QFont("Consolas", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)


class TranscriptionWorker(QThread):
    finished = Signal(str)
    
    def __init__(self, file_path, selected_model):
        super().__init__()
        self.file_path = file_path
        self.selected_model = selected_model
    
    def run(self):

        try:
            # Call backend here
            #result = automatic_music_transcription(self.file_path, 0)
            result = automatic_music_transcription(self.file_path, None, self.selected_model)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Guitar Practice Tool")
        self.resize(1000, 700)
        self.file_name = None

        # --- MAIN LAYOUT SETUP ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # We use a vertical layout for the whole page
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(40, 40, 40, 40)

        # --- SPACER ---
        # Pushes everything to the center initially
        self.top_spacer = QWidget()
        self.top_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.top_spacer)
        
        # --- MODEL SELECTION ---
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
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(30, 30, 30, 30)
        

        # --- AUDIO SETUP ---
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
        
        lbl_original = QLabel("Original Audio")
        lbl_original.setObjectName("PlayerLabel")
        
        lbl_midi = QLabel("Midi Tab Audio")
        lbl_midi.setObjectName("PlayerLabel")
        
        # Upload button
        self.btn_upload = QPushButton("Upload Audio File")
        self.btn_upload.setMinimumHeight(50)
        self.btn_upload.clicked.connect(self.upload_file)
        control_layout.addWidget(self.btn_upload)
        
        self.audio_controls = QWidget()
        audio_inner = QHBoxLayout(self.audio_controls)
        audio_inner.setContentsMargins(0, 0, 0, 0)
        audio_inner.addWidget(lbl_original)
        audio_inner.addWidget(self.btn_original_audio)
        audio_inner.addWidget(self.seek_slider_original)
        audio_inner.addWidget(self.lbl_time_original)
        self.audio_controls.setVisible(False)
        control_layout.addWidget(self.audio_controls)
        
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
        self.btn_transcribe.setMinimumHeight(50)
        self.btn_transcribe.setCursor(Qt.PointingHandCursor)
        self.btn_transcribe.clicked.connect(self.run_transcription)
        control_layout.addWidget(self.btn_transcribe)

        self.main_layout.addWidget(self.control_panel)

        # Editor section
        self.editor = TabEditor()
        self.editor.setPlaceholderText("Generated tabs will appear here...")
        self.editor.setVisible(False) 
        # Stretch uses all the available space
        self.main_layout.addWidget(self.editor, stretch=1)

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

    def connection(self):
        # Connects the export button
        self.btn_gp5.clicked.connect(self.handle_export)
        self.btn_save.clicked.connect(self.handle_pdf)

    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Add Audio File", "", "(*.mp3 *.wav)")
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
            
    def update_slider_midi(self, position):
        self.slider_midi.setValue(position)
        total = self.player_midi.duration()
            
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
        
        self.worker = TranscriptionWorker(self.file_name, self.dropdown_model.currentText())
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
           
        

    def handle_result(self, result_text): 
        """
        Runs when the worker thread finishes
        """
        # Re-enable button
        self.btn_transcribe.setText("TRANSCRIBE")
        self.btn_transcribe.setEnabled(True)

        self.top_spacer.setVisible(False)
        self.bottom_spacer.setVisible(False)
        self.editor.setVisible(True)
        self.midi_controls.setVisible(True)
        self.bottom_actions.setVisible(True)
        
        # Set the text
        self.editor.setPlainText(result_text)
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
            QPlainTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                background-color: #1a1025;
                border: 1px solid #4a148c;
                border-radius: 8px;
                padding: 15px;
                color: #e040fb; /* Neon Pink Text */
                selection-background-color: #d500f9;
                selection-color: #ffffff;
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