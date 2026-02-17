import sys
import os


from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QPlainTextEdit, 
                               QFileDialog, QFrame, QSizePolicy)
from PySide6.QtGui import QFont

current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)              
sys.path.append(root_dir)                          


from backend.export.exporter import export_to_gp5
from backend.export.tab_pdf import export_to_pdf
from backend.main import automatic_music_transcription, goat_music_transcription

class TabEditor(QPlainTextEdit):
    """
    Custom editor for guitar tabs.
    """
    def __init__(self):
        super().__init__()
        font = QFont("Consolas", 12)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)

class TranscriptionWorker(QThread):
    finished = Signal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            # Call backend here
            #result = automatic_music_transcription(self.file_path, 0)
            result = goat_music_transcription(self.file_path, None)
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

        # 1. --- SPACER (Top) ---
        # Pushes everything to the center initially
        self.top_spacer = QWidget()
        self.top_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.top_spacer)

        # Control panel
        self.control_panel = QFrame()
        self.control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(self.control_panel) 
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(30, 30, 30, 30)

        # Header
        lbl_title = QLabel("TRANSCRIBE AUDIO")
        lbl_title.setObjectName("PanelTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(lbl_title)

        # Upload button
        self.btn_upload = QPushButton("Upload Audio File")
        self.btn_upload.setMinimumHeight(50)
        self.btn_upload.clicked.connect(self.upload_file)
        control_layout.addWidget(self.btn_upload)

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
            # Reset view 
            # self.reset_view()

    def run_transcription(self):
        """
        Simulates the transcription process.
        In the real app, this would be called after the ML model finishes.
        """
        if not self.file_name:
            self.btn_upload.setText("Please add a Audio File first.")
            return
        
        self.worker = TranscriptionWorker(self.file_name)
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
        self.bottom_actions.setVisible(True)
        
        # Set the text
        self.editor.setPlainText(result_text)

    def synth_theme(self):
        """
        Synth theme
        """
        self.setStyleSheet("""
            /* BACKGROUND */
            QMainWindow {
                background-color: #000000;
            }
            QWidget {
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
            QLabel#PanelTitle {
                color: #d500f9;
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 2px;
                background-color: transparent;
                margin-bottom: 10px;
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
                box-shadow: 0px 0px 15px #d500f9;
            }
            QPushButton#BtnTranscribe:pressed {
                background-color: #7b1fa2;
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