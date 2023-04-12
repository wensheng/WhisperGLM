import os
import sys
import traceback
import wave
import queue
import argparse
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import ffmpeg
import pythoncom
import pyaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel
from PySide6.QtCore import (
    QEvent, QObject, QRunnable, Qt, QThreadPool, Signal, Slot,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QSpacerItem,
    QTextEdit,
    QSizePolicy,
)
from PySide6.QtGui import QFont

from whisper_audio import load_audio, get_duration, SAMPLE_RATE
for m in ['whisper', 'yt-dlp']:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), m)))
import yt_dlp  # noqa: E402
import whisper  # noqa: E402

FRAMES_PER_BUFFER = 1024
CAPTURE_WINDOW = 15   # every 15 seconds, process captured audio
WIN_INIT_SIZE = (720, 1080)
MAX_LENGTH = 2048
TOP_P = 0.7
TEMPERATURE = 0.9
DL_AUDIO_FILE = os.path.join('data', 'temp.m4a')

model, tokenizer = None, None

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--noglm', default=False,
                    action='store_true', help='will not load ChatGLM model')


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            print("worker finished")
            self.signals.finished.emit()


class ListeningWorker(QRunnable):
    # data_ready = Signal(bytes)
    def __init__(self, buffer, device_info, stop_event):
        super().__init__()
        self.buffer = buffer
        self.device_info = device_info
        self.stop_event = stop_event
        print("listening worker initialized")
        self.signals = WorkerSignals()

    def run(self):
        # Initialize the Core Audio API
        print('Initialize Listener')
        pythoncom.CoInitialize()
        p = pyaudio.PyAudio()

        while True:
            if self.stop_event.is_set():
                break
            buffer = BytesIO()
            # Open a wave buffer for writing
            wf = wave.open(buffer, 'wb')
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)

            stream = p.open(format=pyaudio.paInt16,
                            channels=self.device_info[1],
                            rate=44100,
                            input=True,
                            input_device_index=self.device_info[0],
                            frames_per_buffer=FRAMES_PER_BUFFER)

            # Record audio for the specified duration
            num_frames = int(44100 * CAPTURE_WINDOW / FRAMES_PER_BUFFER)
            for i in range(0, num_frames):
                data = stream.read(FRAMES_PER_BUFFER)
                wf.writeframes(data)
                pythoncom.PumpWaitingMessages()

            # Close the wave file and audio stream
            stream.stop_stream()
            stream.close()
            wf.close()
            print("finished recording chunk")
            self.buffer.put(buffer.getvalue())

        p.terminate()
        # Uninitialize the Core Audio API
        pythoncom.CoUninitialize()
        self.signals.finished.emit()


class TranscribeWorker(QRunnable):
    processed = Signal(str)

    def __init__(self, buffer, text_box, stop_event, noload=False):
        super().__init__()
        print("transcribe worker initialized")
        self.buffer = buffer
        self.text_box = text_box
        self.stop_event = stop_event
        self.noload = noload
        self.signals = WorkerSignals()
        self.model = whisper.load_model("base")
        print("model loaded")

    def run(self):
        while True:
            if self.stop_event.is_set():
                break
            try:
                data = self.buffer.get(block=True, timeout=1)
                print("got buffer")
                if not self.noload:
                    audio = load_audio(data, sr=16000)
                else:
                    audio = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
                _, probs = self.model.detect_language(mel)
                options = whisper.DecodingOptions()
                result = whisper.decode(self.model, mel, options)
                print(result.text)
                self.text_box.append(result.text)
            except queue.Empty:
                pass
        self.signals.finished.emit()


class SegmentWorker(QRunnable):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.signals = WorkerSignals()

    def run(self):
        duration = get_duration(DL_AUDIO_FILE)
        try:
            duration = float(duration)
        except ValueError:
            return
        print("duration", duration)
        ss = 0
        while ss < duration:
            audio_input = ffmpeg.input(DL_AUDIO_FILE).audio
            audio_cut = audio_input.filter('atrim', start=ss, duration=25)
            audio_output = ffmpeg.output(audio_cut, 'pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
            out, _ = ffmpeg.run(audio_output, capture_stdout=True, capture_stderr=True)
            self.buffer.put(out)
            ss += 25 - 0.2
        self.signals.finished.emit()
            

class ChatWorker(QRunnable):
    def __init__(self, buffer, state):
        super().__init__()
        self.buffer = buffer
        self.state = state
        self.signals = WorkerSignals()

    def run(self):
        while True:
            try:
                data = self.buffer.get(block=True, timeout=1)
                print("got buffer")
                audio = load_audio(data, sr=16000)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
                _, probs = self.model.detect_language(mel)
                options = whisper.DecodingOptions()
                result = whisper.decode(self.model, mel, options)
                print(result.text)
                self.text_box.append(result.text)
            except queue.Empty:
                pass


class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.listen_worker = None
        self.segment_worker = None
        self.transcribe_worker = None
        self.buffer = queue.Queue()
        self.state = {'listening': False,
                      'transcribing': False,
                      'downloading': False,
                      'chatting': False}
        self.event_stop_listen = threading.Event()
        self.event_stop_transcribe = threading.Event()
        self.history = []

        layout = QVBoxLayout()

        # radio button group to select download or listen
        rb_layout = QHBoxLayout()
        listen_button = QRadioButton("Listen")
        listen_button.setChecked(True)
        youtube_button = QRadioButton("YouTube")
        radio_button_group = QButtonGroup()
        radio_button_group.setExclusive(True)
        radio_button_group.addButton(youtube_button)
        radio_button_group.addButton(listen_button)
        listen_button.toggled.connect(self.on_listen_button_toggle)

        rb_layout.addWidget(listen_button)
        rb_layout.addWidget(youtube_button)
        layout.addLayout(rb_layout)

        # create a input box with label for URL
        url_label = QLabel("URL:")
        self.url_edit = QLineEdit()
        self.url_edit.setEnabled(False)
        url_layout = QHBoxLayout()
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_edit)
        layout.addLayout(url_layout)

        # buttons to start download or listen
        self.start_youtube_button = QPushButton("Download Audio")
        self.start_youtube_button.setVisible(False)
        self.start_youtube_button.pressed.connect(self.start_yt_download)
        self.start_stop_listen_button = QPushButton("Start Listening")
        self.start_stop_listen_button.pressed.connect(self.start_listening)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setDisabled(True)
        self.stop_button.pressed.connect(self.stop_listening)
        btn_layout = QHBoxLayout()
        spacer1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.combo_box = QComboBox()
        self.set_audio_device_selections()
        btn_layout.addWidget(self.combo_box)
        btn_layout.addItem(spacer1)
        btn_layout.addWidget(self.start_stop_listen_button)
        btn_layout.addWidget(self.start_youtube_button)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addItem(spacer1)
        clr_button = QPushButton("Clear")
        clr_button.pressed.connect(self.clear_text)
        btn_layout.addWidget(clr_button)
        layout.addLayout(btn_layout)

        # Create the text box
        self.text_box = QTextEdit()
        font = QFont()
        font.setPointSize(14)
        self.text_box.setFont(font)
        layout.addWidget(self.text_box)

        btn_layout2 = QHBoxLayout()
        chat_button = QPushButton("Chat")
        #chat_button.pressed.connect(self.start_chat)
        btn_layout2.addWidget(chat_button)
        btn_layout2.addItem(spacer1)
        summary_button = QPushButton("Summarize")
        summary_button.pressed.connect(self.summarize)
        btn_layout2.addWidget(summary_button)
        btn_layout2.addItem(spacer1)
        layout.addLayout(btn_layout2)

        self.summary_box = QTextEdit()
        self.summary_box.setMaximumHeight(300)
        self.summary_box.setFont(font)
        self.summary_box.setVisible(False)
        layout.addWidget(self.summary_box)
        
        self.chat_box = QTextEdit()
        self.chat_box.setMaximumHeight(60)
        self.chat_box.setFont(font)
        self.chat_box.installEventFilter(self)
        layout.addWidget(self.chat_box)

        # Create the layout for the dialog

        status_title_label = QLabel("Status")
        self.status_label = QLabel("")
        status_layout = QHBoxLayout()
        status_layout.addWidget(status_title_label)
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)

        # Set the layout for the dialog
        self.setLayout(layout)
        self.resize(WIN_INIT_SIZE[0], WIN_INIT_SIZE[1])
        # self.threads = {}
        self.threadpool = QThreadPool()
        print("Pool created with maximum %d threads" % self.threadpool.maxThreadCount())
        
    def set_audio_device_selections(self) -> None:
        """
        In my PC, I have 34 devices
        We pick first 15 devices and select those that starts with "Microphone", "Stereo Mix"
        """
        p = pyaudio.PyAudio()
        c = p.get_device_count()
        n = 15 if c > 15 else c
        for i in range(n):
            dev = p.get_device_info_by_index(i)
            if (dev['name'].startswith(('Microphone', 'Stereo Mix', 'Microsoft Sound Mapper',))
                and dev['maxInputChannels'] > 0):
                # Can not use tuple because of pyqt bug, had to use list
                self.combo_box.addItem(dev['name'], [dev['index'], dev['maxInputChannels']])
        p.terminate()
        idx = self.combo_box.findData([2, 2])  # default to Stereo Mix
        if idx:
            self.combo_box.setCurrentIndex(idx)

    def eventFilter(self, obj, event):
        if obj is self.chat_box and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and self.chat_box.hasFocus():
                self.send_chat(self.chat_box.toPlainText())
                self.text_box.append(f'Me: {self.chat_box.toPlainText()}\n')
                self.chat_box.setText('')
                return True
        return False
    
    @Slot()
    def clear_text(self):
        self.text_box.setText('')

    @Slot()
    def on_listen_button_toggle(self, is_listen):
        if is_listen:
            self.url_edit.setEnabled(False)
            self.start_youtube_button.setVisible(False)
            self.start_stop_listen_button.setVisible(True)
            self.stop_button.setVisible(True)
            self.event_stop_listen.set()
        else:
            self.url_edit.setEnabled(True)
            self.start_youtube_button.setVisible(True)
            self.start_stop_listen_button.setVisible(False)
            self.stop_button.setVisible(False)

    def start_yt_download(self):
        self.status_label.setText("Downloading youtube audio...")
        worker = Worker(self.yt_download, self.url_edit.text())
        worker.signals.finished.connect(self.start_transcribe_yt)
        self.threadpool.start(worker)

    def yt_download(self, url):
        """
        yt_dlp.main(argv) is doing some weird stuff and can not be used
        """
        yt_dlp._real_main(argv=[url, '--no-update', '--force-overwrites',
                          '-q', '-f', '139', '-o', DL_AUDIO_FILE]),
        
    @Slot()
    def start_transcribe_yt(self):
        print("Starting transcribe YT")
        # self.state['downloading'] = True
        self.event_stop_transcribe.clear()
        self.segment_worker = SegmentWorker(self.buffer)
        self.segment_worker.signals.finished.connect(self.on_segmentation_finished)
        self.threadpool.start(self.segment_worker)
        self.transcribe_worker = TranscribeWorker(self.buffer, self.text_box,
                                                  self.event_stop_transcribe, noload=True)
        self.threadpool.start(self.transcribe_worker)
        self.status_label.setText("Started transcription!")
        
    @Slot()
    def on_segmentation_finished(self):
        self.event_stop_transcribe.set()

    @Slot()
    def start_listening(self):
        """
        The listening thread will continuously run until user presses the stop button.
        The transcribe thread will continuously run until all audio has been transcribed.
        """
        if not self.start_stop_listen_button.isEnabled():
            return

        if self.state['listening']:
            self.event_stop_listen.set()
            self.status_label.setText("Stop Listening requested")
            self.start_stop_listen_button.setText("Please wait...")
            self.start_stop_listen_button.setEnabled(False)
            self.state['listening'] = False
        else:
            self.state['listening'] = True
            self.event_stop_listen.clear()
            self.event_stop_transcribe.clear()
            self.start_stop_listen_button.setText("Stop Listening")
            device_info = self.combo_box.currentData()
            self.listen_worker = ListeningWorker(self.buffer, device_info, self.event_stop_listen)
            self.listen_worker.signals.finished.connect(self.on_listening_stopped)
            self.threadpool.start(self.listen_worker)
            self.transcribe_worker = TranscribeWorker(self.buffer, self.text_box, self.event_stop_transcribe)
            self.threadpool.start(self.transcribe_worker)
            
    @Slot()
    def on_listening_stopped(self):
        self.start_stop_listen_button.setText("Start Listening")
        self.start_stop_listen_button.setEnabled(True)

    def send_chat(self, prompt):
        # TODO: will use one long-running worker
        worker = Worker(self.process_chat, prompt, self.history)
        worker.signals.result.connect(self.handle_chat_res)
        self.threadpool.start(worker)
        
    @Slot()
    def handle_chat_res(self, res):
        self.history = res['history']
        self.text_box.append(f'ChatGLM: {res["response"]}\n')

    def process_chat(self, prompt, history):
        global tokenizer, model
        if not model:
            return {'response': 'Model not loaded', 'history': history}
        response, history = model.chat(
            tokenizer, prompt, history=history,
            max_length=MAX_LENGTH, top_p=TOP_P,
            temperature=TEMPERATURE)
        return {'response': response, 'history': self.history}

    @Slot()
    def summarize(self):
        pass

    @Slot()
    def stop_listening(self):
        self.state['listening'] = False


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.noglm:
        try:
            model_path = os.path.join('data', 'chatglm-6b-int4')
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
            model = model.eval()
        except Exception as e:
            print("Error loading ChatGLM model", e)
    else:
        print("Will not load ChatGLM model")
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.exec()
    dialog.event_stop_listen.set()
    dialog.event_stop_transcribe.set()
