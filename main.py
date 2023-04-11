import os
import sys
import traceback
import wave
import queue
import argparse
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
CAPTURE_WINDOW = 20   # every 20 seconds, process captured audio
WIN_INIT_SIZE = (720, 1080)
MAX_LENGTH = 2048
TOP_P = 0.7
TEMPERATURE = 0.9
DL_AUDIO_FILE = os.path.join('data', 'temp.m4a')

model, tokenizer = None, None

parser = argparse.ArgumentParser()
parser.add_argument('--noglm', type=bool, default=False, help='will not load ChatGLM model')


class TaskManager(QObject):
    """
    Not Used
    """
    finished = Signal(object)

    def __init__(self, parent=None, max_workers=None):
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def executor(self):
        return self._executor

    def submit(self, fn, *args, **kwargs):
        future = self.executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self._internal_done_callback)

    def _internal_done_callback(self, future):
        data = future.result()
        self.finished.emit(data)
 

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    '''
    finished = Signal(str)
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        if self.kwargs.get('finished'):
            self.finished = self.kwargs.get('finished')
            del self.kwargs['finished']
        else:
            self.finished = ''
        assert isinstance(self.finished, str)
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
            self.signals.finished.emit(self.finished)


class ListeningWorker(QRunnable):
    # data_ready = Signal(bytes)
    def __init__(self, buffer, state):
        super().__init__()
        self.buffer = buffer
        self.state = state
        print("listening worker initialized")
        self.signals = WorkerSignals()

    def run(self):
        # Initialize the Core Audio API
        print('Initialize Listener')
        pythoncom.CoInitialize()
        p = pyaudio.PyAudio()

        while True:
            if self.state['listening'] is False:
                break
            buffer = BytesIO()
            # Open a wave buffer for writing
            wf = wave.open(buffer, 'wb')
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)

            stream = p.open(format=pyaudio.paInt16,
                            channels=2,
                            rate=44100,
                            input=True,
                            input_device_index=2,
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


class TranscribeWorker(QRunnable):
    processed = Signal(str)

    def __init__(self, buffer, text_box, state, noload=False):
        super().__init__()
        print("transcribe worker initialized")
        self.buffer = buffer
        self.text_box = text_box
        self.state = state
        self.noload = noload
        self.model = whisper.load_model("base")
        print("model loaded")

    def run(self):
        while True:
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


class SegmentWorker(QRunnable):
    def __init__(self, buffer, state):
        super().__init__()
        self.buffer = buffer
        self.state = state

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
        self.start_listen_button = QPushButton("Start Listening")
        self.start_listen_button.pressed.connect(self.start_listening)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setDisabled(True)
        self.stop_button.pressed.connect(self.stop_listening)
        btn_layout = QHBoxLayout()
        spacer1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        combo_box = QComboBox()
        combo_box.addItems(["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese"])
        btn_layout.addWidget(combo_box)
        btn_layout.addItem(spacer1)
        btn_layout.addWidget(self.start_listen_button)
        btn_layout.addWidget(self.start_youtube_button)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addItem(spacer1)
        clr_button = QPushButton("Clear")
        clr_button.pressed.connect(self.clear_text)
        btn_layout.addWidget(clr_button)
        layout.addLayout(btn_layout)
        # button_group2 = QButtonGroup()
        # button_group2.addButton(self.start_youtube_button)
        # button_group2.addButton(self.start_listen_button)
        # button_group2.addButton(self.stop_button)
        # layout.addWidget(self.start_listen_button)
        # layout.addWidget(self.start_youtube_button)
        # layout.addWidget(self.stop_button)

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

    def eventFilter(self, obj, event):
        if obj is self.chat_box and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and self.chat_box.hasFocus():
                print("enter pressed")
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
            self.start_listen_button.setVisible(True)
            self.stop_button.setVisible(True)
        else:
            self.url_edit.setEnabled(True)
            self.start_youtube_button.setVisible(True)
            self.start_listen_button.setVisible(False)
            self.stop_button.setVisible(False)

    def start_yt_download(self):
        self.status_label.setText("Downloading youtube audio...")
        worker = Worker(self.yt_download, self.url_edit.text(),
                        finished='Download finished!')
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
        self.segment_worker = SegmentWorker(self.buffer, self.state)
        self.threadpool.start(self.segment_worker)
        if not self.transcribe_worker:
            self.transcribe_worker = TranscribeWorker(self.buffer, self.text_box, self.state, noload=True)
            self.threadpool.start(self.transcribe_worker)
        self.status_label.setText("Started transcription!")

    @Slot()
    def start_listening(self):
        """
        The listening thread will continuously run until user presses the stop button.
        The transcribe thread will continuously run until all audio has been transcribed.
        """
        self.state['listening'] = True
        self.listen_worker = ListeningWorker(self.buffer, self.state)
        self.threadpool.start(self.listen_worker)
        self.transcribe_worker = TranscribeWorker(self.buffer, self.text_box, self.state)
        self.threadpool.start(self.transcribe_worker)

    def send_chat(self, prompt):
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
