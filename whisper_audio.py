"""
load_audio is a modified version of the original from the OpenAI repository
"""
import ffmpeg
import numpy as np


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000


def load_audio(buffer: bytes, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input('pipe:')
            .output('pipe:', format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(input=buffer, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_duration(file: str) -> float:
    """
    Get the duration of an audio file in seconds
    """
    data = ffmpeg.probe(file)
    if 'streams' not in data:
        return 0

    if not data['streams'][0] or 'duration' not in data['streams'][0]:
        return 0

    return data['streams'][0]['duration']
