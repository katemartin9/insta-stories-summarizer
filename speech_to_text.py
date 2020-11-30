import moviepy.editor as mp
import speech_recognition as sr
from files_ops import create_directory


def save_to_wav(file, i):
    video = mp.VideoFileClip(file)
    create_directory('audio_output')
    out_file = f"audio_output/story{i}.wav"
    video.audio.write_audiofile(out_file)
    return out_file


def speech_to_text(file, i):
    try:
        out_file = save_to_wav(file, i)
    except AttributeError as e:
        return None
    sample_audio = sr.AudioFile(out_file)
    recog = sr.Recognizer()
    with sample_audio as audio_file:
        audio_content = recog.record(audio_file)
        # language codes https://cloud.google.com/speech-to-text/docs/languages
        # TODO: detect language
    try:
        text = recog.recognize_google(audio_content, language="en-US")
        # text = recog.recognize_google(audio_content, language="ru-RU")
    except Exception as e:
        return None
    return text
