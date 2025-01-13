# Import modules
print("Importing modules")
import os
import re
import ffmpeg
from   faster_whisper import WhisperModel
import argparse
import pytubefix  # Install with: pip install -U pytubefix
from   pytubefix.cli import on_progress
from   typing import Iterator, TextIO

# Function: Extract audio from video file and save to .WAV
def extract_audio(filename, acodec="pcm_s16le", ac=1, ar="16k"):
    outfile = '.'.join(os.path.basename(filename).split('.')[:-1]) + '.wav'
    print(f"Extracting audio as: {outfile} (codec: {acodec}, {ac} channels @{ar}Hz)")
    ffmpeg.input(filename).output(
        outfile,
        acodec=acodec, ac=ac, ar=ar
    ).run(quiet=True, overwrite_output=True)
    return outfile

# Function: Format timestamp for SRT
def format_timestamp(seconds:float, always_include_hours:bool=False) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Function: Write SRT subtitle
def write_srt(transcript:Iterator[dict], file:TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

# Function: Remove repetitive strings
def remove_repetitions_and_sequences(input_string: str, char_repeats: int=6, seq_repeats: int=3) -> str:
    # Remove invalid characters
    input_string = input_string.replace("�", "")

    # Handle repetitive multi-character sequences
    sequence_pattern = rf'((.+?)\2{{{seq_repeats},}})'
    result = re.sub(sequence_pattern, lambda m: m.group(2) * seq_repeats, input_string)
    
    # Handle single-character repetitions
    single_char_pattern = rf'(.)\1{{{char_repeats},}}'
    result = re.sub(single_char_pattern, lambda m: m.group(1) * char_repeats, result)
    
    return result

def cleanup_text(input_dict:dict) -> dict:
    input_string = input_dict.get("text")
    output_string = remove_repetitions_and_sequences(input_string)
    input_dict["text"] = output_string
    return input_dict

def remove_consecutive_duplicates(lst):
    if not lst:  # Handle empty list
        return []

    result = [lst[0]]  # Initialize with the first element
    for i in range(1, len(lst)):
        if lst[i].get("text") != lst[i - 1].get("text"):  # Compare with the previous element
            result.append(lst[i])
    return result

def shorten_long_duration(input_dict:dict) -> dict:
    length_duration = round(float(input_dict.get("end")) - float(input_dict.get("start")), 2)
    length_text = len(input_dict.get("text"))
    #print(f"Text: {input_dict.get("text")} ({length_text}), Duration: {length_duration}")
    default_length = 10
    if length_duration > default_length:
        input_dict["end"] = float(input_dict.get("start")) + default_length
        print(f"Shortened length: {input_dict.get("start")} --> {input_dict.get("end")} ({length_duration}) {input_dict.get("text")}")
    return input_dict

def get_youtube_video(url:str) -> str:
    # Barack Obama 2004 DNC Speech: https://www.youtube.com/watch?v=ueMNqdB1QIE
    # Teddy Roosevelt 1912 Speech: https://www.youtube.com/watch?v=uhlzdjPGxrs
    if "youtube" in url.replace(".",""):
        yt = pytubefix.YouTube(url)#, on_progress_callback=on_progress)
        print(f"Downloading Youtube video: {yt.title}")

        ys = yt.streams.get_highest_resolution()
        outfile = f"{yt.title}.mp4"
        filepath = ys.download(filename=outfile)
    else:
        print(f"URL {url} is not a valid YouTube video")
        outfile = ""
    return outfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='autosub.py',
        description='AutoSub automatically extracts subtitles from video or audio files using OpenAI Whisper',
    )
    parser.add_argument('filename', help="Path and name of the video file to extract from, or URL of YouTube video")
    parser.add_argument('-l', '--language', help="Override language of video file, e.g. en, ja, ko, zh")
    parser.add_argument('-c', '--chunks', help="Override number of random chunks to use for detecting language")
    parser.add_argument('-t', '--translate', help="Automatically translate subtitles to English", action='store_true')
    args = parser.parse_args()

    filename = args.filename
    if "http" in filename:
        print("Attempting to download YouTube video")
        filename = get_youtube_video(filename)
    else:
        if not os.path.exists(filename):
            print("ERROR: File not found")
            exit()
        else:
            print(f"Processing video: {os.path.basename(filename)}")

    audio_file = extract_audio(filename)
    # NOTE: Available Whisper models:
    # tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo
    model_name = 'large-v2'
    print(f"Loading Whisper model: {model_name}")
    # Load Whisper model to run on GPU with FP16
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

    # Set default Whisper options
    options = {"task": "transcribe"}

    # Automatically detect language if not overridden
    if args.language:
        options['language'] = args.language

    if args.translate:
        options['task'] = 'translate'

    """
    import silero_vad
    vad_model = silero_vad.load_silero_vad()
    wav = silero_vad.read_audio(audio_file)
    # threshold (default float=0.5): Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
    # probabilities ABOVE this value are considered as SPEECH. It is better to tune this
    # parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
    # neg_threshold (default float=None): Silence threshold for determining the end of speech. If a probability is lower
    # than neg_threshold, it is always considered silence. Values higher than neg_threshold
    # are only considered speech if the previous sample was classified as speech; otherwise,
    # they are treated as silence. This parameter helps refine the detection of speech
    # transitions, ensuring smoother segment boundaries.
    # min_speech_duration_ms (default int=0): Final speech chunks shorter min_speech_duration_ms are thrown out.
    # max_speech_duration_s (default float=float("inf")): Maximum duration of speech chunks in seconds. Chunks longer
    # than max_speech_duration_s will be split at the timestamp of the last silence that
    # lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
    # split aggressively just before max_speech_duration_s.
    # min_silence_duration_ms (default int=2000): In the end of each speech chunk wait for min_silence_duration_ms before separating it
    # speech_pad_ms (default int=400): Final speech chunks are padded by speech_pad_ms each side
    speech_timestamps = silero_vad.get_speech_timestamps(
        audio=wav,
        model=vad_model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        threshold=0.5,
        neg_threshold=0.3,
    )
    """

    print("Extracting subtitles")
    segments, info = model.transcribe(
        audio=audio_file,
        language=options.get("language"),
        task=options.get("task"),
        beam_size=10,
        log_progress=False,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,  # The library integrates the Silero VAD model to filter out parts of the audio without speech
        vad_parameters=dict(threshold=0.05, neg_threshold=0.01, min_silence_duration_ms=2000),  # Customize VAD parameters
    )
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # NOTE: segments is a generator so the transcription only starts when you iterate over it
    # The transcription can be run to completion by gathering the segments in a list or a for loop
    # segments = list(segments)  # The transcription will actually run here.
    list_transcribe = []
    for segment in segments:
        dict_segment = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        }
        list_transcribe.append(dict_segment)
        segment_info = f"{dict_segment.get("start")} --> {dict_segment.get("end")} {dict_segment.get("text")}"
        print(segment_info)
    print(f"Transcribed {len(list_transcribe)} segments")

    # Clean up transcribed text to remove repetitions and shorten long durations
    list_transcribe = [cleanup_text(item) for item in list_transcribe]
    list_transcribe = remove_consecutive_duplicates(list_transcribe)
    list_transcribe = [shorten_long_duration(item) for item in list_transcribe]
    print(f"Cleaned up to {len(list_transcribe)} segments")

    srt_file = outfile = '.'.join(os.path.basename(filename).split('.')[:-1]) + '.srt'
    with open(srt_file, "w", encoding="utf-8") as srt:
        write_srt(list_transcribe, file=srt)
    print(f"Wrote subtitle file to: {srt_file}")

# NOTE: ffmpeg needs to be installed on the system first
# On Windows, install with: choco install ffmpeg

# NOTE: Install required modules with:
# pip install -U openai-whisper ffmpeg-python
# pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124