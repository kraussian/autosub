# Import modules
print("Importing modules")
import os
import ffmpeg
import whisper
import argparse
import random
import numpy as np
import pydub
from   collections import Counter
from   typing import Iterator, TextIO

# HACK: Fix torch FutureWarning on weights_only while loading pickle data
import functools
whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)

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

def play_chunk_audio(chunk_file):
    # Convert audio chunk to AudioSegment and play it
    chunk_audio_segment = pydub.AudioSegment(
        (chunk_file * 32767).astype(np.int16).tobytes(),  # Convert to 16-bit PCM
        frame_rate=whisper.audio.SAMPLE_RATE,
        sample_width=2,  # 16-bit = 2 bytes
        channels=1       # Mono audio
    )
    print(f"Playing chunk {i+1}...")
    pydub.playback.play(chunk_audio_segment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='autosub.py',
        description='AutoSub automatically extracts subtitles from video using OpenAI Whisper',
    )
    parser.add_argument('filename', help="Path and name of the video file to extract from")
    parser.add_argument('-l', '--language', help="Override language of video file, e.g. en, ja, ko, zh")
    parser.add_argument('-c', '--chunks', help="Override number of random chunks to use for detecting language")
    parser.add_argument('-t', '--translate', help="Automatically translate subtitles to English", action='store_true')
    args = parser.parse_args()

    filename = args.filename
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
    model = whisper.load_model(model_name, device="cuda")

    # Load the audio file
    audio = whisper.load_audio(audio_file)

    # Set default Whisper options
    options = {"task": "transcribe"}

    # Automatically detect language if not overridden
    if args.language:
        options['language'] = args.language
    else:
        # Chunk size and total number of samples
        chunk_size = 30 * whisper.audio.SAMPLE_RATE  # 30 seconds in samples
        total_samples = len(audio)

        # Number of random chunks to sample
        if args.chunks:
            num_random_chunks = args.chunks
        else:
            num_random_chunks = 10
        print(f"Total audio length (samples): {total_samples}")
        print(f"Sampling {num_random_chunks} random chunks...")

        # Generate random start points for chunks
        random_starts = [
            random.randint(0, max(0, total_samples - chunk_size)) for _ in range(num_random_chunks)
        ]

        # Detect language for each chunk
        language_counts = Counter()
        for i, start in enumerate(random_starts):
            end = start + chunk_size
            chunk = audio[start:end]
            chunk = whisper.pad_or_trim(chunk)  # Ensure 30 seconds (or padded if shorter)

            # Generate log-mel spectrogram
            log_mel = whisper.log_mel_spectrogram(chunk, n_mels=128).to("cuda")
            log_mel = log_mel.unsqueeze(0)  # Add batch dimension

            # Detect language
            print(f"    Detecting language for random chunk {i+1}/{num_random_chunks}...")
            _, probs = model.detect_language(log_mel)

            # Extract language probabilities
            if isinstance(probs, list) and len(probs) == 1 and isinstance(probs[0], dict):
                probs = probs[0]

            # Find the most likely language for this chunk
            detected_language = max(probs, key=probs.get)
            print(f"    Chunk {i+1}: Detected language = {detected_language}")

            # Play chunk audio to manually verify if the detection is correct
            #play_chunk_audio(chunk)

            # Update language counts
            language_counts[detected_language] += 1

        # Aggregate results
        most_likely_language = language_counts.most_common(1)[0][0]
        print(f"Overall most likely language: {most_likely_language}")

        options['language'] = most_likely_language

        # Test on short sample
        """
        short_options = {"task": "transcribe", "language": most_likely_language}
        short_result = model.transcribe(
            language_detection_audio,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=False,
            **short_options
        )
        short_result.get("text")
        """

    if args.translate:
        options['task'] = 'translate'
    print("Extracting subtitles")
    result = model.transcribe(
        audio_file,
        fp16=False,
        verbose=True,
        temperature=0.0,
        condition_on_previous_text=False,
        **options
    )

    srt_file = outfile = '.'.join(os.path.basename(filename).split('.')[:-1]) + '.srt'
    with open(srt_file, "w", encoding="utf-8") as srt:
        write_srt(result["segments"], file=srt)

# NOTE: ffmpeg needs to be installed on the system first
# On Windows, install with: choco install ffmpeg

# NOTE: torch needs to be in installed
# pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu124
