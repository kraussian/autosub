import os
import re
import ffmpeg  # Install with: pip install -U ffmpeg-python
import pytubefix  # Install with: pip install -U pytubefix
from   typing import Iterator, TextIO

# Function: Extract audio from video file and save to .WAV
def extract_audio(filename:str, overwrite:bool=False, silent:bool=True, acodec:str="pcm_s16le", ac:int=1, ar:str="16k"):
    outfile = '.'.join(os.path.basename(filename).split('.')[:-1]) + '.wav'
    if os.path.exists(outfile) and not overwrite:
        print(f"Audio file already exists. To overwrite, pass overwrite=True")
    else:
        print(f"Extracting audio as: {outfile} (codec: {acodec}, {ac} channels @{ar}Hz)")
        ffmpeg.input(filename).output(
            outfile,
            acodec=acodec, ac=ac, ar=ar
        ).run(quiet=silent, overwrite_output=True)
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
def write_srt(transcript:Iterator[dict], translation:Iterator[dict]=[], outfile:TextIO="output.srt", dry_run:bool=False):
    for i, segment in enumerate(transcript, start=1):
        output_text = f"{i}\n"
        output_text += f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
        output_text += f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
        if len(translation) > 0:
            output_text += f'<font color="#gray">{segment["text"].strip().replace("-->", "->")}</font>\n'
            output_text += f'{translation[i-1].get("text").strip().replace("-->", "->")}\n'
        else:
            output_text += f'{segment["text"].strip().replace("-->", "->")}\n'
        if dry_run:
            print(output_text, flush=True)
        else:
            print(output_text, file=outfile, flush=True)

# Function: Remove repetitive strings
def remove_repetitions_and_sequences(input_string:str, char_repeats:int=6, seq_repeats:int=3, DEBUG=False) -> str:
    if DEBUG:
        print(f"Received input: {input_string}")

    # Remove invalid characters
    input_string = input_string.replace("�", "")

    # Handle single-character repetitions first
    single_char_pattern = rf'(.)\1{{{char_repeats},}}'
    result = re.sub(single_char_pattern, lambda m: m.group(1) * char_repeats, input_string)
    if DEBUG:
        print(f"Removed single-char sequences: {result}")

    # Handle repetitive multi-character sequences
    sequence_pattern = rf'((.+?)\2{{{seq_repeats},}})'
    result = re.sub(sequence_pattern, lambda m: m.group(2) * seq_repeats if len(m.group(2)) > 1 else m.group(0), result)
    if DEBUG:
        print(f"Removed multi-char sequences: {result}")

    return result

def cleanup_text(input_dict:dict) -> dict:
    input_string = input_dict.get("text", "").strip()
    # Remove prefix "-" at start of lines
    input_string = re.sub(r"^- ", "", input_string, flags=re.MULTILINE)
    output_string = remove_repetitions_and_sequences(input_string)
    input_dict["text"] = output_string
    return input_dict

def remove_dup_segments(lst):
    if not lst:  # Handle empty list
        return []

    result = [lst[0]]  # Initialize with the first element
    for i in range(1, len(lst)):
        if lst[i].get("text") != lst[i - 1].get("text"):  # Compare with the previous element
            result.append(lst[i])
    return result

def adjust_duration(input_dict:dict) -> dict:
    length_duration = round(float(input_dict.get("end")) - float(input_dict.get("start")), 2)
    length_text = len(input_dict.get("text"))
    length_max = max(length_text / 4, 10)
    if length_duration > length_max:
        input_dict["end"] = float(input_dict.get("start")) + length_max
        print(f"Shortened length: {input_dict.get("start")} --> {input_dict.get("end")} ({length_duration} -> {length_max}) {input_dict.get("text")}")
    """
    length_min = min(length_text / 4, 10)
    if length_duration < length_min:
        input_dict["end"] = float(input_dict.get("start")) + length_min
        print(f"Increased length: {input_dict.get("start")} --> {input_dict.get("end")} ({length_duration} -> {length_min}) {input_dict.get("text")}")
    """
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
