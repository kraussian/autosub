# AutoSub

AutoSub automatically extracts subtitles from video or audio files using OpenAI Whisper

There are two versions of AutoSub:

1. `autosub.py`: Uses [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and [Silero VAD](https://github.com/snakers4/silero-vad) to greatly improve speed and quality of transcriptions
2. `autosub-pure.py`: Uses the base [Whisper](https://github.com/openai/whisper) by OpenAI. Only for reference purposes, as the main implementation is superior in every aspect

## Usage Instructions

```shell
python autosub.py [-h] [-l LANGUAGE] [-c CHUNKS] [-t] filename

positional arguments:
  filename              Path and name of the video file to extract from, or URL of YouTube video

options:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        Override language of video file, e.g. en, ja, ko, zh
  -c CHUNKS, --chunks CHUNKS
                        Override number of random chunks to use for detecting language
  -t, --translate       Automatically translate subtitles to English
```
