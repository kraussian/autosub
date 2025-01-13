# AutoSub

AutoSub automatically extracts subtitles from video or audio files using OpenAI Whisper

There are two versions of AutoSub:

1. `autosub.py`: Uses [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and [Silero VAD](https://github.com/snakers4/silero-vad) to greatly improve speed and quality of transcriptions
2. `autosub-pure.py`: Uses the base [Whisper](https://github.com/openai/whisper) by OpenAI. Only for reference purposes, as the main implementation is superior in every aspect
