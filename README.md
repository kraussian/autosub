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

## Example

Let's try extracting subtitles for [Obama's iconic 2004 DNC Keynote Speech](https://www.youtube.com/watch?v=ueMNqdB1QIE).  
`python autosub.py https://www.youtube.com/watch?v=ueMNqdB1QIE`

This will extract the video as `Obama's 2004 DNC keynote speech.mp4`, then extract the audio track as `Obama's 2004 DNC keynote speech.wav`. And finally generates the subtitle file `Obama's 2004 DNC keynote speech.srt`.  

Below are the first few lines of the subtitle file, showing how the speech has been transcribed.  

Example of the output can be seen [here](https://youtu.be/-3USli_2nbA)

```text
1
00:00:06,130 --> 00:00:13,130
Let me express my deepest gratitude for the privilege of addressing this convention.

2
00:00:13,130 --> 00:00:21,130
Tonight is a particular honor for me because, let's face it, my presence on this stage is pretty unlikely.

3
00:00:21,130 --> 00:00:28,130
My father was a foreign student, born and raised in a small village in Kenya.

4
00:00:28,130 --> 00:00:32,130
He grew up herding goats, went to school in a tin roof shack.

5
00:00:32,130 --> 00:00:39,470
His father, my grandfather, was a cook, a domestic servant to the British.

6
00:00:39,470 --> 00:00:42,610
But my grandfather had larger dreams for his son.
```
