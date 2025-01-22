# AutoSub

AutoSub automatically extracts subtitles from video or audio files using OpenAI Whisper

There are two versions of AutoSub:

1. `autosub.py`: Uses [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and [Silero VAD](https://github.com/snakers4/silero-vad) to greatly improve speed and quality of transcriptions
2. `autosub-pure.py`: Uses the base [Whisper](https://github.com/openai/whisper) by OpenAI. Only for reference purposes, as the main implementation is superior in every aspect

## Usage Instructions

```shell
usage: autosub.py [-h] [-l LANGUAGE] [-t] [-o] [-k] filename

AutoSub automatically extracts subtitles from video or audio files using OpenAI Whisper

positional arguments:
  filename              Path and name of the video file to extract from, or URL of YouTube video

options:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        Override language of video file, e.g. en, ja, ko, zh
  -t, --translate       Automatically translate subtitles to English
  -o, --openai          Use OpenAI API to translate subtitles, keeping transcription
  -k, --keep            Keep WAV file created during process
```

## Translating with OpenAI

AutoSub provides the option to keep both the audio transcription in original language as well as add an English translation to the generated subtitles.

To do this, run the program with the `--openai` option, for example: `python autosub.py filename.mp4 --openai`

To use OpenAI models such as `gpt-4o`, follow these instructions:

1. Create a text file named `.env` in the same folder where you have downloaded `autosub.py`
2. Add the following entries to the `env` file
3. Replace the value of `OPENAI_MODEL` (`gpt-4o`) with the GPT model you wish to use
4. Replace the value of `OPENAI_API_KEY` (`MY-KEY`) with the API Key you have obtained from OpenAI

```ini
OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=MY-KEY
```

## Translating with OpenAI API-compatible LLM

Alternatively, you can also use other LLMs such as Llama which is hosted through an OpenAI API-compatible endpoint. If you want to do this on your own machine, I highly recommend [LM Studio](https://lmstudio.ai/).

In order to do this, follow these instructions:

1. Create a text file named `.env` in the same folder where you have downloaded `autosub.py`
2. Add the following entries to the `env` file. Replace `APIKEY` with the API Key you have obtained from OpenAI
3. Replace the value of `OPENAI_HOST` (`localhost`) with the IP address of your API server
4. Replace the value of `OPENAI_PORT` (`1234`) with the port of your API server. You can leave this blank if your API server doesn't use a non-standard port; it will be defaulted to `80`
5. Replace the value of `OPENAI_MODEL` (`llama-3.3-instruct`) with the model you wish to use

```ini
OPENAI_HOST=localhost
OPENAI_PORT=1234
OPENAI_MODEL=llama-3.3-instruct
```

## Example: Extract from YouYube

Let's try extracting subtitles for [Obama's iconic 2004 DNC Keynote Speech](https://www.youtube.com/watch?v=ueMNqdB1QIE).

Command: `python autosub.py https://www.youtube.com/watch?v=ueMNqdB1QIE`

This will extract the video as `Obama's 2004 DNC keynote speech.mp4`, then extract the audio track as `Obama's 2004 DNC keynote speech.wav`. And finally generates the subtitle file `Obama's 2004 DNC keynote speech.srt`.  

![sample_obama](https://github.com/user-attachments/assets/e0746078-121a-4293-adcd-8df8a7588858)

Full speech example [on YouTube](https://youtu.be/-3USli_2nbA)

## Example: Transcribe + Translate Non-English Speech

For this example, let's try with non-English speech in MP3 audio file taking a Japanese narration from [Mitsue Links](https://www.mitsue.co.jp/english/service/audio_and_video/audio_production/narrators_sample.html) as the sample. Asuka Yokoyama's `Sample 1 (38sec.)` was used in this example.

Command: `python autosub.py 01.mp3 --openai`

The `--openai` option was used to show the original Japanese transcription as well as show the English translation.

Here's what the output looks like:

![sample_japanese](https://github.com/user-attachments/assets/2a257b0c-a3f4-48d2-824d-b41d2a27cfae)
