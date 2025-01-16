import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='autosub.py',
        description='AutoSub automatically extracts subtitles from video or audio files using OpenAI Whisper',
    )
    parser.add_argument('filename', help="Path and name of the video file to extract from, or URL of YouTube video")
    parser.add_argument('-l', '--language', help="Override language of video file, e.g. en, ja, ko, zh")
    parser.add_argument('-t', '--translate', help="Automatically translate subtitles to English", action='store_true')
    parser.add_argument('-o', '--openai', help="Use OpenAI API to translate subtitles, keeping transcription", action='store_true')
    parser.add_argument('-k', '--keep', help="Keep WAV file created during process", action='store_true')
    args = parser.parse_args()

    print("Importing modules")
    # Import Python modules
    import os
    import sys
    import re
    import shutil
    from   faster_whisper import WhisperModel  # Install with: pip install faster-whisper
    # Import custom modules
    from   translate_openai import process_translation
    from   helpers import extract_audio, get_youtube_video, cleanup_text, remove_dup_segments, adjust_duration
    from   helpers import write_srt

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

    OPENAI = False
    # Check if user wants to use OpenAI API to translate subtitles
    if args.openai:
        OPENAI = True
    # Check if user wants to translate instead of transcribe
    if args.translate:
        if not OPENAI:
            options['task'] = "translate"

    print("Extracting subtitles")
    vad_params = dict(
        threshold=0.05,  # Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        neg_threshold=0.01,  # Silence threshold for determining the end of speech. If a probability is lower than neg_threshold, it is always considered silence.
        min_speech_duration_ms=0,  # Final speech chunks shorter min_speech_duration_ms are thrown out
        max_speech_duration_s=1000,  # Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100ms (if any)
        #min_silence_duration_ms=500,  # In the end of each speech chunk wait for min_silence_duration_ms before separating it
        #speech_pad_ms=100,  # Final speech chunks are padded by speech_pad_ms each side
    )
    segments, info = model.transcribe(
        audio=audio_file,
        language=options.get("language"),
        task=options.get("task"),
        beam_size=10,
        log_progress=False,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,  # The library integrates the Silero VAD model to filter out parts of the audio without speech
        vad_parameters=vad_params,  # Customize VAD parameters
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
    list_transcribe = remove_dup_segments(list_transcribe)
    list_transcribe = [adjust_duration(item) for item in list_transcribe]
    print(f"Cleaned up to {len(list_transcribe)} segments")

    if not args.keep:
        print(f"Deleting audio file: {audio_file}")
        os.remove(audio_file)

    # Translate subtitles using OpenAI API
    list_translate = []
    if OPENAI:
        list_translate = process_translation(list_transcribe)
        # Final check to see if iterative translation succeeded
        if len(list_transcribe) != len(list_translate):
            print("ERROR: Translation failed")
            sys.exit(-1)

    srt_file = outfile = '.'.join(os.path.basename(filename).split('.')[:-1]) + '.srt'
    # Test: write_srt(list_transcribe, list_translate, dry_run=True)
    with open(srt_file, "w", encoding="utf-8") as srt:
        write_srt(list_transcribe, list_translate, outfile=srt)
    print(f"Wrote subtitle file to: {srt_file}")
    if not "http" in filename:
        file_dir = os.path.dirname(os.path.abspath(filename))
        if file_dir != os.getcwd():
            print(f"Moving subtitle file to: {file_dir}")
            shutil.copy2(srt_file, file_dir)
            os.remove(srt_file)

# NOTE: ffmpeg needs to be installed on the system first
# On Windows, install with: choco install ffmpeg

# NOTE: torch needs to be in installed
# pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu124
