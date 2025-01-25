# NOTE: Installation instructions on Windows
# py -3.10 -m venv venv
# .\venv\Scripts\Activate
# pip install git+https://github.com/m-bain/whisperx.git
# pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# pip install numpy<2
# pip install python-dotenv regex av pytubefix openai

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='autosub.py',
        description='AutoSub automatically extracts subtitles from video or audio files using OpenAI Whisper',
    )
    parser.add_argument('filename', help="Path and name of the video or audio file to extract from, or URL of YouTube video")
    parser.add_argument('-l', '--language', help="Override language of video file, e.g. en, ja, ko, zh")
    parser.add_argument('-t', '--translate', help="Automatically translate subtitles to English", action='store_true')
    parser.add_argument('-o', '--openai', help="Use OpenAI API to translate subtitles, keeping transcription", action='store_true')
    parser.add_argument('--beamsize', help="Override the beam size used by Whisper")
    parser.add_argument('--noprev', help="Override the condition_on_previous_text parameter to False", action='store_true')
    parser.add_argument('--threshold', help="Override the threshold used for VAD")
    parser.add_argument('--debug', help="Add debug logs to program execution", action='store_true')
    parser.add_argument('--keep', help="Keep WAV file created during process", action='store_true')
    args = parser.parse_args()

    print("Importing modules")
    # Import Python modules
    import os
    import sys
    import pickle
    import shutil
    import whisperx  # Install with: pip install git+https://github.com/m-bain/whisperx.git
    import torch     # Install with: pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    import gc
    import dotenv
    # Import custom modules
    from   translate_openai import process_translation
    from   helpers import extract_audio, get_youtube_video, write_srt
    from   helpers import adjust_segments, cleanup_text, remove_dup_segments, adjust_duration

    # Load environment variables
    dotenv.load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    # Set debug mode
    DEBUG = args.debug

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

    # Define WhisperX parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16"
    batch_size = 16
    # NOTE: Available Whisper models: tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo
    model_name = 'large-v3'
    print(f"Loading Whisper model: {model_name}")
    model = whisperx.load_model(whisper_arch=model_name, device=device, compute_type=compute_type)

    # Load audio file
    audio_file = extract_audio(filename)
    audio = whisperx.load_audio(audio_file)

    # Set default Whisper options
    options = {"task": "transcribe"}

    # Automatically detect language if not overridden
    if args.language:
        options['language'] = args.language

    # Check if user wants to use OpenAI API to translate subtitles
    OPENAI = args.openai
    # Check if user wants to translate instead of transcribe
    if args.translate:
        if not OPENAI:
            options['task'] = "translate"

    print("Extracting subtitles")
    result = model.transcribe(
        audio=audio,
        batch_size=batch_size,
        language=options.get("language", None),
        task=options.get("task"),
    )

    # Align Whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result.get("language"), device=device)
    result = whisperx.align(result.get("segments"), model_a, metadata, audio, device, return_char_alignments=False)

    # Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # NOTE: segments is a generator so the transcription only starts when you iterate over it
    # The transcription can be run to completion by gathering the segments in a list or a for loop
    # segments = list(segments)  # The transcription will actually run here.
    print("Transcribing text")
    list_transcribe = []
    for segment in result.get("segments"):
        print(f"    {segment.start} --> {segment.end} {segment.text}")
        #if WORD_TIMESTAMPS:
        #    for word in segment.words:
        #        print(f"    [{round(word.start, 2)} -> {round(word.end, 2)}] {word.word}")
        list_transcribe.append(segment)
    print(f"Transcribed {len(list_transcribe)} segments")

    # Adjust segments to merge incomplete sentences and split at punctuation marks
    #print("Post-processing segments")
    #list_transcribe_adj = adjust_segments(list_transcribe)
    #_ = [print(f"    {seg.get('start')} --> {seg.get('end')} {seg.get('text')}") for seg in list_transcribe_adj]

    # Clean up transcribed text to remove repetitions and shorten long durations
    list_transcribe_clean = [cleanup_text(item) for item in list_transcribe]
    #list_transcribe = remove_dup_segments(list_transcribe)
    list_transcribe_clean = [adjust_duration(item) for item in list_transcribe_clean]
    if len(list_transcribe) != len(list_transcribe_clean):
        print(f"Cleaned up to {len(list_transcribe_clean)} segments")

    # Store transcription as Pickle in debug mode
    if DEBUG:
        with open(file="list_transcribe.pkl", mode="wb") as f:
            pickle.dump(obj=list_transcribe_clean, file=f, protocol=pickle.HIGHEST_PROTOCOL)
            print("    Pickled to list_transcribe.pkl for debugging")

    if not args.keep:
        print(f"Deleting audio file: {audio_file}")
        os.remove(audio_file)

    # Translate subtitles using OpenAI API
    list_translate = []
    if OPENAI:
        list_translate = process_translation(list_original=list_transcribe_clean, DEBUG=DEBUG)
        # Check to see if translation succeeded
        if len(list_transcribe_clean) != len(list_translate):
            print("ERROR: Translation failed")
            sys.exit(-1)

    srt_file = outfile = '.'.join(os.path.basename(filename).split('.')[:-1]) + '.srt'
    # Test: write_srt(list_transcribe, list_translate, dry_run=True)
    with open(srt_file, "w", encoding="utf-8") as srt:
        write_srt(list_transcribe_clean, list_translate, outfile=srt)
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
