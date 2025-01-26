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
    import torch     # Install with: pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    import gc
    import dotenv
    import warnings
    import logging
    # Import custom modules
    from   translate import process_translation
    from   helpers import extract_audio, get_youtube_video, write_srt
    from   helpers import adjust_segments, cleanup_text, remove_dup_segments, adjust_duration

    # Load environment variables
    dotenv.load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    # Set debug mode
    DEBUG = args.debug

    # Filter "Bad things might happen" warning from Pyannote
    warnings.filterwarnings('ignore', message="Model was trained with .*", module='pyannote')
    # Filter "degrees of freedom is <= 0" warning from Pyannote
    warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')
    # Filter "1Torch was not compiled with flash attention" warning from TorchAudio
    warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
    # Filter "1Torch was not compiled with flash attention" warning from Transformers
    warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

    # Suppress logging messages from SpeechBrain
    logger = logging.getLogger('speechbrain')
    logging.disable(logging.CRITICAL)

    # Disable TensorFloat-32 (TF32) as it might lead to reproducibility issues and lower accuracy
    # Reference: https://github.com/pyannote/pyannote-audio/issues/1370
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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

    # Set default Whisper options
    options = {"task": "transcribe"}

    # Automatically detect language if not overridden
    if args.language:
        options['language'] = args.language

    # Set ASR Options
    asr_options = {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    # Define WhisperX parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16"
    batch_size = 16
    # NOTE: Available Whisper models: tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo
    model_name = 'large-v3'
    print(f"Loading Whisper model: {model_name}")
    model = whisperx.load_model(
        whisper_arch=model_name, device=device, compute_type=compute_type,
        language=options.get("language", None),
        asr_options=asr_options,
        vad_method="silero",  # Default: "pyannote"
        vad_options=None,
    )

    # Load audio file
    audio_file = extract_audio(filename)
    audio = whisperx.load_audio(audio_file)

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
        print_progress=True,
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

    list_transcribe = []
    for segment in result.get("segments"):
        print(f'    {segment["start"]} --> {segment["end"]} {segment["text"].strip()}')
        list_transcribe.append(segment)
    print(f"Transcribed {len(list_transcribe)} segments")

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
