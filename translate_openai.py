import os
import re
import openai  # Install with: pip install -U openai
import dotenv  # Install with pip install -U python-dotenv
from   tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

# Define app parameters
MODEL = os.getenv("OPENAI_MODEL")
HOST = os.getenv("OPENAI_HOST")
PORT = os.getenv("OPENAI_PORT", 80)
API_URL = f"http://{HOST}:{PORT}/v1"

def process_user_input(user_input:str, conversation_history:list=[], model:str=MODEL, DEBUG:bool=False) -> str:
    """Process user input and return assistant's response."""
    conversation_history.append({"role": "user", "content": user_input})

    # Setup OpenAI-compatible client
    client = openai.OpenAI(
        base_url=API_URL,
        api_key="my-key"
    )

    response = client.chat.completions.create(
        model=model,
        messages=conversation_history,
    )

    final_response = ""
    if DEBUG:
        print(f"Response received: {response}")
    if response:
        final_response = response.choices[0].message.content.strip()
        conversation_history.append(
            {"role": "assistant", "content": final_response}
        )
    else:
        print("ERROR: No response received!")

    return final_response

# Function: Estimate token count
def estimate_token_count(prompt:str=""):
    prompt_tokens = len(prompt)
    response_tokens = prompt_tokens
    return prompt_tokens + response_tokens

# Function: Split text into chunks
def split_text_into_chunks(text, max_characters):
    # Divide MAX_TOKENS into half to account for response tokens
    max_characters = int(max_characters / 2)
    chunks = []
    while len(text) > max_characters:
        # Find the last newline within the max_characters range to keep segments intact
        split_point = text[:max_characters].rfind("\n")
        if split_point == -1:  # If no newline, split at max_characters
            split_point = max_characters
        chunks.append(text[:split_point].strip())
        text = text[split_point:].strip()
    chunks.append(text)  # Add the remaining text
    return chunks

def process_translation(list_transcribe:list=[]) -> list:
    """
    for seg in tqdm(list_transcribe, total=len(list_transcribe), desc="Fixing Transcription"):
        orig_text = seg.get("text")
        input_text = f"This text has been transcribed using OpenAI Whisper. Look for obvious errors and correct it: {orig_text}. Give me only the corrected text, nothing else."
        res = process_user_input(input_text)
        if orig_text != res:
            print(f"{orig_text} ->\n{res}")
    """
    list_translate = []
    trans_text = ""
    for i, segment in enumerate(list_transcribe):
        trans_text += f'[{i}] {segment.get("text")}\n'
    trans_text = trans_text.strip()
    # Define the prompt structure
    base_prompt = (
        "Translate the following text to English. "
        "Do not add any new lines or modify the original structure. "
        "If a sentence doesn't make sense, make your best guess as to what it should have been and translate that. "
        "Give me only the output, nothing else.\n"
    )
    # Use only 90% of MAX_TOKENS to be conservative
    #MAX_TOKENS = int(int(os.getenv("MAX_TOKENS", 8000)) * 0.9)
    MAX_TOKENS = 2000  # Split into chunks if prompt > 2k characters
    prompt = f"{base_prompt}---BEGIN TEXT---\n{trans_text}\n---END TEXT---"
    if estimate_token_count(prompt) < MAX_TOKENS:
        print(f"Translating {len(list_transcribe)} segments, {len(trans_text)} characters")
        res = process_user_input(user_input=prompt, conversation_history=[])
        list_res = res.split("\n")
        if len(list_res) == len(list_transcribe):
            for item in list_res:
                item_text = re.findall(r"\[[a-zA-Z0-9]+\] (.*)", item.strip())[0]
                list_translate.append({"text": item_text})
    else:
        chunks = split_text_into_chunks(trans_text, MAX_TOKENS)
        print(f"Transcription is very long ({len(trans_text)} chars). Splitting into {len(chunks)} chunks for translation")
        for chunk in chunks:
            print(f"    Translating chunk of size {len(chunk)}")
            terminate = False
            prompt = f"{base_prompt}---BEGIN TEXT---\n{chunk}\n---END TEXT---"
            RETRIES = 5
            for attempt in range(1, RETRIES+1):
                res = process_user_input(user_input=prompt, conversation_history=[])
                res = re.sub(r"---.*?---", "", res).strip()
                list_res = res.split("\n")                    
                # Retry translation if number of chunks do not match
                if len(list_res) != len(chunk.split("\n")):
                    print("    Translation size and chunk size do not match. Retrying")
                else:
                    break  # Break out of loop since process is successful
            for item in list_res:
                try:
                    regex_result = re.findall(r"\[[a-zA-Z0-9]+\] (.*)", item.strip())
                    if len(regex_result) > 0:
                        item_text = regex_result[0]
                    else:
                        item_text = item.strip()
                except Exception as e:
                    print("    Chunk translation failed. Terminating loop")
                    terminate = True
                    break
                list_translate.append({"text": item_text})
            if terminate:
                break
    if len(list_transcribe) != len(list_translate):
        """
        # Print all original and translated segments to manually see where the mismatch is happening
        for i, segment in enumerate(list_transcribe):
            print(f"{segment.get('text')} -> {list_translate[i].get('text')}")
        """
        # Iterate through each transcribed text and translate
        print("Bulk translation failed. Translating each segment separately")
        list_translate = []
        for seg in tqdm(list_transcribe, total=len(list_transcribe), desc="Translating"):
            #input_text = f"Translate this text to English: {seg.get('text')}. Give me only the English translation, nothing else."
            prompt = f"{base_prompt}---BEGIN TEXT---\n{seg.get('text')}\n---END TEXT---"
            res = process_user_input(prompt, conversation_history=[])
            res = re.sub(r"---.*?---", "", res).strip()
            list_translate.append({"text": res})
    return list_translate
