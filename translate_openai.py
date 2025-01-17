import os
import sys
import re
import pickle
import openai  # Install with: pip install -U openai
import dotenv  # Install with pip install -U python-dotenv
from   tqdm import tqdm  # Install with: pip install -U tqdm

# Load environment variables
dotenv.load_dotenv()

# Define app parameters
MODEL = os.getenv("OPENAI_MODEL")
HOST = os.getenv("OPENAI_HOST", "")
PORT = os.getenv("OPENAI_PORT", 80)
API_KEY = os.getenv("OPENAI_API_KEY", "MY-KEY")
API_URL = f"http://{HOST}:{PORT}/v1"

# Function: Run user input against LLM
def process_llm(user_input:str, conversation_history:list=[], model:str=MODEL, temperature:float=0.2, num_choices:int=3, DEBUG:bool=False) -> str:
    def quality_score(translation, original):
        match = re.search(r"---BEGIN TEXT---\n(.*?)\n---END TEXT---", original, re.DOTALL)
        if match:
            # Extract the text between the markers and split into lines
            original_segments = match.group(1).splitlines()
        else:
            print(f"Received input:\n{original}")
            raise ValueError("No text found between ---BEGIN TEXT--- and ---END TEXT---")
        # Split the translation into lines (segments)
        translated_segments = translation.splitlines()
        # Compare the number of segments
        segment_match_score = -abs(len(translated_segments) - len(original_segments))
        # Return the score (higher is better)
        return segment_match_score

    # Append current user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Setup OpenAI-compatible client
    if HOST == "":
        client = openai.OpenAI()  # Use OpenAI
    else:
        client = openai.OpenAI(base_url=API_URL)  # Use custom OpenAI-compatible endpoint

    # Generate chat response from LLM
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        n=num_choices,
        messages=conversation_history,
    )

    final_response = ""
    if DEBUG:
        print(f"Response received: {response}")
    if response:
        best_translation = max(response.choices, key=lambda x: quality_score(x.message.content.strip(), user_input))
        if DEBUG:
            print(f"Best translation:\n{best_translation.message.content.strip()}")
        final_response = best_translation.message.content.strip()
        #final_response = response.choices[0].message.content.strip()
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
        res = process_llm(input_text)
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
        "Preserve the original structure, including the number of segments and line breaks. "
        "Do not add, remove, or modify lines. "
        "Provide the translation only.\n"
    )
    # Use only 90% of MAX_TOKENS to be conservative
    #MAX_TOKENS = int(int(os.getenv("MAX_TOKENS", 8000)) * 0.9)
    MAX_TOKENS = 1024  # Split into chunks if prompt > 1k characters
    prompt = f"{base_prompt}---BEGIN TEXT---\n{trans_text}\n---END TEXT---"
    if estimate_token_count(prompt) < MAX_TOKENS:
        print(f"Translating {len(list_transcribe)} segments, {len(trans_text)} characters")
        res = process_llm(user_input=prompt, conversation_history=[])
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
            RETRIES = 3
            for attempt in range(1, RETRIES+1):
                res = process_llm(user_input=prompt, conversation_history=[])
                res = re.sub(r"---.*?---", "", res).strip()
                list_res = res.split("\n")
                # Retry translation if number of chunks do not match
                if len(list_res) != len(chunk.split("\n")):
                    print(f"    Translation size {len(list_res)} and chunk size {len(chunk.split('\n'))} do not match. Retrying...")
                else:
                    break  # Break out of loop since process is successful
            if attempt == RETRIES:
                with open(file="chunk.txt", mode="w", encoding="utf-8") as f:
                    f.write(chunk)
                with open(file="res.txt", mode="w", encoding="utf-8") as f:
                    f.write(res)
                print("    Failed translation after maximum number of retries. chunk.txt and res.txt saved for debugging")
                sys.exit(-1)
                #break  # Maximum number of unsuccessful retries, terminal loop
            for item in list_res:
                try:
                    regex_result = re.findall(r"\[[a-zA-Z0-9]+\] (.*)", item.strip())
                    if len(regex_result) > 0:
                        item_text = regex_result[0]
                    else:
                        item_text = item.strip()
                    # Capitalize first word of each sentence if it's not already capitalized
                    if not item_text[0].isupper():
                        item_text = item_text[0].upper() + item_text[1:]
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
            res = process_llm(prompt, conversation_history=[])
            res = re.sub(r"---.*?---", "", res).strip()
            list_translate.append({"text": res})
    return list_translate
