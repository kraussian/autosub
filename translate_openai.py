import os
import sys
import re
import pickle
import openai  # Install with: pip install -U openai
import dotenv  # Install with pip install -U python-dotenv
import tqdm    # Install with: pip install -U tqdm

# Load environment variables
dotenv.load_dotenv()

# Define app parameters
MODEL = os.getenv("OPENAI_MODEL")
HOST = os.getenv("OPENAI_HOST", "")
PORT = os.getenv("OPENAI_PORT", 80)
API_KEY = os.getenv("OPENAI_API_KEY", "MY-KEY")
API_URL = f"http://{HOST}:{PORT}/v1"

# Function: Run user input against LLM
def process_llm(user_input:str, conversation_history:list=[], model:str=MODEL, temperature:float=0.5, DEBUG:bool=False) -> str:
    # Append current user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Setup OpenAI-compatible client
    if HOST == "":
        client = openai.OpenAI(api_key=API_KEY)  # Use OpenAI
    else:
        client = openai.OpenAI(base_url=API_URL)  # Use custom OpenAI-compatible endpoint

    # Generate chat response from LLM
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=conversation_history,
    )

    final_response = ""
    if DEBUG:
        print(f"Response received: {response}")
    if response:
        final_response = response.choices[0].message.content.strip()
        # Clean up artifacts in text
        final_response = re.sub(r"---.*?---", "", final_response).replace("`", "").replace("---", "").strip()
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

def split_chunks(lst, chunk_size:int=20):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_translation(list_transcribe:list=[], DEBUG:bool=False) -> list:
    """
    for seg in tqdm.tqdm(list_transcribe, total=len(list_transcribe), desc="Fixing Transcription"):
        orig_text = seg.text
        input_text = f"This text has been transcribed using OpenAI Whisper. Look for obvious errors and correct it: {orig_text}. Give me only the corrected text, nothing else."
        res = process_llm(input_text)
        if orig_text != res:
            print(f"{orig_text} ->\n{res}")
    """
    list_translate = []
    orig_text = ""
    for i, segment in enumerate(list_transcribe):
        orig_text += f'[{i}] {segment.text}\n'
    orig_text = orig_text.strip()
    # Define the prompt structure
    base_prompt = (
        "Translate the following sentences to English while keeping the numbering, structure and line breaks intact. ",
        "Make sure each line corresponds to the same number in the original text. ",
        "Do not skip or merge any lines. ",
        "Ensure the number of lines of the translation exactly match the number of lines in the original text. ",
        "Provide the translation only.\n",
    )
    chunks = split_chunks(orig_text.splitlines())
    print(f"Splitting into {len(chunks)} chunks for translation")
    for chunk in tqdm.tqdm(iterable=chunks, desc="Translating", total=len(chunks)):
        chunk_text = "\n".join(chunk)
        if DEBUG:
            print(f"    Translating chunk of size {len(chunk_text)}")
        terminate = False
        prompt = f"{"".join(base_prompt)}---BEGIN TEXT---\n{chunk_text}\n---END TEXT---"
        RETRIES = 10
        for attempt in range(1, RETRIES+1):
            res = process_llm(user_input=prompt, conversation_history=[])
            list_res = res.splitlines()
            # Retry translation if number of chunks do not match
            if len(list_res) != len(chunk):
                if DEBUG:
                    print(f"        Translation size {len(list_res)} and chunk size {len(chunk)} do not match. Retrying...")
            else:
                break  # Break out of loop since process is successful
        if attempt == RETRIES:
            with open(file="text.txt", mode="w", encoding="utf-8") as f:
                f.write(orig_text)
            with open(file="chunk.txt", mode="w", encoding="utf-8") as f:
                f.write("\n".join(chunk))
            with open(file="res.txt", mode="w", encoding="utf-8") as f:
                f.write(res)
            print("    Failed translation after maximum number of retries. text.txt, chunk.txt and res.txt saved for debugging")
            sys.exit(-1)
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
        # Iterate through each transcribed text and translate
        print("Bulk translation failed. Translating each segment separately")
        list_translate = []
        for seg in tqdm.tqdm(list_transcribe, total=len(list_transcribe), desc="Translating"):
            prompt = f"{base_prompt}---BEGIN TEXT---\n{seg.text}\n---END TEXT---"
            res = process_llm(prompt, conversation_history=[])
            res = re.sub(r"---.*?---", "", res).strip()
            list_translate.append({"text": res})
    return list_translate
