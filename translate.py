import os
import sys
import regex as re  # Install with: pip install -U regex
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
        final_response = re.sub(r"---.*?---", "", final_response).replace("`", "").replace("---", "").replace("BEGIN TEXT", "").replace("END TEXT", "").strip()
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

def process_translation(list_original:list=[], DEBUG:bool=False) -> list:
    def split_chunks(segments:list=[], chunk_size:int=10):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        return [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]

    list_translate = []
    text_original = ""
    for i, segment in enumerate(list_original):
        text_original += f'[{i}] {segment.text}\n'
    text_original = text_original.strip()
    # Define the prompt structure
    base_prompt = (
        "You are a professional translator. ",
        "Your task is to translate non-English text into fluent English while strictly adhering to the following guidelines:\n",
        "1. **Important**: Translate the meaning of the text into English. Do not romanize or transliterate the text.\n",
        "2. Preserve the original numbering, structure, and line breaks exactly as they appear in the input.\n",
        "3. Ensure each line in the translation corresponds to the same number in the original text.\n",
        "4. **Critical** The number of lines in the translation must exactly match the number of lines in the original text. ",
        "If the input has 20 lines, the output must also have 20 lines.\n",
        "5. Do not modify the format of the line indicators (e.g., keep them as `[1]`, `[2]`, etc.).\n",
        "6. Provide the translation only, without any additional commentary or explanations.\n",
    )
    chunks = split_chunks(text_original.splitlines())
    print(f"Splitting into {len(chunks)} chunks for translation")
    for chunk in tqdm.tqdm(iterable=chunks, desc="Translating", total=len(chunks)):
        chunk_text = "\n".join(chunk)
        if DEBUG:
            print(f"    Translating chunk of size {len(chunk_text)}")
        terminate = False
        prompt = f'{"".join(base_prompt)}\n---BEGIN TEXT---\n{chunk_text}\n---END TEXT---'
        RETRIES = 10
        for attempt in range(1, RETRIES+1):
            res = process_llm(user_input=prompt, conversation_history=[])
            list_res = res.splitlines()
            # Retry translation if number of chunks do not match, or translation contains empty strings
            if (len(list_res) != len(chunk)) or not all([len(re.sub(r"\[\w+\]", "", item).strip()) > 0 for item in list_res]):
                if DEBUG:
                    print(f"        Translation {len(list_res)} and chunk {len(chunk)} do not match. Retrying...")
            else:
                break  # Break out of loop since process is successful
        if attempt == RETRIES:
            with open(file="prompt.txt", mode="w", encoding="utf-8") as f:
                f.write(prompt)
            with open(file="chunk.txt", mode="w", encoding="utf-8") as f:
                f.write("\n".join(chunk))
            with open(file="res.txt", mode="w", encoding="utf-8") as f:
                f.write(res)
            print("    Failed translation after maximum number of retries. prompt.txt, chunk.txt and res.txt saved for debugging")
            sys.exit(-1)
        for item in list_res:
            try:
                regex_result = re.findall(r".*?\[\w+\].*?[ ]+(.*)", item.strip())
                item_text = regex_result[0] if len(regex_result) > 0 else item.strip()
                # Capitalize first word of each sentence if it's not already capitalized
                if not item_text[0].isupper():
                    item_text = item_text[0].upper() + item_text[1:]
            except Exception as e:
                print("    Chunk translation failed. Terminating loop")
                terminate = True
                break
            if DEBUG: print(f"    Translated: {item_text}")
            list_translate.append({"text": item_text})
        if terminate:
            break
    if len(list_original) != len(list_translate):
        # Iterate through each transcribed text and translate
        print("Bulk translation failed. Translating each segment separately")
        list_translate = []
        for seg in tqdm.tqdm(list_original, total=len(list_original), desc="Translating"):
            prompt = f"{base_prompt}---BEGIN TEXT---\n{seg.text}\n---END TEXT---"
            res = process_llm(prompt, conversation_history=[])
            res = re.sub(r"---.*?---", "", res).strip()
            list_translate.append({"text": res})
    return list_translate
