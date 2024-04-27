import os
import subprocess


languages = {
    "Indonesia": "id",
    "USA": "en",
    "China": "ch",
    "Nigeria": "hau",
    "Russia": "ru",
    "India": "ind",
    "Ecuador": "ecu",
    "Egypt": "egp"
}

prompts = [
    # "data/prompt/prompt_chat_gpt/prompt1.txt",
    "data/prompt/prompt_chat_gpt/prompt1_cot.txt",
]

# Loop through models, prompts, and languages
for prompt in prompts:
    for country, lang in languages.items():
        # get prompt name from the path
        try:
            prompt_name = prompt.split("/")[-1].split(".")[0]

            # Adjust command based on the model
            is_cot_str = "--is-cot" if "cot" in prompt else ""
            cmd = f"HF_HOME=../hf-cache/ python -m survey_llm.chatgpt --template_txt_path {prompt} --yaml_survey_path data/survey/{lang}-wvs.yaml --out_dir answers/chat-gpt-answer/1/{lang}-{prompt_name}/ {is_cot_str}"
            
            # Execute the command
            subprocess.run(cmd, shell=True, check=True)
            cmd = f"HF_HOME=../hf-cache/ python -m survey_llm.chatgpt --template_txt_path {prompt} --yaml_survey_path data/survey/{lang}-wvs.yaml --out_dir answers/chat-gpt-answer/2/{lang}-{prompt_name}/ {is_cot_str}"
            subprocess.run(cmd, shell=True, check=True)
            cmd = f"HF_HOME=../hf-cache/ python -m survey_llm.chatgpt --template_txt_path {prompt} --yaml_survey_path data/survey/{lang}-wvs.yaml --out_dir answers/chat-gpt-answer/3/{lang}-{prompt_name}/ {is_cot_str}"
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            print(f"Error: {e}")
