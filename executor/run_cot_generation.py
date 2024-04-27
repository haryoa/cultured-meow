import os
import subprocess

models = {
    "MISTRAL": "mistralai/Mistral-7B-Instruct-v0.2",
    "QWEN": "Qwen/Qwen1.5-7B-Chat",
    "SAILOR": "sail/Sailor-7B-Chat",
    "llama3": "meta-llama/Meta-Llama-3-8B",
}

languages = {
    "Indonesia": "id",
    "USA": "en",
    "China": "ch",
    "Nigeria": "hau",
    "Russia": "rus",
    "India": "ind",
    "Ecuador": "ecu",
    "Egypt": "egp"
}

# Define your prompts
prompts = [
    "data/prompt/prompt_cot/prompt1.txt",
    "data/prompt/prompt_cot/prompt2.txt",
    "data/prompt/prompt_cot/prompt3.txt",
]

# Loop through models, prompts, and languages
for model_key, model in models.items():
    for prompt in prompts:
        for country, lang in languages.items():
            # get prompt name from the path
            try:
                prompt_name = prompt.split("/")[-1].split(".")[0]

                print(f"Running {model_key} for {country}")
                # Adjust command based on the model
                cmd = f"HF_HOME=../hf-cache/ python -m survey_llm.cot_generator --template_txt_path {prompt} --yaml_survey_path data/survey/{lang}-wvs.yaml --model_name {model} --out_dir answers/open-cot-answer/{lang}-{model_key}-{prompt_name}/ --device cuda"
                # Execute the command
                subprocess.run(cmd, shell=True, check=True)
            except Exception as e:
                print(f"Error: {e}")
    # Clear the cache after processing each model for all countries and prompts
    os.system("rm -rf ../hf-cache/hub/")
