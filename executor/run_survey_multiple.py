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

prompts = [
    "data/prompt/prompt_country/prompt1.txt",
    "data/prompt/prompt_country/prompt2.txt",
    "data/prompt/prompt_country/prompt3.txt",
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
                cmd = f"HF_HOME=../hf-cache/ python -m survey_llm survey-model {prompt} data/survey/{lang}-wvs.yaml data/survey-answer/{lang}-wvs.yaml {country} {model} answers/open-default-answer/{lang}-{model_key}-{prompt_name}.json --device cuda --is-instruction"
                # Execute the command
                subprocess.run(cmd, shell=True, check=True)
            except Exception as e:
                print(f"Error: {e}")
    # Clear the cache after processing each model for all countries and prompts
    os.system("rm -rf ../hf-cache/hub/")


# For prompt non_country

prompts_non_country = [
    "data/prompt/prompt_non_country/prompt1.txt",
    "data/prompt/prompt_non_country/prompt2.txt",
    "data/prompt/prompt_non_country/prompt3.txt",
]
for model_key, model in models.items():
    for prompt in prompts_non_country:
        for _, lang in languages.items():
            try:
                if not (model_key == "QWEN" and lang in ["id", "en",  "ch", "hau", "rus"]):
                    continue
                prompt_name = prompt.split("/")[-1].split(".")[0]
                print(f"Running {model_key} for non_country in {lang}")
                # Adjust command based on the model
                cmd = f"HF_HOME=../hf-cache/ python -m survey_llm survey-model {prompt} data/survey/{lang}-wvs.yaml data/survey-answer/{lang}-wvs.yaml non {model} answers/open-default-answer/non-{lang}-{model_key}-{prompt_name}.json --device cuda --is-instruction"
                # Execute the command
                subprocess.run(cmd, shell=True, check=True)
            except Exception as e:
                print(f"Error: {e}")
        # Clear the cache after processing each model for all countries and prompts
    os.system("rm -rf ../hf-cache/hub/")
