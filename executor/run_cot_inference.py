import os
import subprocess
import fire
from glob import glob


def main(langs: tuple[str] | None = None, device: str = "cuda"):
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

    prompts_non_country = [
        "data/prompt/prompt_cot/prompt3.txt",
    ]

    dir_cot = "answers/open-cot-answer/{lang}-{model_key}-prompt3/*/*.yaml"
    if langs is not None:
        # languages_split = langs.split(",")  # e.g.: Indonesia,USA,China
        print(langs)
        languages = {lang: languages[lang] for lang in langs}
        
    for model_key, model in models.items():
        for prompt in prompts_non_country:
            for _, lang in languages.items():
                try:
                    prompt_name = prompt.split("/")[-1].split(".")[0]
                    print(f"Running {model_key} for non_country in {lang}")
                    # Adjust command based on the model
                    cot_path = glob(dir_cot.format(lang=lang, model_key=model_key))[0]
                    cmd = f"HF_HOME=../hf-cache/ python -m survey_llm survey-model {prompt} data/survey/{lang}-wvs.yaml data/survey-answer/{lang}-wvs.yaml non {model} open-cot-inference-answer/non-{lang}-{model_key}-{prompt_name}.json --device {device} --cot-path {cot_path} --is-instruction"
                    # Execute the command
                    subprocess.run(cmd, shell=True, check=True)
                except Exception as e:
                    print(f"Error: {e}")
            # Clear the cache after processing each model for all countries and prompts
        os.system("rm -rf ../hf-cache/hub/")


if __name__ == "__main__":
    fire.Fire(main)
