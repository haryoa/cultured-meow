import fire
import torch
import json
import torch.nn.functional as F
from rich.console import Console
from transformers import AutoModelWithLMHead, AutoTokenizer

from survey_llm.data import load_survey_questions
from survey_llm.infer import do_inference_on_survey
from survey_llm.post_preprocess import create_csv_given_json_answer
from pathlib import Path


console = Console()


def survey_model(
    template_path: str,
    survey_yaml_path: str,
    survey_candidate_answer_path: str,
    country: str,
    model_name: str,
    output_path: str,
    device: str = "cpu",
    trust_remote: bool = False,
    is_instruction: bool = False,
    cot_path: str = None,
):
    console.log("Your arguments are:")
    console.log(locals())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote
    )
    console.print(f"Loading survey questions from {survey_yaml_path}...")
    survey_qs = load_survey_questions(
        template_path,
        survey_yaml_path,
        survey_candidate_answer_path,
        country,
        tokenizer=(
            tokenizer if is_instruction else None
        ),  # It would change the format of the survey questions to instruction format
        cot_path=cot_path,
    )

    console.print(f"Doing inference on survey using model {model_name}...")
    results = do_inference_on_survey(
        model_name,
        survey_qs,
        device,
        trust_remote,
    )

    console.print(f"Saving the results to {output_path}...")

    
    with open(output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    fire.Fire()
