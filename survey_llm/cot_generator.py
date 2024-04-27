from itertools import combinations, permutations
from yaml import safe_load
import torch
import torch.nn.functional as F
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.table import Table
from tqdm.rich import trange, tqdm
from tqdm import tqdm as tqdm_base
import yaml
from pathlib import Path
import fire

console = Console()


def do_inference_on_survey(
    model_name: str,
    survey_questions: dict[str, str],
    out_path: str,
    device: str = "cpu",
    trust_remote: bool = False,
):
    """
    Perform inference on a survey using a language model.

    Args:
        model_name (str): The name or path of the pre-trained language model.
        survey_questions (dict[str, str]): A dictionary containing the survey questions.
            The keys are the question IDs and the values are the survey questions.
        out_path (str): The path to save the output YAML file.
        device (str, optional): The device to run the model on (e.g., "cpu", "cuda").
            Defaults to "cpu".
        trust_remote (bool, optional): Whether to trust remote code when loading the tokenizer.
            Defaults to False.

    Returns:
        dict[str, str]: A dictionary containing the generated responses for each question.

    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
    )
    model = model.eval()

    all_outputs = {}
    for key, questions in survey_questions.items():
        inputs = tokenizer.encode(
            questions, add_special_tokens=False, return_tensors="pt"
        )
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=512)
        decoded = tokenizer.decode(
            outputs[0][inputs.shape[-1] :], skip_special_tokens=True
        )
        all_outputs[key] = decoded

    with open(out_path, "w") as f:
        yaml.dump(all_outputs, f)

    return all_outputs


def _format_to_llm_input(
    tokenizer,
    user_input: str,
):
    """
    Format user input for the language model.

    Args:
        tokenizer: The tokenizer object.
        user_input (str): The user input.

    Returns:
        str: The formatted input for the language model.

    """
    messages = [
        {"role": "user", "content": str(user_input)},
        {"role": "assistant", "content": "Let's think step by step."},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_survey_questions(
    template_txt_path: str,
    yaml_survey_path: str,
    tokenizer=None,
):
    """
    Load survey questions from a YAML file and format them using a template.

    Args:
        template_txt_path (str): The path to the template text file.
        yaml_survey_path (str): The path to the YAML file containing the survey questions.
        tokenizer: The tokenizer object. Defaults to None.

    Returns:
        dict[str, str]: A dictionary containing the formatted survey questions.

    """
    with open(template_txt_path, "r") as f:
        template = f.read()
    with open(yaml_survey_path, "r") as f:
        survey = safe_load(f)
    returned_template = {}
    for key, questions in survey.items():
        template_filled = template.format_map({"question": questions})
        print(f"Processing with template filled: {template_filled}")
        returned_template[key] = _format_to_llm_input(tokenizer, template_filled)

    return returned_template


def main(template_txt_path, yaml_survey_path, model_name, out_dir, device="cuda"):
    """
    Main function to generate responses for survey questions.

    Args:
        template_txt_path (str): The path to the template text file.
        yaml_survey_path (str): The path to the YAML file containing the survey questions.
        model_name (str): The name or path of the pre-trained language model.
        out_dir (str): The directory to save the output YAML file.
        device (str, optional): The device to run the model on (e.g., "cpu", "cuda").
            Defaults to "cuda".

    """
    template_txt_name = Path(template_txt_path).stem
    out_path_cot = Path(out_dir) / f"{template_txt_name}-{model_name}-cot.yaml"
    out_path_cot = str(out_path_cot)
    # path out_dir create
    Path(out_path_cot).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    survey_questions = load_survey_questions(
        template_txt_path, yaml_survey_path, tokenizer
    )
    do_inference_on_survey(model_name, survey_questions, out_path_cot, device=device)


if __name__ == "__main__":
    fire.Fire(main)
