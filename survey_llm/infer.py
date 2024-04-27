import torch
import torch.nn.functional as F
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.table import Table
from tqdm.rich import trange, tqdm
from tqdm import tqdm as tqdm_base


console = Console()


@torch.inference_mode()
def get_result_from_model(
    model: AutoModelForCausalLM,
    map_of_questions: dict[str, list[dict[str, torch.Tensor]]],
    text_survey_questions: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """
    Get the result from the model by selecting the most probable answer for each question.

    Args:
        model (AutoModelWithLMHead): The language model to use for inference.
        map_of_questions (dict[str, str]): A dictionary mapping each question to a list of possible answer utterances.

    Returns:
        dict[str, int]: A dictionary mapping each question to the index of the most probable answer utterance.
    """
    results = {}
    table = Table()
    table.add_column("Question", justify="right", style="cyan")
    table.add_column("Answer", justify="left", style="magenta")
    for key, questions in map_of_questions.items():

        max_index = 0
        # set it to min inf
        max_log_prob = float("-inf")
        for idx, answer_question_utterance in enumerate(tqdm(questions)):
            answer_question_utterance = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in answer_question_utterance.items()
            }
            output_ids = answer_question_utterance["input_ids"][:, 1:]
            input_ids = answer_question_utterance["input_ids"][:, :-1]
            output = model(input_ids, labels=output_ids)
            logits = output.logits
            log_probs = torch.gather(
                F.log_softmax(logits, dim=2),
                2,
                output_ids.unsqueeze(2),
            )
            log_probs = log_probs.mean()
            if log_probs > max_log_prob:
                max_log_prob = log_probs
                max_index = idx

        results[key] = max_index
        # Create table for the results

        # take last 30 char.
        table.add_row(
            key,
            (
                str(max_index)
                if text_survey_questions is None
                else text_survey_questions[key][max_index][:]
            ),
        )
        console.print(f"Question: {key}\n-----------------\n")
        print(
            (
                str(max_index)
                if text_survey_questions is None
                else text_survey_questions[key][max_index][:]
            )
        )
        console.print("\n")
    return results


def do_inference_on_survey(
    model_name: str,
    survey_questions: dict[str, str],
    device: str = "cpu",
    trust_remote: bool = False,
):
    """
    Perform inference on a survey using a language model.

    Args:
        model_name (str): The name or path of the pre-trained language model.
        survey_questions (dict[str, str]): A dictionary containing the survey questions.
            The keys are the question IDs and the values are the actual questions.
        device (str, optional): The device to run the inference on (e.g., "cpu", "cuda").
            Defaults to "cpu".
        trust_remote (bool, optional): Whether to trust remote code when loading the model.
            Defaults to False.

    Returns:
        dict: A dictionary containing the results of the inference.

    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, trust_remote_code=trust_remote
    )
    model = model.eval()
    map_of_questions = {
        k: [tokenizer(sent, return_tensors="pt") for sent in v]
        for k, v in survey_questions.items()
    }

    # hacky, if the name of the model contains "sealion" then remove "token_type_ids" from the value
    if "sealion" in model_name:
        for k, v in map_of_questions.items():
            for idx, sent in enumerate(v):
                if "token_type_ids" in sent:
                    del map_of_questions[k][idx]["token_type_ids"]
    results = get_result_from_model(model, map_of_questions, survey_questions)
    return results
