import fire
from openai import OpenAI
from dotenv import load_dotenv
from yaml import safe_load
from rich.live import Live
from rich.table import Table
import time
import yaml
from pathlib import Path


load_dotenv()


def _format_to_llm_input(prompt: str, key: str, is_cot: bool = False):
    if key == "materialistic-q155":
        prompt_injected = (
            prompt
            + ". ANSWER WITH THE LANGUAGE PROVIDED IN THE QUESTION. YOU MUST ANSWER BY PROVIDING 2 ANSWER CHOICES, TWO. "
        )
    elif key == "child-obey1-q8":
        prompt_injected = (
            prompt
            + ". ANSWER WITH THE LANGUAGE PROVIDED IN THE QUESTION. YOU MUST ANSWER BY PROVIDING 5 ANSWER CHOICES, FIVE. "
        )
    else:
        prompt_injected = prompt

    messages = [
        (
            {"role": "system", "content": "You must follow the user."}
            if not is_cot
            else {"role": "system", "content": "You need to answer with rationale and explanation with the question's language!"}
        ),
        {"role": "user", "content": prompt_injected},
    ]

    if is_cot:
        messages.append({"role": "assistant", "content": "Let's think step-by-step."})

    return messages


def load_survey_questions(
    template_txt_path: str,
    yaml_survey_path: str,
    is_cot: bool = False,
):
    """
    yaml_survey_path: str
        Path to the yaml file containing the survey questions
        it contains the following fields:
            key: id of the question
            question: the survey question
    """
    with open(template_txt_path, "r") as f:
        template = f.read()
    with open(yaml_survey_path, "r") as f:
        survey = safe_load(f)
    returned_template = {}
    for key, questions in survey.items():
        template_filled = template.format_map({"question": questions})
        returned_template[key] = _format_to_llm_input(
            template_filled, key, is_cot=is_cot
        )
    return returned_template


def main(template_txt_path, yaml_survey_path, out_dir, is_cot=False):
    """
    Run the survey completion process using GPT-3.5 Turbo.

    Args:
        template_txt_path (str): The path to the template text file.
        yaml_survey_path (str): The path to the YAML survey file.
        out_dir (str): The output directory to save the survey responses.
        is_cot (bool, optional): Whether to use the Conversational API. Defaults to False.
    """

    out_dir = Path(out_dir)
    # make dir
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    survey_templates = load_survey_questions(
        template_txt_path, yaml_survey_path, is_cot=is_cot
    )

    answers = {}
    title = f"Survey Responses of {yaml_survey_path}"

    def generate_survey_response_table_display():
        table = Table(title=title)
        table.add_column("Question ID", style="cyan", no_wrap=True)
        table.add_column("Answer", style="green")
        for key, answer in answers.items():
            table.add_row(key, answer)
        return table

    with Live(
        generate_survey_response_table_display(), refresh_per_second=4
    ) as live:  # update 4 times a second to feel fluid
        for key, template in survey_templates.items():
            prompt = template
            live.console.print(f"Prompt: {prompt}")
            live.update(generate_survey_response_table_display())
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
            )
            answers[key] = response.choices[0].message.content
            live.update(generate_survey_response_table_display())
            time.sleep(0.5)

    yaml_survey_name = Path(yaml_survey_path).stem
    out_path = Path(out_dir) / f"{yaml_survey_name}-answers-chatgpt.yaml"
    print(f"Saving answers to {out_path}")
    with open(out_path, "w") as f:
        yaml.dump(answers, f)


if __name__ == "__main__":
    fire.Fire(main)
