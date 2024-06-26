from itertools import combinations, permutations

from yaml import safe_load


def _format_to_llm_input(
    tokenizer,
    user_input: str,
    llm_output: str,
    additional_cot: str = None,
):
    """
    Formats the input and output strings into a format suitable for the Language Model.

    Args:
        tokenizer: The tokenizer object used to tokenize the input and output strings.
        user_input (str): The user's input string.
        llm_output (str): The output string generated by the Language Model.
        additional_cot (str, optional): Additional context to be included in the assistant's content. Defaults to None.

    Returns:
        str: The formatted input and output strings in a format suitable for the Language Model.
    """
    assistant_content = "" if additional_cot is None else "Let's think step by step. " + additional_cot + "\n\n"
    messages = [
        {"role": "user", "content": str(user_input)},
        {"role": "assistant", "content": assistant_content + str(llm_output)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def load_survey_questions(
    template_txt_path: str,
    yaml_survey_path: str,
    yaml_survey_answer_path: str,
    nation: str,
    tokenizer=None, 
    cot_path: str = None,
    return_child_and_material_answer_list: bool = False
):
    """
    Load survey questions and generate a template with filled answers.

    Parameters:
    - template_txt_path (str): Path to the template text file.
    - yaml_survey_path (str): Path to the YAML file containing the survey questions.
    - yaml_survey_answer_path (str): Path to the YAML file containing the survey answers.
    - nation (str): The nation for which the survey is being loaded.
    - tokenizer (optional): Tokenizer object for formatting the template. Default is None.
    - cot_path (str, optional): Path to the file containing additional COTs (Children of the Nation). Default is None.
    - return_child_and_material_answer_list (bool, optional): Whether to return child and material answer lists. Default is False.

    Returns:
    - dict: A dictionary containing the generated template with filled answers.
    - list (optional): A list of child answers. Only returned if `return_child_and_material_answer_list` is True.
    - list (optional): A list of material answers. Only returned if `return_child_and_material_answer_list` is True.
    """
    with open(template_txt_path, "r") as f:
        template = f.read()
    with open(yaml_survey_path, "r") as f:
        survey = safe_load(f)
    with open(yaml_survey_answer_path, "r") as f:
        survey_answers = safe_load(f)
    if cot_path:
        with open(cot_path, "r") as f:
            additional_cots: dict[str, str] = safe_load(f)
    returned_template = {}
    child_answers = None
    material_answers = None
    for i, j in survey.items():
        is_printed = False
        template_with_answers = []
        template_filled = template.format_map({"nation": nation, "question": j})
        print(f"Processing with template filled: {template_filled}")
        current_survey_answers = survey_answers[i]
        if i == "child-obey1-q8":
            # use combinatoric of the answers (11C5)
            current_survey_answers = list(combinations(current_survey_answers, 5))
            child_answers = current_survey_answers
            # change each combination separated by , and the last one with and
            current_survey_answers = [
                ", ".join(current_survey_answer[:-1])
                + ", "
                + current_survey_answer[-1]
                for current_survey_answer in current_survey_answers
            ]
        if i == "materialistic-q155":
            # use 4P2 of the answers
            current_survey_answers = list(permutations(current_survey_answers, 2))
            material_answers = current_survey_answers
            # change each combination separated by and (due to 2)
            current_survey_answers = [
                "{}, {}".format(current_survey_answer[0], current_survey_answer[1])
                for current_survey_answer in current_survey_answers
            ]
        for survey_answer in current_survey_answers:
            if cot_path:
                additional_cot = additional_cots[i]
            else:
                additional_cot = None
            if tokenizer:
                instruction_format = _format_to_llm_input(
                    tokenizer, template_filled, survey_answer, additional_cot
                )
                if not is_printed:
                    print(f"Instruction format: {instruction_format}")
                    is_printed = True
                template_with_answers.append(instruction_format)
            else:
                template_with_answers.append(template_filled + f"\n{survey_answer}")
        returned_template[i] = template_with_answers
    if return_child_and_material_answer_list:
        return returned_template, child_answers, material_answers
    return returned_template


def output_child_obey_score(
    template_path: str,
    survey_yaml_path: str,
    survey_candidate_answer_path: str,
    country: str,
    out_txt_path: str = "data/child-obey-scores.txt",
):
    """
    Calculate and output the child obey scores based on survey responses.

    Args:
        template_path (str): The path to the survey template file.
        survey_yaml_path (str): The path to the survey YAML file.
        survey_candidate_answer_path (str): The path to the survey candidate answer file.
        country (str): The country for which the scores are calculated.
        out_txt_path (str, optional): The path to the output text file. Defaults to "data/child-obey-scores.txt".
    """

    survey_qs = load_survey_questions(
        template_path, survey_yaml_path, survey_candidate_answer_path, country
    )

    dict_model = {
        "Independence": 1,
        "Determination": 1,
        "Religious": -1,
        "Obedience": -1,
    }

    scores = []

    for ans in survey_qs["child-obey1-q8"]:
        score = 0
        for key, value in dict_model.items():
            answer_part = ans.split("\n\n")[-1]
            if key in ans:
                if key in answer_part:
                    score += value

        scores.append(score)

    with open(out_txt_path, "w") as f:
        for score in scores:
            f.write(f"{score}\n")
    return scores


def output_materiliastic_score(
    template_path: str,
    survey_yaml_path: str,
    survey_candidate_answer_path: str,
    country: str,
    out_txt_path: str = "data/materiliastic.txt",
):
    """
    Calculate and output the child obey scores based on survey responses.

    Args:
        template_path (str): The path to the survey template file.
        survey_yaml_path (str): The path to the survey YAML file.
        survey_candidate_answer_path (str): The path to the survey candidate answer file.
        country (str): The country for which the scores are calculated.
        out_txt_path (str, optional): The path to the output text file. Defaults to "data/child-obey-scores.txt".
    """
    # Define the order list to map the strings to the matrix indices

    survey_qs = load_survey_questions(
        template_path, survey_yaml_path, survey_candidate_answer_path, country
    )

    # survey_qs['materialistic-q155'][-1]
    scores = []
    for ans in survey_qs["materialistic-q155"]:
        answer_part = ans.split("\n\n")[-1]
        score = extract_priorities_and_get_value(answer_part)
        scores.append(score)
    with open(out_txt_path, "w") as f:
        for score in scores:
            f.write(f"{score}\n")
    return scores


def get_matrix_value(first_priority, second_priority):
    """
    Get the value from the matrix based on the given priorities.

    Args:
        first_priority (str): The first priority.
        second_priority (str): The second priority.

    Returns:
        int: The value from the matrix corresponding to the given priorities.
    """
    order = ["Maintaining", "Giving", "Fighting", "Protecting"]
    # Define the matrix from the image

    matrix = [
        [None, 3, 1, 3],  # Maintain order
        [2, None, 2, 4],  # More to say
        [1, 3, None, 3],  # Fight raising prices
        [2, 4, 2, None],  # Freedom of speech
    ]
    # Get the index for the first and second priority from the order list
    first_index = [i for i, x in enumerate(order) if x in first_priority][0]
    second_index = [i for i, x in enumerate(order) if x in second_priority][0]
    # Return the corresponding value from the matrix
    return matrix[first_index][second_index]


def extract_priorities_and_get_value(string):
    """
    Extracts priorities from a string and returns the corresponding matrix value.

    Args:
        string (str): The string containing the priorities in the format "(priority1, priority2)".

    Returns:
        int: The matrix value corresponding to the extracted priorities.
    """
    priorities = string.strip("()").split(", ")
    first_priority = priorities[0]
    second_priority = priorities[1]

    value = get_matrix_value(first_priority, second_priority)
    return value
