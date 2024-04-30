import dotenv, os, openai, csv, random
from collections import defaultdict
from datasets import load_dataset
from multiprocessing import Pool

RESULTS_DIR = 'results'
NUM_EXAMPLES = 4
NUM_TRIALS = 32

dotenv.load_dotenv()
gpt = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def prompt_gpt(prompt: str) -> str:
    response = gpt.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    output = response.choices[0].message.content.strip()
    return output


def format_examples(examples: list[dict[str, str]]) -> str:
    formatted_examples = ""

    for example in examples:
        formatted_examples += f"Question: {example['q']}\nAnswer: {example['a']}\n"

    return formatted_examples


def run_trial(case: str, task_class: str, examples: list[dict[str, str]], task: dict[str, str],
              results: list[list[str]]):
    formatted_examples = format_examples(examples)
    prompt = f"{formatted_examples}Question: {task['q']}"  # formatted_examples ends in new line character
    result = prompt_gpt(prompt)
    results.append([task_class, case, prompt, result, task['a']])


def process_trial(t: int):
    math_dataset = load_dataset('gsm8k', 'main')
    csqa_dataset_raw = load_dataset('tau/commonsense_qa')
    csqa_dataset = list(csqa_dataset_raw['train'])  # only use train set so there is an answer key
    strategyqa_dataset = load_dataset('ChilleD/StrategyQA')
    saycan_dataset_raw = load_dataset('chiayewken/saycan')['test']  # only one set
    saycan_dataset = []

    for i in range(len(saycan_dataset_raw)):
        saycan_dataset.append({
            'q': saycan_dataset_raw['INPUT'][i],
            'a': saycan_dataset_raw['OUTPUT'][i]
        })

    math_task = random.choice(math_dataset['test'])
    csqa_task = csqa_dataset.pop(random.randint(0, len(saycan_dataset)))

    csqa_task_choices = ""

    for i in range(5):
        csqa_task_choices += f"\n{csqa_task['choices']['label'][i]}: {csqa_task['choices']['text'][i]}"

    strategyqa_task = random.choice(strategyqa_dataset['test'])
    saycan_task = saycan_dataset.pop(random.randint(0, len(saycan_dataset)))

    tasks_by_task_class = {
        "Math": {
            'q': math_task['question'],
            'a': math_task['answer']
        },
        "CSQA": {
            'q': csqa_task['question'] + csqa_task_choices,
            'a': csqa_task['answerKey']
        },
        "StrategyQA": {
            'q': strategyqa_task['facts'] + '\n' + strategyqa_task['question'],
            'a': str(strategyqa_task['answer'])
        },
        "SayCan": saycan_task
    }

    examples_by_task_class = defaultdict(list)

    for example in random.sample(list(math_dataset['train']), NUM_EXAMPLES):
        examples_by_task_class["Math"].append({
            'q': example['question'],
            'a': example['answer']
        })

    for example in random.sample(csqa_dataset, NUM_EXAMPLES):
        choices = ""

        for i in range(5):
            choices += f"\n{example['choices']['label'][i]}: {example['choices']['text'][i]}"

        examples_by_task_class["CSQA"].append({
            'q': example['question'] + choices,
            'a': example['answerKey']
        })

    for example in random.sample(list(strategyqa_dataset['train']), NUM_EXAMPLES):
        examples_by_task_class["StrategyQA"].append({
            'q': example['facts'] + '\n' + example['question'],
            'a': str(example['answer'])
        })

    examples_by_task_class["SayCan"] = random.sample(saycan_dataset, NUM_EXAMPLES)

    results = [["Task Class", "Case", "Prompt", "Response", "Expected"]]

    for task_class, examples in examples_by_task_class.items():
        task = tasks_by_task_class[task_class]
        # case 0: use 0 examples
        run_trial("0", task_class, [], task, results)
        half_n = NUM_EXAMPLES // 2
        # case 1: use n / 2 examples
        # case 1a: use the first n / 2 examples
        case1a_examples = examples[:half_n]
        run_trial("1a", task_class, case1a_examples, task, results)
        # case 1b: use the other n / 2 examples
        case1b_examples = examples[half_n:]
        run_trial("1b", task_class, case1b_examples, task, results)
        # case 2: use all n examples
        case2_examples = examples
        run_trial("2", task_class, case2_examples, task, results)
        # case 3: use n / 2 examples, duplicated (for n total examples)
        # case 3a: use the first n / 2 examples
        case3a_examples = examples[:half_n] + examples[:half_n]
        run_trial("3a", task_class, case3a_examples, task, results)
        # case 3b: use the other n / 2 examples
        case3b_examples = examples[half_n:] + examples[half_n:]
        run_trial("3b", task_class, case3b_examples, task, results)

    with open(f'{RESULTS_DIR}/{t + 1}.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        for row in results:
            writer.writerow(row)


if __name__ == '__main__':
    with Pool(NUM_TRIALS) as p:
        p.map(process_trial, range(NUM_TRIALS))
