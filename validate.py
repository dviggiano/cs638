import dotenv, os, openai, csv, random
from collections import defaultdict
from datasets import load_dataset
from multiprocessing import Pool

dotenv.load_dotenv()
gpt = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def prompt_gpt(prompt: str) -> str:
    response = gpt.chat.completions.create(
        model='gpt-3.5-turbo',
        max_tokens=3,
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    output = response.choices[0].message.content.strip()
    return output


def validate(i):
    i += 1

    with open(f'results/{i}.csv', mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        


if __name__ == '__main__':
    with Pool(32) as p:
        p.map(validate, range(32))
