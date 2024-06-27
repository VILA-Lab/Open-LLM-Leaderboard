import pandas as pd
import os
from openai import OpenAI
import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import argparse

show_progress = True
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def get_chat_completion(prompt):
    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(messages=prompt,
                                                        model='gpt-4-turbo')
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            attempt += 1
            time.sleep(3)
    print(f"Failed to get chat completion after {max_retries} attempts")
    return "Cannot be answered"

def run(prompts, num_threads):
    with ThreadPool(num_threads) as pool:
        iter = pool.imap(get_chat_completion, prompts)
        idx_and_result = list(tqdm(iter, total=len(prompts), disable=False))
    return idx_and_result

def create_chat_prompt(question, llm_answer, answer):
    sys_msg = """Evaluate the answer of a AI model to a question. You will be provided with the question, the AI model’s answer, and the correct answer. Your task is to evaluate the AI model’s response and determine whether it is Correct or Incorrect.
            Grade the AI model answers based ONLY on their factual accuracy. It is OK if the AI model answer contains more information than the true answer, as long as it does not contain any conflicting statements. Otherwise, it should be marked as Incorrect. Ignore differences in punctuation and phrasing between the AI model’s answer and the true answer.
            Example Format:
            QUESTION: question here
            STUDENT ANSWER: student’s answer here
            TRUE ANSWER: true answer here
            GRADE: Correct or Incorrect here
            Your response should include only the verdict without any justification or reasoning"""
    user_prompt = f"""QUESTION: {question}\n AI ANSWER: {llm_answer}\n TRUE ANSWER: {answer}\n GRADE: """
    return [
        {"role": "system", "content": sys_msg}, 
        {"role": "user", "content": user_prompt}
    ]

def expand_labels(row):
    new_row = {}
    for item in row:
        new_row[item['label']] = item['text']
    return new_row

def evaluate(datasets, model, num_threads=10):
    for dataset in datasets:
        print(f"Processing {dataset}")
        df = pd.read_json(f"questions/{dataset}")
        expanded_df = df['options'].apply(expand_labels).apply(pd.Series)
        df = pd.concat([df, expanded_df], axis=1)
        responses = pd.read_json(f"{model}/{dataset}")
        df['os_answer'] = responses['os_answer']
        df['input'] = df.apply(lambda x: create_chat_prompt(x['question'], x['os_answer'], x[x['answerKey']]), axis=1)
        prompts = df['input'].tolist()
        df['os_eval'] = run(prompts, num_threads)
        df['gold_answer'] = df.apply(lambda x: x[x['answerKey']], axis=1)
        os.makedirs(f"evaluations/{model}", exist_ok=True)
        df[['question', 'gold_answer', 'os_answer', 'os_eval']].to_json(f"evaluations/{model}/{dataset}", orient='records')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPT4o')
    parser.add_argument('--parallel', type=int, default=10)
    args = parser.parse_args()
    files = os.listdir("questions")
    evaluate(files, args.model, num_threads=args.parallel)