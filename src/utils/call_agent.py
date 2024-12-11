import argparse
import os
import time
import warnings
from random import random

import pandas as pd
import requests
from IPython import embed
from openai import OpenAI
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION")
)

def ask_agent(model, history):
    count = 0
    try:
        count += 1
        if model in ['gpt-4-vision-preview', 'gpt-4-turbo', 'gpt-4o']:
            params = {
                "model": model,
                "messages": history,
                "max_tokens": 4096,
            }
            response = client.chat.completions.create(**params)
            resp = response.choices[0].message.content
        else:
            print(f"Unrecognized model name: {model}")
            return 0
    except Exception as e:
        print('OpenAI API error: ', e)
        if isinstance(e, client.exceptions.OpenAIError) and e.http_status in [429, 502, 500]:
            time.sleep(60 + 10 * random())  # wait for 60 seconds and try again
            if count < 25:
                resp = ask_agent(model, history)
            else:
                return e
        elif isinstance(e, client.exceptions.OpenAIError) and e.http_status == 400:  # images cannot be processed due to safety filter
            if len(history) == 4 or len(history) == 2:  # if dataset exemplars are unsafe
                return e
            else:  # else, remove the last experiment and try again
                resp = ask_agent(model, history[:-2])
    return resp