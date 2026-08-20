import random

import asyncio
import os, json
from tqdm.asyncio import tqdm_asyncio

import aiolimiter

import openai
from openai import AsyncOpenAI, OpenAIError

async def _throttled_openai_chat_completion_acreate_single(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    json_format: bool = False,
    n: int = 1,
):
    async with limiter:
        for _ in range(10):
            try:
                if json_format:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        response_format={"type": "json_object"},
                    )
                else:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                await asyncio.sleep(random.randint(10, 20)) 
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                await asyncio.sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion_single(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
    json_format: bool = False,
    n: int = 1,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate_single(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            json_format=json_format,
            n=n,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if n == 1:
            if json_format:
                outputs.append(json.loads(response.choices[0].message.content))
            else:
                outputs.append(response.choices[0].message.content)
        else:
            if json_format:
                outputs.append([json.loads(response.choices[i].message.content) for i in range(n)])
            else:
                outputs.append([response.choices[i].message.content for i in range(n)])
    return outputs

async def _throttled_openai_chat_completion_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    json_format: bool = False,
    n: int = 1,
):
    async with limiter:
        for _ in range(10):
            try:
                if json_format:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                        response_format={"type": "json_object"},
                    )
                else:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        # max_completion_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                    )
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                await asyncio.sleep(random.randint(10, 20))  # 增加重试等待时间
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                await asyncio.sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 150,
    json_format: bool = False,
    n: int = 1,
):
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            json_format=json_format,
            n=n,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if n == 1:
            try:
                if json_format:
                    outputs.append(json.loads(response.choices[0].message.content))
                else:
                    outputs.append(response.choices[0].message.content)
            except Exception as e:
                print(e)
                outputs.append("")
        else:
            if json_format:
                outputs.append([json.loads(response.choices[i].message.content) for i in range(n)])
            else:
                outputs.append([response.choices[i].message.content for i in range(n)])
    return outputs
