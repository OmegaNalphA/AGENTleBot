import time

import openai

from completions.models import ChatCompletionCreateParams, ChatCompletionResponse


def llm_call(completion_in: ChatCompletionCreateParams) -> ChatCompletionResponse:
    while True:
        try:
            response = openai.ChatCompletion.create(**completion_in.model_dump())
            return ChatCompletionResponse(**response)
        except openai.error.RateLimitError:
            print("Rate limit exceeded, waiting 10 seconds")
            time.sleep(10)
        except openai.error.Timeout:
            print("OpenAI API timeout occurred. Waiting 10 seconds and trying again.")
            time.sleep(10)
        except openai.error.APIError:
            print("OpenAI API error occurred. Waiting 10 seconds and trying again.")
            time.sleep(10)
        except openai.error.APIConnectionError:
            print(
                """OpenAI API connection error occurred. 
                Check your network settings, proxy configuration, 
                SSL certificates, or firewall rules. 
                Waiting 10 seconds and trying again."""
            )
            time.sleep(10)
        except openai.error.InvalidRequestError:
            print(
                """OpenAI API invalid request. 
                Check the documentation for the specific API method you are calling 
                and make sure you are sending valid and complete parameters.
                Waiting 10 seconds and trying again."""
            )
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print(
                "OpenAI API service unavailable. Waiting 10 seconds and trying again."
            )
            time.sleep(10)
        else:
            break
