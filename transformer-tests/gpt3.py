import openai

def read_apikey():
    api_key_path = './transformer-tests/gpt3-api-key.txt'
    with open(api_key_path, 'r') as file:
        api_key = file.read().replace('\n', '')
    return api_key


def gpt3(message):
    openai.api_key = read_apikey()
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=message,
        temperature=0,
        max_tokens=60, # 10 for intrusion
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text