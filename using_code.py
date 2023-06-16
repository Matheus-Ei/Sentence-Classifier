import asyncio
from rasa.core.agent import Agent

# Carregar o modelo treinado
agent = Agent.load("models\your_model.gz")

async def get_bot_response(user_input):
    responses = await agent.handle_text(user_input)
    return responses

while True:
    user_input = input("User: ")

    if user_input.lower() == 'sair':
        break

    # Obter a resposta do bot
    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(get_bot_response(user_input))

    # Exibir as respostas do bot
    for response in responses:
        if 'text' in response:
            print("Bot:", response["text"])
        else:
            print("Bot: Resposta inválida")