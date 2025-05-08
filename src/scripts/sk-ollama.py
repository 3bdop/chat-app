import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig

from src.embeddings.vector import retriever

# Initialize kernel and services
kernel = Kernel()
chat_service = OllamaChatCompletion(ai_model_id="llama3", host="http://localhost:11434")
kernel.add_service(chat_service)

# Configure prompt template
prompt_template = """
Your name is Daleel and you are an assistant for Ebla Computer Consultancy. 
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know, don't try to make up an answer.

Relevant Context: {{$data}}

Chat History:
{{$history}}

Current Question: {{$question}}

"""

# Create semantic function
prompt_config = PromptTemplateConfig(
    template=prompt_template,
    template_format="semantic-kernel",
    input_variables=[
        {"name": "data", "description": "Relevant context"},
        {"name": "question", "description": "User question"},
        {"name": "history", "description": "Chat history"},
    ],
)

assistant_function = kernel.add_function(
    function_name="ebla_assistant",
    plugin_name="EblaPlugin",
    prompt_template_config=prompt_config,
)

# Initialize chat history
chat_history = ChatHistory()


async def main():
    print("\n---------------------------------------------------------------")
    print("Hey😀! Feel free to ask about Ebla Computer Consultancy (type 'q' to quit)!")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == "q":
                print("\nbye👋")
                break

            # Retrieve relevant context
            data = retriever.invoke(user_input)  # Your existing retriever

            # Prepare arguments
            arguments = KernelArguments(
                data=data, question=user_input, history=chat_history
            )

            # Get response
            response = await kernel.invoke(
                function=assistant_function,
                arguments=arguments,
            )

            # Update chat history
            chat_history.add_user_message(user_input)
            chat_history.add_assistant_message(str(response))

            print(f"\nAssistant: {response}")

        except Exception as e:
            print(f"\nError: {str(e)}")
            break


if __name__ == "__main__":
    asyncio.run(main())
