import requests
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# vars
n_threads=2 # CPU cores
n_batch=512 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_gpu_layers=43 # Change this value based on your model and your GPU VRAM pool.
n_ctx=4096 # Context window
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
title = '🦜🔗 Chatbot LLama2 GGML running in Kubernetes'
description = 'Chatbot LLama2 GGML'
port = 8080

# load the model
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

def download_model(model_name_or_path, model_basename):
    try:
        model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
        return model_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

def load_model(model_path, n_gpu_layers, n_batch, n_ctx):
    try:
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Loading model
        llm = LlamaCpp(
            model_path=model_path,
            max_tokens=1024,  # Replace with a named constant
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager,
            verbose=True,
            n_ctx=n_ctx,
            stop=['USER:'],  # Dynamic stopping when such token is detected.
            temperature=0.4,  # Replace with a named constant
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def prepare(model_name_or_path, model_basename, n_gpu_layers, n_batch, n_ctx):
    model_path = download_model(model_name_or_path, model_basename)
    llm = load_model(model_path, n_gpu_layers, n_batch, n_ctx)
    return llm, model_path

def prompt_model():
    template = """''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.
    USER: {question}
    ASSISTANT: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

# Define a function classify_image(inp) that preprocesses input image, performs prediction using 
# inception_net, and returns a dictionary of class labels with corresponding probabilities.
def generate_prompt(llm):
    prompt = prompt_model()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    # return llm_chain.run(prompt, llm)
    response = llm_chain.run(prompt, llm)
    return {"text": response}

# Define a run function that sets up an image and label for classification using the gr.Interface.
def run(llm, port):
    try:
        gr.Interface(fn=generate_prompt, inputs=["text"], outputs=["text"],
                     title=title, description=description).launch(server_port=port, share=True)
    except Exception as e:
        print(f"Error running Gradio interface: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        llm, model_path = prepare(model_name_or_path, model_basename, n_gpu_layers, n_batch, n_ctx)
        run(llm, port)
    except KeyboardInterrupt:
        print("Application terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")