import requests
import gradio as gr
import langchain
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tempfile  # Import the tempfile module
import logging  # Import the logging module
from huggingface_hub import hf_hub_download


### Variables are in defined also in the Configmap
### This are used as a fallback
# n_threads=2 # CPU cores
# n_batch=512 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
# n_gpu_layers=43 # Change this value based on your model and your GPU VRAM pool.
# n_ctx=4096 # Context window
# n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
# title = '🦜🔗 Chatbot LLama2 on Kubernetes'
# description = 'Chatbot using LLama2 GGML model running on top of Kubernetes'
# port = 8080
# model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
# model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

def load_config():
    config = {
        "n_threads": int(os.getenv("n_threads", 2)),
        "n_batch": int(os.getenv("n_batch", 512)),
        "n_gpu_layers": int(os.getenv("n_gpu_layers", 40)),
        "n_ctx": int(os.getenv("n_ctx", 4096)),
        "title": os.getenv("title", "🦜🔗 Chatbot LLama2 on Kubernetes"),
        "description": os.getenv("description", "Chatbot using LLama2 GGML model running on top of Kubernetes"),
        "port": int(os.getenv("port", 8080)),
        "model_name_or_path": os.getenv("model_name_or_path", "TheBloke/Llama-2-13B-chat-GGML"),
        "model_basename": os.getenv("model_basename", "llama-2-13b-chat.ggmlv3.q5_1.bin"),
        "model_storage_path": os.getenv("model_storage_path", "/mnt/models"),
    }
    logging.info(f"Loaded configuration: {config}")
    return config

def download_model(model_name_or_path, model_basename, model_storage_path):
    try:
        logging.info(f"Downloading model: {model_name_or_path}/{model_basename}")
        model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
        return model_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

def prepare(model_name_or_path, model_basename, n_gpu_layers, n_batch, n_ctx, model_storage_path):
    try:
        logging.info("Preparing model...")
        model_path = download_model(model_name_or_path, model_basename, model_storage_path)
        llm = load_model(model_path, n_gpu_layers, n_batch, n_ctx)
        return llm, model_path
        logging.info("Model preparation completed.")
    except Exception as e:
        logging.error(f"Error preparing model: {str(e)}")
        raise

def load_model(model_path, n_gpu_layers, n_batch, n_ctx):
    try:
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        logging.info(f"Loading model from path: {model_path}")

        # Loading model directly using the specified path
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
        logging.error(f"Error loading model: {str(e)}")
        raise


def prompt_template():
    logging.info("Prompt template created.")
    template = """''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.
    USER: {question}
    ASSISTANT: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

def build_chain(llm):
    logging.info("Langchain LLMChain built.")
    prompt = prompt_template()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return prompt, llm_chain

def generate(prompt):
    logging.info(f"Received prompt: {prompt}")
    # The prompt will get passed to the LLM Chain!
    return llm_chain.run(prompt)
    # And will return responses

# Define a run function that sets up an image and label for classification using the gr.Interface.
def run(llm, port):
    try:
        logging.info(f"Starting Gradio interface on port {port}...")
        interface = gr.Interface(fn=generate, inputs=["text"], outputs=["text"],
                     title=title, description=description, theme=gr.themes.Soft())
        interface.launch(server_name="0.0.0.0",server_port=port, share=True)
        logging.info("Gradio interface launched.")

    except Exception as e:
        logging.error(f"Error running Gradio interface: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load Config
        config = load_config()

        # Extract configuration variables
        model_name_or_path = config.get("model_name_or_path", "TheBloke/Llama-2-13B-chat-GGML")
        model_basename = config.get("model_basename", "llama-2-13b-chat.ggmlv3.q5_1.bin")
        n_gpu_layers = config.get("n_gpu_layers", 40)
        n_batch = config.get("n_batch", 512)
        n_ctx = config.get("n_ctx", 4096)
        model_storage_path = config.get("model_storage_path", "/mnt/models")
        port = config.get("port", 8080)
        title = config.get("title", "🦜🔗 Chatbot LLama2 on Kubernetes")
        description = config.get("description", "Chatbot using LLama2 GGML model running on top of Kubernetes")
        
        # Download and load the model
        llm, model_path = prepare(model_name_or_path, model_basename, n_gpu_layers, n_batch, n_ctx, model_storage_path)

        # Build the Langchain LLMChain
        prompt, llm_chain = build_chain(llm)

        # Execute Gradio App
        run(llm, port)
    except KeyboardInterrupt:
        logging.info("Application terminated by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
