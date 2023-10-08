# Llama2 in Kubernetes with Gradio only with CPU (no GPU required)!

This repo uses GGML Llama2 Optimization models to run the Llama2 13B model on a CPU (no GPU needed!) 

![Llama In K8s no GPU](./assets/llama0.png)

* Performance in AMD EPYC 7R32, 8vCPUs and 32gb RAM (m5a.2xlarge) -> 35 seconds

```md
$ kubectl logs -f -n k8s-llama2 deploy/k8s-llama2 --tail=8
Llama.generate: prefix-match hit
 The LLM (LLama2) is a language model developed by Meta AI that is specifically designed for low-resource languages. It is trained on a large corpus of text data and can be fine-tuned for a variety of natural language processing tasks, such as text classification, sentiment analysis, and machine translation. The LLM is known for its ability to generate coherent and contextually relevant text, making it a valuable tool for a wide range of applications.'' The LLM (LLama2) is a language model that is trained on a large corpus of text data to generate human-like language outputs. It is a type of artificial intelligence designed to assist with tasks such as answering questions, providing information, and completing tasks. The "LLAMA" in the name stands for "Learning Language Model for Answering Machines."

llama_print_timings:        load time = 10129.60 ms
llama_print_timings:      sample time =    71.25 ms /    84 runs   (    0.85 ms per token,  1178.96 tokens per second)
llama_print_timings: prompt eval time =     0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)
llama_print_timings:        eval time = 29505.45 ms /    84 runs   (  351.26 ms per token,     2.85 tokens per second)
llama_print_timings:       total time = 29766.70 ms
```

* No GPU used

```md
$ nvidia-smi | grep processes -A3 -B2
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## GGML for Llama2

GGML was designed to be used in conjunction with the llama.cpp library, also created by Georgi Gerganov. The library is written in C/C++ for efficient inference of Llama models. It can load GGML models and run them on a CPU. Originally, this was the main difference with GPTQ models, which are loaded and run on a GPU. 

## Model used by default

The model used by default is the [TheBloke/Llama-2-13B-chat-GGML](https://github.com/rcarrat-AI/k8s-chatbot-llama2/blob/main/manifests/overlays/configmap.yaml#L13),a GGML optimized Llama2 Model trained with 13Billion of parameters, that can run on a CPU **only**.

## Prerequisites

* Kubernetes Cluster
* Nginx Ingress Controller

>NOTE: this example uses Kind Cluster with Nginx Ingress Controller.

## Deploy Llama2 in Kubernetes

* Deploy Llama2 in Kubernetes

```md
kubectl apply -k manifests/overlays/
```

## Development

* Adjust the Makefile variables with your own specs.

* You can modify the image base and use your own:

```md
make all
```