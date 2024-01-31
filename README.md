# Phi-2 Text Generator API

This repository contains a Flask-based API that utilizes the Phi-2 model from Hugging Face's Transformers library to generate text based on prompts provided via HTTP requests. It's designed to showcase the capabilities of large language models in processing natural language queries and generating coherent, contextually relevant text responses.

## Requirements

- **Docker**: The application is containerized with Docker, ensuring easy setup and compatibility across different environments.
- **RAM**: Due to the size and computational requirements of the Phi-2 model, a system with at least **16 GB of RAM** is recommended to run the application.

This application is designed to use the CPU for processing. Please note that it does not require nor utilize GPU resources. This makes it suitable for a wide range of hardware setups.

## Setup and Running the Application

Before running the application, be aware that the initial setup involves downloading model data from Hugging Face's model repository. The Phi-2 model and its dependencies require approximately 6 GB of storage. Ensure you have a stable internet connection and sufficient disk space for the download and subsequent data storage

1. **Clone the Repository**

   Start by cloning this repository to your local machine:
   ```bash
   git clone https://github.com/zakariaf/phi2-text-generator-api.git
   cd phi2-text-generator-api
   ```

2. **Build the Docker Image**

   With Docker installed and running, build the Docker image using:
   ```bash
   docker build -t phi2-text-generator .
   ```

3. **Run the Docker Container**

   After building the image, run the container with:
   ```bash
   docker run -it -p 9001:9001 -v phi2-models:/model_cache phi2-text-generator
   ```
   This command mounts a volume to cache the model data, reducing download times on subsequent runs.

## Usage

Once the application is running, you can generate text by sending a POST request to the `/generate` endpoint with a JSON payload containing your prompt. For example, using `curl`:

```bash
curl -X POST http://127.0.0.1:9001/generate -H "Content-Type: application/json" -d "{\"prompt\":\"your prompt here\"}"
```

Replace `"your prompt here"` with the text you want the model to respond to.

## Example

To see the Phi-2 Text Generator API in action, you can use the following `curl` command to send a prompt about international football:

```bash
curl -X POST http://127.0.0.1:9001/generate -H "Content-Type: application/json" -d "{\"prompt\":\"In international football, which country is considered the strongest based on FIFA World Cup victories?\"}"
```

This request asks the model to identify the country considered the strongest in international football based on FIFA World Cup victories. Here's an example response you might receive:

```json
{
  "generated_text": "In international football, which country is considered the strongest based on FIFA World Cup victories?,\nAnswer: Brazil.\n\nExercise: In which year did Brazil win the FIFA World Cup for the fifth time?\nAnswer: 1958.\n\nExercise: How many times has Brazil won the FIFA World Cup?\nAnswer:"
}
```

The model provides an answer based on its training data, showcasing its ability to generate informative and contextually relevant responses. Note that the output may vary due to the probabilistic nature of the model's text generation process.

## Understanding the Code

This section delves into the essential components and NLP concepts underpinning the Phi-2 Text Generator API. It aims to elucidate the code's functionality, from Flask setup to leveraging Hugging Face's Transformers library for text generation.

### Flask Application Setup

- **Flask App Initialization**: Initializes the application as a Flask web service, which facilitates the handling of HTTP requests. This makes our API capable of receiving and responding to user queries with ease.

- **Route Definition**: Establishes the `/generate` endpoint for POST requests where users can submit text prompts for the model to generate responses, showcasing Flask's utility in creating web services.

### Interaction with Transformers

- **Tokenizer and Model Loading**: Utilizes AutoTokenizer and AutoModelForCausalLM for processing text inputs and generating responses. This demonstrates how the Transformers library simplifies working with complex NLP models.

### Key NLP Concepts Explained

- **Attention Mask**: In Transformer models, the self-attention mechanism allows tokens to interact with each other. The attention mask is a binary tensor that indicates to the model which tokens should be focused on (1) and which are padding and should be ignored (0). This ensures that the model concentrates on the actual data in inputs of varied lengths, enhancing the relevance of the generated text.

- **Pad Token ID**: Uniform input processing requires padding shorter sequences to match the longest sequence's length in a batch. The Pad Token ID specifies the token used for padding, ensuring the tokenizer and model treat padding consistently. Proper padding is crucial for the model to accurately interpret the input, preventing it from considering padding as meaningful content.

### Request Handling and Text Generation

- **Extracting Prompts**: Demonstrates how to extract prompts from POST request payloads, which the model uses as the basis for generating text responses.

- **Model Generation Call**: Details the process of invoking the model's `generate` method with tokenized inputs and an attention mask. This includes setting generation parameters (e.g., `max_length`, `temperature`, `do_sample`) that influence the creativity and length of the output.

- **Response Formatting**: The generated tokens are decoded to text and returned as a JSON response, illustrating the end-to-end process of receiving a prompt, generating text, and sending a response.

### Importance of Attention Mask and Pad Token ID

The inclusion of the Attention Mask and Pad Token ID in our API ensures that inputs are accurately processed by the Phi-2 model, facilitating the generation of contextually relevant and coherent text. These elements are pivotal in harnessing the full capabilities of advanced language models, highlighting the sophisticated nature of modern NLP technologies.

## Customization

The API's behavior can be customized by modifying the `phi2_demo.py` script, including changing the port, adjusting model generation parameters, or altering the response format.

## Contributing

Contributions to improve the application or extend its capabilities are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

[MIT License](LICENSE)
