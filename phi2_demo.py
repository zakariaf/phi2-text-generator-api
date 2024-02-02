# Import necessary libraries from Flask for creating the web app and handling requests,
# and from transformers for interacting with the Phi-2 model.
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

# Initialize the Flask app.
app = Flask(__name__)

# Print statements for console feedback when loading starts and completes.
# Helpful for debugging and to know when the app is ready.
print("Starting to load the tokenizer and model...")

# Load the tokenizer and model using Hugging Face's Transformers library.
# The tokenizer turns text inputs into a format the model can understand,
# and the model is what generates the text based on those tokenized inputs.
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
# Pad Token ID: This is the specific token used for padding the input sequences to achieve a uniform length
# across all sequences in a batch. Padding becomes necessary because machine learning models, especially those used in NLP,
# are trained on and expect input data to be presented in batches of consistent size.
# The consistent length is achieved by filling in the shorter sequences with a pad token
# so that every sequence in the batch matches the length of the longest sequence.
# The uniformity in sequence length is crucial for the model to process the data correctly.

# Here, we check if the tokenizer already has a pad token defined.
# If not, we manually add a special token for padding purposes. We choose '[PAD]' as the pad token symbol.
# This step ensures that our data preprocessing aligns with the model's requirements for input data format.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Note: For generative models like GPT, which are designed to produce sequences of text,
    # it might be more appropriate to use the eos_token (end-of-sequence token) as the pad token.
    # This is because the eos_token naturally signifies the end of a generated text sequence,
    # making it a suitable choice for padding in scenarios where maintaining the integrity of textual output is important.
    # However, when we explicitly set the pad token to '[PAD]',
    # we are ensuring that our data preprocessing step introduces a clear and distinct token for padding,
    # which may be more suitable for tasks that require distinguishing padded areas from actual sequence content.

    # tokenizer.pad_token = tokenizer.eos_token # An alternative approach for models like GPT.

config = AutoConfig.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", config=config)

print("Model and tokenizer loaded successfully.")

# Define a route for the generate endpoint, allowing POST requests.
# This is the main interface of the API, where users can send prompts for text generation.
@app.route('/generate', methods=['POST'])
def generate_text():
    if request.method == 'POST':
        # Extract the 'prompt' data from the incoming JSON request.
        data = request.json
        prompt = data.get('prompt', '')

        # Check if a prompt was provided.
        if prompt:
            # Encode the prompt to a format suitable for the model, with padding and truncation.
            input_ids = tokenizer.encode(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

            # Generate an attention mask for the input. This helps the model know which parts of the input are actual data vs padding.
            # Attention Mask: A binary tensor indicating the position of padded indices so the model does not attend to them.
            # For each token in the input, the mask will be 1 if the token is not padding and 0 if it is padding.
            attention_mask = input_ids.ne(tokenizer.pad_token_id).int()

            # Generate text based on the encoded input and additional parameters.
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id, # Explicitly setting pad token ID for generation.
                temperature=0.7,  # Adjust based on desired creativity
                top_k=50,  # Consider tuning
                top_p=0.95,  # Consider tuning
                do_sample=True,  # Enable sampling-based generation
                max_new_tokens=50  # Control the length of the generated output
                # max_length=200,  # Commented out to rely on max_new_tokens
            )

            # Decode the generated tokens to a string and return it as part of a JSON response.
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            return jsonify({'generated_text': generated_text})
        else:
            # Return an error if no prompt was provided in the request.
            return jsonify({'error': 'No prompt provided'}), 400

# Start the Flask app on host 0.0.0.0 (making it accessible from other machines) and port 9001.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
