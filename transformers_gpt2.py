#This code is based on the run_generation.py example provided at 
#https://github.com/huggingface/transformers/blob/master/examples/run_generation.py

import torch
import streamlit as st
import argparse

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer)

st.title("Demo of GPT2 using Transformers and Streamlit")
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer)
}

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

no_cuda = True
inp_length = st.text_input("Enter how long output sentence should be",'40')
inp_text = st.text_input("Enter starting text","This is a good day")
prompt_text = inp_text
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
n_gpu = torch.cuda.device_count()

model_type = 'gpt2'
model_name_or_path = 'gpt2'
model_class, tokenizer_class = MODEL_CLASSES[model_type]

tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
@st.cache(allow_output_mutation=True)
def load_model():
    model = model_class.from_pretrained(model_name_or_path)
    return model

model = load_model()
model.to(device)

length = adjust_length_to_model(int(inp_length), max_sequence_length=model.config.max_position_embeddings)

encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
encoded_prompt = encoded_prompt.to(device)

#@st.cache(persist=True)
def gen_text():
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
    )
    return output_sequences

output_sequences = gen_text()
stop_token = None
generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
text = text[: text.find(stop_token) if stop_token else None]

st.write(text)