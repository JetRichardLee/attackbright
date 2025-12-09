import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
from retrievers import attack_by_surrogate
import torch
import torch.nn.functional as F


#"""
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-4B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

#prompt = "today is a good day. Any activity suggested?"
doc = "salt could be used to make snow melt faster"
messages = [
    {"role": "system", "content": "You are a helpful assistant. A knowledge document content is provided followed by the user. Please generate a query with a practical scenario that (1) seems not directly related to the document content (2) the document would be helpful to solve the query."},
    {"role": "user", "content": doc}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=200
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

print(generated_ids)
query = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Original Query: ", query)


doc = "sea water contains a lot of salt"

messages = [
            {"role": "system", "content": "You are a helpful query writer for information retrieval system. A knowledge document content is provided followed by the user. Please first indentify its knowledge topic and then generate an unrelated query that totally belongs to another knowledge topic. You could choose any topic unrelated to the document."},
         {"role": "user", "content": doc}
        ]

text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=200  
)
generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

"""
messages = [
    {"role": "system", "content": "You are a helpful assistant. I'm giving you a document and a query, please answer 'yes' or 'no' to indicate if they are related."},
    {"role": "user", "content": f"Query: {query}"},
    {"role": "user", "content": f"Document: {doc}"}
]
#prompt=f"Is this query and document related?  Query: {query} Document: {doc} Answer:"
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
yes_id =  tokenizer(["yes"], return_tensors="pt").input_ids

yes_id =  yes_id[0]
no_id =  tokenizer(["no"], return_tensors="pt").input_ids
no_id = no_id[0]
print(yes_id)
print(no_id)
#print(model_inputs)

with torch.no_grad():
    outputs = model(**model_inputs)

# Get logits for next token (the last position)
print("shapes:")
print(outputs.logits.shape)
next_token_logits = outputs.logits[:, -1, :]  # shape [1, vocab_size]

# Convert logits to probabilities
probs = F.softmax(next_token_logits, dim=-1)
print(probs.shape)
print(probs[0,yes_id])
print(probs[0,no_id])
print(None.shape)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=200
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print(generated_ids)

ans = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(ans)

#messages = [
#    {"role": "system", "content": "You are a helpful assistant. A knowledge document content is provided followed by the user. Please generate a query with a practical scenario that (1) seems not directly related to the document content (2) the document would be helpful to solve the query."},
#    {"role": "user", "content": doc}
#]
"""
"""
doc_idx = tokenizer([doc], return_tensors="pt")["input_ids"].to(device)
query_idx = tokenizer([query], return_tensors="pt")["input_ids"].to(device)

word_embedding_layer = model.get_input_embeddings()
#embedding_matrix = word_embedding_layer.weight.data

# Get word embeddings for input sequence
doc_embeddings = word_embedding_layer(doc_idx)
#print(doc_embeddings.requires_grad)# = True

query_embeddings = word_embedding_layer(query_idx)[0,-1]

cos_sim = F.cosine_similarity(F.normalize(doc_embeddings, p=2, dim=1), query_embeddings, dim=-1).mean()
print("Original Doc: ",doc)
print("SIM: ", cos_sim)
#"""
#adv_doc,_ = attack_by_surrogate(doc,num_sts_tokens=5,num_q=5,epochs=200)
#print("Adv Document: ",adv_doc)


#adv_doc_idx = tokenizer([adv_doc], return_tensors="pt")["input_ids"].to(device)
#adv_embeddings = word_embedding_layer(adv_doc_idx)
#adv_sim = F.cosine_similarity(F.normalize(adv_embeddings, p=2, dim=1), query_embeddings, dim=-1).mean()


#print("Adv SIM: ",adv_sim)
