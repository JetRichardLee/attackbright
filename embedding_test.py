import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
device = "cuda" # the device to load the model onto
from retrievers import attack_by_surrogate
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import numpy as np
from tools import gcg_step,get_nonascii_toks,gcg_step_show,gcg_step2,gcg_step_adq


def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embedding(text, model,tokenizer):
    batch_dict = tokenizer([text], max_length=4096, padding=True, truncation=True, return_tensors='pt').to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().detach().numpy()
    return embeddings.reshape(1,-1)



def get_embedding_C(text, model,tokenizer):

    doc_token = tokenizer([text], max_length=8192, padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
    doc_token = doc_token[0].reshape(1,-1)
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(doc_token).cpu().detach().to(dtype=torch.float32)
    #print(doc_embeddings.shape)
    return doc_embeddings[0,-1,:].reshape(1,-1)

def sample_query_Non_casual(doc,model,tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. A knowledge document content is provided followed by the user. Please generate a query with a practical scenario that (1) seems not directly related to the document content (2) the document would be helpful to solve the query."},
        {"role": "user", "content": doc}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    batch_dict = tokenizer([text], max_length=4096, padding=True, truncation=True, return_tensors='pt').to(model.device)
    outputs = model(**batch_dict)
    #print(outputs)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    #print(response)
    return response

def sample_query(doc,model,tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. A knowledge document content is provided followed by the user. Please generate a query with a practical scenario that (1) seems not directly related to the document content (2) the document would be helpful to solve the query."},
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
    #print(response)
    return response
def sample_query_q(query,doc,model,tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. A query targeting an enssetial topic is given by the user as following, and the corresponding document is also given, please generate some similar queries towarding the document."},
        {"role": "user", "content": "**Document**: "+doc},
        {"role": "user", "content": "**Query**: "+query}
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
    #print(response)
    return response

def attack_by_target(doc,o_query,t_model,t_tokenizer ,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100):
    #attacking a surrogate model without interaction with origianl
    if model==None:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-4B-Chat",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")
    max_length = 8192#kwargs.get('doc_max_length',8192)
    
    query_ts = []
    #print("start generating query")
    doc_embedding = get_embedding(doc,t_model,t_tokenizer)
    query_embedding = get_embedding(o_query,t_model,t_tokenizer)
    embeddings=[]
    embeddings_sim_d = []
    embeddings_sim_q = []
    embeddings_sim_q = []
    embeddings_sim_ad = []

    #doc_s_embedding = get_embedding_C(doc,model,tokenizer)
    #query_s_embedding = get_embedding_C(o_query,model,tokenizer)

    for i in range(num_q):
        query = sample_query(doc,model,tokenizer)
        #if i ==0:
        #    print(query)
        #query=o_query
        #print("Generate Query: ",query)
        embeddings.append(get_embedding(query,t_model,t_tokenizer))
        embeddings_sim_d.append(cosine_similarity(embeddings[-1],doc_embedding))
        embeddings_sim_q.append(cosine_similarity(embeddings[-1],query_embedding))
        query_token = t_tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
        query_ts.append(query_token[0])

    #print("query finish")
    forbidden_tokens = get_nonascii_toks(t_tokenizer)
    doc_token = t_tokenizer([doc], max_length=max_length, padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
    doc_token = doc_token[0].reshape(1,-1)
    st = doc_token.shape[1]
    sts_tokens = torch.full((1, num_sts_tokens), t_tokenizer.encode('*')[0]).to(model.device)  # Insert optimizable tokens
    ed = doc_token.shape[1]+sts_tokens.shape[1]
    adv_idxs = [_ for _ in range(st,ed)]
    #print(adv_idxs)
    adv_doc_t = torch.cat([doc_token,sts_tokens],dim=1)
    for e in range(epochs):
        model.eval()
        #doc_emb = model.encode(doc, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        #if e%100==0:
        #    adv_doc_t,min_loss = gcg_step_show(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4,tokenizer=tokenizer)
        #else:    
        adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, t_model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
       
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = t_tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    adv_doc_embedding = get_embedding(adv_doc,t_model,t_tokenizer)
    
    for i in range(num_q):
        embeddings_sim_ad.append(cosine_similarity(embeddings[i],adv_doc_embedding))

    print("Original Sim: ",cosine_similarity(query_embedding,doc_embedding))
    #print("doc shift: ", cosine_similarity(doc_embedding,doc_s_embedding))
    #print("query shift: ", cosine_similarity(query_embedding,query_s_embedding))
    print("/////////////")
    print("Doc-queries Sim: ", embeddings_sim_d)
    print("Avg: ", sum(embeddings_sim_d)/len(embeddings_sim_d))
    print("Min: ", min(embeddings_sim_d))
    print("/////////////")
    print("ADoc-queries Sim: ", embeddings_sim_ad)
    print("Avg: ", sum(embeddings_sim_ad)/len(embeddings_sim_ad))
    print("Min: ", min(embeddings_sim_ad))
    print("/////////////")
    print("Oq-queries Sim: ", embeddings_sim_q)
    print("Avg: ", sum(embeddings_sim_q)/len(embeddings_sim_q))
    print("Max: ", max(embeddings_sim_q))
    print("/////////////")

    print("Adv Sim: ",cosine_similarity(query_embedding,adv_doc_embedding))

    return adv_doc,adv_doc_t

def attack_by_surrogate(doc,o_query,t_model,t_tokenizer ,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100):
    #attacking a surrogate model without interaction with origianl
    if model==None:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-4B-Chat",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")
    max_length = 8192#kwargs.get('doc_max_length',8192)
    
    query_ts = []
    #print("start generating query")
    doc_embedding = get_embedding(doc,t_model,t_tokenizer)
    query_embedding = get_embedding(o_query,t_model,t_tokenizer)
    embeddings=[]
    embeddings_sim_d = []
    embeddings_sim_q = []
    embeddings_sim_q = []
    embeddings_sim_ad = []

    #doc_s_embedding = get_embedding_C(doc,model,tokenizer)
    #query_s_embedding = get_embedding_C(o_query,model,tokenizer)

    for i in range(num_q):
        query = sample_query_Non_casual(doc,model,tokenizer)
        #if i ==0:
        #    print(query)
        #query=o_query
        #print("Generate Query: ",query)
        embeddings.append(get_embedding(query,t_model,t_tokenizer))
        embeddings_sim_d.append(cosine_similarity(embeddings[-1],doc_embedding))
        embeddings_sim_q.append(cosine_similarity(embeddings[-1],query_embedding))
        query_token = tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
        query_ts.append(query_token[0])

    #print("query finish")
    forbidden_tokens = get_nonascii_toks(tokenizer)
    doc_token = tokenizer([doc], max_length=max_length, padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
    doc_token = doc_token[0].reshape(1,-1)
    st = doc_token.shape[1]
    sts_tokens = torch.full((1, num_sts_tokens), tokenizer.encode('*')[0]).to(model.device)  # Insert optimizable tokens
    ed = doc_token.shape[1]+sts_tokens.shape[1]
    adv_idxs = [_ for _ in range(st,ed)]
    #print(adv_idxs)
    adv_doc_t = torch.cat([doc_token,sts_tokens],dim=1)
    for e in range(epochs):
        model.eval()
        #doc_emb = model.encode(doc, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        #if e%100==0:
        #    adv_doc_t,min_loss = gcg_step_show(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4,tokenizer=tokenizer)
        #else:    
        adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
       
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    adv_doc_embedding = get_embedding(adv_doc,t_model,t_tokenizer)
    
    for i in range(num_q):
        embeddings_sim_ad.append(cosine_similarity(embeddings[i],adv_doc_embedding))

    print("Original Sim: ",cosine_similarity(query_embedding,doc_embedding))
    #print("doc shift: ", cosine_similarity(doc_embedding,doc_s_embedding))
    #print("query shift: ", cosine_similarity(query_embedding,query_s_embedding))
    print("/////////////")
    print("Doc-queries Sim: ", embeddings_sim_d)
    print("Avg: ", sum(embeddings_sim_d)/len(embeddings_sim_d))
    print("Min: ", min(embeddings_sim_d))
    print("/////////////")
    print("ADoc-queries Sim: ", embeddings_sim_ad)
    print("Avg: ", sum(embeddings_sim_ad)/len(embeddings_sim_ad))
    print("Min: ", min(embeddings_sim_ad))
    print("/////////////")
    print("Oq-queries Sim: ", embeddings_sim_q)
    print("Avg: ", sum(embeddings_sim_q)/len(embeddings_sim_q))
    print("Max: ", max(embeddings_sim_q))
    print("/////////////")

    print("Adv Sim: ",cosine_similarity(query_embedding,adv_doc_embedding))

    return adv_doc,adv_doc_t

#"""
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-4B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

#prompt = "today is a good day. Any activity suggested?"
#doc = "salt could be used to make snow melt faster"
with open('doc.txt', 'r') as file:
    doc = file.read()
#print(content)
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
query = "In what way do developed countries steal trillions of dollars from developing countries?\n According to the Guardian, rich Western countries 'steal' large amounts of money from poor countries, much more than they give in development aid.\n If we add theft through trade in services to the mix, it brings total net resource outflows to about $3tn per year. That’s 24 times more than the aid budget. In other words, for every $1 of aid that developing countries receive, they lose $24 in net outflows. These outflows strip developing countries of an important source of revenue and finance for development. The GFI report finds that increasingly large net outflows have caused economic growth rates in developing countries to decline, and are directly responsible for falling living standards.\n I would like to know in what way the money is ‘stolen’, if this is illegal and if this is indeed ‘directly responsible for falling living standards’."
print("Original Query: ", query)


t_tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
t_model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral',device_map="auto").eval()
#max_length = kwargs.get('doc_max_length',4096)

#t_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-4B-Chat",torch_dtype="auto",device_map="auto")
#t_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")


adv_doc,_ = attack_by_target(doc,query,t_model,t_tokenizer, num_sts_tokens=10,num_q=10,epochs=100)
#print("Adv Document: ",adv_doc)

