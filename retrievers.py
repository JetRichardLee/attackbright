import os.path
import time
import torch
import json
import cohere
import numpy as np
import vertexai
import pytrec_eval
import tiktoken
import voyageai
from tqdm import tqdm,trange
import torch.nn.functional as F
from gritlm import GritLM
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from InstructorEmbedding import INSTRUCTOR
from numpy.random import randint
from random import sample
from peft import LoraConfig, get_peft_model

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from tools import gcg_step,get_nonascii_toks,gcg_step_show,gcg_step_adq,gcg_step_avg,gcg_step_adq2,gcg_step_adq3,gcg_step_adq4,gcg_step_adq4_tk,gcg_step_adq3_tk
from tools import gcg_step_avg_h,gcg_step_adq3_h,gcg_step_adq4_h

def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def get_embedding_google(texts,task,model,dimensionality=768):
    success = False
    while not success:
        try:
            new_texts = []
            for t in texts:
                if t.strip()=='':
                    print('empty content')
                    new_texts.append('empty')
                else:
                    new_texts.append(t)
            texts = new_texts
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
            embeddings = model.get_embeddings(inputs, **kwargs)
            success = True
        except Exception as e:
            print(e)
    return [embedding.values for embedding in embeddings]

def get_embedding_openai(texts, openai_client,tokenizer,model="text-embedding-3-large"):
    texts =[json.dumps(text.replace("\n", " ")) for text in texts]
    success = False
    threshold = 6000
    count = 0
    cur_emb = None
    exec_count = 0
    while not success:
        exec_count += 1
        if exec_count>5:
            print('execute too many times')
            exit(0)
        try:
            emb_obj = openai_client.embeddings.create(input=texts, model=model).data
            cur_emb = [e.embedding for e in emb_obj]
            success = True
        except Exception as e:
            print(e)
            count += 1
            threshold -= 500
            if count>4:
                print('openai cut',count)
                exit(0)
            new_texts = []
            for t in texts:
                new_texts.append(cut_text_openai(text=t, tokenizer=tokenizer,threshold=threshold))
            texts = new_texts
    if cur_emb is None:
        raise ValueError("Fail to embed, openai")
    return cur_emb

TASK_MAP = {
    'biology': 'Biology',
    'earth_science': 'Earth Science',
    'economics': 'Economics',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'Stack Overflow',
    'sustainable_living': 'Sustainable Living',
}

def add_instruct_concatenate(texts,task,instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_list(texts,task,instruction):
    return [[instruction.format(task=task),t] for t in texts]

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores

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

def sample_query_div(doc,model,tokenizer):

    div = randint(1,5)
    if div ==1:
        messages = [
            {"role": "system", "content": "You are a helpful query writer for information retrieval system. A knowledge document content is provided followed by the user. Please generate a query with content in which the human asker is facing a practical problem and the document would be helpful to solve the it."},
         {"role": "user", "content": doc}
        ]
    elif div==2:
        messages = [
            {"role": "system", "content": "You are a helpful query writer for information retrieval system. A knowledge document content is provided followed by the user. Please generate a query by questioning some factors of the document content."},
         {"role": "user", "content": doc}
        ]
    elif div==3:
        messages = [
            {"role": "system", "content": "You are a helpful query writer for information retrieval system. A knowledge document content is provided followed by the user. Please generate a query targeting the document and containing five keywords from the document."},
         {"role": "user", "content": doc}
        ]
    elif div==4:
        messages = [
            {"role": "system", "content": "You are a helpful query writer for information retrieval system. A knowledge document content is provided followed by the user. Please generate a one-sentence summary for this document."},
         {"role": "user", "content": doc}
        ]
    elif div==5:
        messages = [
            {"role": "system", "content": "You are a helpful query writer for information retrieval system. A knowledge document content is provided followed by the user. Please generate a query that has similar semantics but contains few overlaps with the document."},
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

def sample_query_out(doc,model,tokenizer):

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
    #print(response)
    return response


def L_finetune(emb_in,emb_out):
    #avg_in = torch.mean(torch.stack(emb_in), dim=0).detach()
    loss = torch.zeros(1).to(emb_in[0].device)
    #for emb in emb_in:
    #    loss -= pairwise_cosine_similarity(emb,avg_in)[0]/len(emb_in)
    for emb1 in emb_in:
        for emb2 in emb_out:
            loss += pairwise_cosine_similarity(emb1,emb2)[0]/(len(emb_in)*len(emb_out))
    for emb1 in emb_in:
        for emb2 in emb_in:
            loss -= pairwise_cosine_similarity(emb1,emb2)[0]/(len(emb_in)*len(emb_in))
    return loss

def finetune_doc(model,tokenizer,doc_in,doc_out,num_epochs=3, batch_size=1, lr=5e-5):
    max_length = 8192

    lora_config = LoraConfig(
        r=8,                   # rank of low-rank update
        lora_alpha=16,          # scaling
        target_modules=["q_proj", "v_proj"],  # only attention query/value matrices
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model.train()
    model = get_peft_model(model, lora_config)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    word_embedding_layer = model.get_input_embeddings()
    for _ in range(num_epochs):
        doc_emb_in = []

        for doc in doc_in:
            doc_token_in = tokenizer([doc], max_length=max_length, padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
            doc_token_in = doc_token_in[0].reshape(1,-1)
        
            outputs = model(doc_token_in,output_hidden_states=True)  # shape: [batch_size, seq_length, hidden_dim]
            last_hidden_state = outputs.hidden_states[-1]
            doc_emb_in.append(last_hidden_state[:,-1])

        doc_emb_out = []

        for doc in doc_out:
            doc_token_out = tokenizer([doc], max_length=max_length, padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
            doc_token_out = doc_token_out[0].reshape(1,-1)
            outputs = model(doc_token_out,output_hidden_states=True)  # shape: [batch_size, seq_length, hidden_dim]

            last_hidden_state = outputs.hidden_states[-1]
            doc_emb_out.append(last_hidden_state[:,-1])
        loss = L_finetune(doc_emb_in,doc_emb_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model = model.merge_and_unload()
    for param in model.parameters():
        param.requires_grad = True
    #model.eval()
    #print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss.item()}")
    return model

def attack_by_surrogate(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100):
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
    for i in range(num_q):
        query = sample_query(doc,model,tokenizer)
        #print(query)
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
        #adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4)
        #adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        adv_doc_t,min_loss = gcg_step_avg_h(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    return adv_doc,adv_doc_t

def attack_by_surrogate_div(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=20):
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
    for i in range(num_q):
        query = sample_query_div(doc,model,tokenizer)
        #print(query)
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
        #adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4)
        #adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        adv_doc_t,min_loss = gcg_step_avg_h(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        torch.cuda.empty_cache()
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    return adv_doc,adv_doc_t
    
def attack_by_original(doc,query_o,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100,q_rate=0.3):
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
    for i in range(num_q):
        query = query_o#sample_query(doc,model,tokenizer)
        #print(query)
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
    return adv_doc,adv_doc_t

    
def attack_by_advl(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100,q_rate=0.3):
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
    for i in range(num_q):
        #query = sample_query(doc,model,tokenizer)
        query = sample_query_div(doc,model,tokenizer)
        #print(query)
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
    q_list=[_ for _ in range(num_q)]
    
    for e in range(5):
        #query_ts,_ = gcg_step_adq3(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        query_ts,_ = gcg_step_adq3_h(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

    for e in range(epochs):
        model.eval()
        #doc_emb = model.encode(doc, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        #if e%100==0:
        #    adv_doc_t,min_loss = gcg_step_show(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4,tokenizer=tokenizer)
        #else:    
        #adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)  
        adv_doc_t,min_loss = gcg_step_avg_h(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

        #adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #q_index= sample(q_list,int(q_rate*num_q))
        if e%25==13:
            query_ts,_ = gcg_step_adq4_h(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
            #query_ts,_ = gcg_step_adq4(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #query_ts,_ = gcg_step_adq3(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #query_ts,_ = gcg_step_adq2(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #for q in q_index:
        #    query_ts[q],_ = gcg_step_adq(adv_doc_t,query_ts[q], model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    return adv_doc,adv_doc_t

    
    
def attack_by_advl_ft(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100,q_rate=0.3,N_FT=50):
    #attacking a surrogate model without interaction with origianl
    if model==None:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-4B-Chat",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")
    max_length = 8192#kwargs.get('doc_max_length',8192)
    
    ft_querys_in = []
    for i in range(N_FT):
        #query = sample_query(doc,model,tokenizer)
        ft_query = sample_query_div(doc,model,tokenizer)
        ft_querys_in.append(ft_query)

    ft_querys_out = []
    for i in range(N_FT):
        #query = sample_query(doc,model,tokenizer)
        ft_query = sample_query_out(doc,model,tokenizer)
        ft_querys_out.append(ft_query)
    
    model = finetune_doc(model,tokenizer,ft_querys_in,ft_querys_out,num_epochs=1, batch_size=1, lr=5e-4)    

    
    query_ts = []
    #print("start generating query")
    for i in range(num_q):
        #query = sample_query(doc,model,tokenizer)
        query = sample_query_div(doc,model,tokenizer)
        #print(query)
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
    q_list=[_ for _ in range(num_q)]
    
    for e in range(5):
        query_ts,_ = gcg_step_adq3_h(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

    for e in range(epochs):
        model.eval()
        #doc_emb = model.encode(doc, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        #if e%100==0:
        #    adv_doc_t,min_loss = gcg_step_show(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4,tokenizer=tokenizer)
        #else:    
        adv_doc_t,min_loss = gcg_step_avg_h(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #q_index= sample(q_list,int(q_rate*num_q))
        if e%25==13:
            query_ts,_ = gcg_step_adq4_h(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #query_ts,_ = gcg_step_adq3(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #query_ts,_ = gcg_step_adq2(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #for q in q_index:
        #    query_ts[q],_ = gcg_step_adq(adv_doc_t,query_ts[q], model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    return adv_doc,adv_doc_t

def attack_by_advl_tokens(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100,q_rate=0.3):
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
    adv_qidxs = []

    #print("start generating query")
    for i in range(num_q):
        query = sample_query(doc,model,tokenizer)
        #print(query)
        query_token = tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
        st = query_token[0].shape[0]
        sts_tokens = torch.full((1,num_sts_tokens), tokenizer.encode('*')[0])[0].to(model.device) 
        ed = query_token[0].shape[0]+sts_tokens.shape[0]
        query_ts.append(torch.cat([query_token[0],sts_tokens],dim=0))
        adv_qidxs.append([_ for _ in range(st,ed)])
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
    q_list=[_ for _ in range(num_q)]
    

    sts_tokens = torch.full((1, num_sts_tokens), tokenizer.encode('*')[0]).to(model.device)


    #for e in range(10):
    #    query_ts,_ = gcg_step_adq3_tk(adv_doc_t,query_ts,adv_qidxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

    for e in range(epochs):
        model.eval()
        #doc_emb = model.encode(doc, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        #if e%100==0:
        #    adv_doc_t,min_loss = gcg_step_show(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4,tokenizer=tokenizer)
        #else:    
        #adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

        q_index= sample(q_list,int(q_rate*num_q))
        #query_ts,_ = gcg_step_adq4_tk(adv_doc_t,query_ts,adv_qidxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #query_ts,_ = gcg_step_adq3(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #query_ts,_ = gcg_step_adq2(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #for q in q_index:
        #    query_ts[q],_ = gcg_step_adq(adv_doc_t,query_ts[q], model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    return adv_doc,adv_doc_t


def attack_by_surrogate_emb(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100):
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
    for i in range(num_q):
        query = sample_query(doc,model,tokenizer)
        #print(query)
        query_token = tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
        query_ts.append(query_token[0])
    
    word_embedding_layer = model.get_input_embeddings()
    query_embs = [word_embedding_layer(query_token.reshape(1,-1))[:,-1].detach() for query_token in query_ts]
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
        #adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4)
        adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
       
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    adv_emb = word_embedding_layer(adv_doc_t.reshape(1,-1))[:,-1].detach()
    return adv_doc,adv_doc_t,query_embs,adv_emb

def attack_by_adiv_emb(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100):
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
    for i in range(num_q):
        query = sample_query_div(doc,model,tokenizer)
        #print(query)
        query_token = tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
        query_ts.append(query_token[0])
    
    word_embedding_layer = model.get_input_embeddings()
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
        #adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4)
        adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    adv_emb = word_embedding_layer(adv_doc_t.reshape(1,-1))[:,-1].detach()
    query_embs = [word_embedding_layer(query_token.reshape(1,-1))[:,-1].detach() for query_token in query_ts]
    return adv_doc,adv_doc_t,query_embs,adv_emb

def attack_by_adv_emb(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100):
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
    for i in range(num_q):
        query = sample_query_div(doc,model,tokenizer)
        #print(query)
        query_token = tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
        query_ts.append(query_token[0])
    
    word_embedding_layer = model.get_input_embeddings()
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
    
    for e in range(5):
        query_ts,_ = gcg_step_adq3(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)

    for e in range(epochs):
        model.eval()
        #doc_emb = model.encode(doc, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        #if e%100==0:
        #    adv_doc_t,min_loss = gcg_step_show(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4,tokenizer=tokenizer)
        #else:    
        #adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4)
        adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        if e%25==13:
            query_ts,_ = gcg_step_adq4(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    adv_emb = word_embedding_layer(adv_doc_t.reshape(1,-1))[:,-1].detach()
    query_embs = [word_embedding_layer(query_token.reshape(1,-1))[:,-1].detach() for query_token in query_ts]
    return adv_doc,adv_doc_t,query_embs,adv_emb

def attack_by_adv2_emb(doc,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100):
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
    for i in range(num_q):
        query = sample_query(doc,model,tokenizer)
        #print(query)
        query_token = tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
        query_ts.append(query_token[0])
    
    word_embedding_layer = model.get_input_embeddings()
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
        #adv_doc_t,min_loss = gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=12, batch_size=4)
        adv_doc_t,min_loss = gcg_step_avg(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
       
        query_ts,_ = gcg_step_adq3(adv_doc_t,query_ts, model, forbidden_tokens, top_k=10, num_samples=16, batch_size=4)
        #if e %100==0:
        #print(adv_doc_t)
            #adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
            #print(f"epoch {e}- loss:{min_loss} adv_doc:{adv_doc}")
    adv_doc = tokenizer.batch_decode(adv_doc_t, skip_special_tokens=True)[0]
    adv_emb = word_embedding_layer(adv_doc_t.reshape(1,-1))[:,-1].detach()
    query_embs = [word_embedding_layer(query_token.reshape(1,-1))[:,-1].detach() for query_token in query_ts]
    return adv_doc,adv_doc_t,query_embs,adv_emb

def attack_by_original_emb(doc,query_o,model=None,tokenizer=None,num_sts_tokens=10,num_q=10, epochs=100,q_rate=0.3):
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

    query = query_o#sample_query(doc,model,tokenizer)
    #print(query)
    query_token = tokenizer([query], padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
    query_ts.append(query_token[0])
    
    word_embedding_layer = model.get_input_embeddings()
    query_emb = word_embedding_layer(query_token.reshape(1,-1))[:,-1].detach()

    #print("query finish")
    forbidden_tokens = get_nonascii_toks(tokenizer)
    doc_token = tokenizer([doc], max_length=max_length, padding=True, truncation=True, return_tensors='pt')["input_ids"].to(model.device)
    doc_token = doc_token[0].reshape(1,-1)
    doc_emb = word_embedding_layer(doc_token)[:,-1].detach()
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
    adv_emb = word_embedding_layer(adv_doc_t.reshape(1,-1))[:,-1].detach()
    return adv_doc,adv_doc_t,query_emb,doc_emb,adv_emb


























def model_score(tokenizer,model,doc,query):
    messages = [
    {"role": "system", "content": "You are a helpful assistant. I'm giving you a document and a query, please answer 'yes' or 'no' to indicate if they are related."},
    {"role": "user", "content": f"Query: {query}"},
    {"role": "user", "content": f"Document: {doc}"}
    ]
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    yes_id =  tokenizer(["yes"], return_tensors="pt").input_ids

    yes_id =  yes_id[0]
    no_id =  tokenizer(["no"], return_tensors="pt").input_ids
    no_id = no_id[0]
    #print(yes_id)
    #print(no_id)
    #print(model_inputs)

    with torch.no_grad():
        outputs = model(**model_inputs)

    # Get logits for next token (the last position)
    #print("shapes:")
    #print(outputs.logits.shape)
    next_token_logits = outputs.logits[:, -1, :]  # shape [1, vocab_size]

    # Convert logits to probabilities
    probs = F.softmax(next_token_logits, dim=-1)
    #print(probs.shape)
    #print(probs[0,yes_id])
    #print(probs[0,no_id])
    return probs[0,yes_id]/(probs[0,yes_id]+probs[0,no_id])

@torch.no_grad()
def get_scores_sf_qwen_e5(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
        model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral',device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model = model.eval()
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('encode_batch_size',1)

    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    #if os.path.isfile(cache_path):
        # already exists so we can just load it
    #    doc_emb = np.load(cache_path, allow_pickle=True)
    

    # need a function of :
    # docs = optimized(model, documents, g_id)


    for start_idx in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue
        #print(documents[start_idx:start_idx+batch_size])
        batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        # doc_emb[start_idx] = embeddings
        doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, np.array(embeddings)), axis=0)

        # save the embeddings every 1000 iters, you can adjust this as needed
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
        
    np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    #print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    #print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return query_emb,doc_emb,scores

@torch.no_grad()
def retrieval_sf_qwen_e5(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
        model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral',device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model = model.eval()
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('encode_batch_size',1)

    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    #if os.path.isfile(cache_path):
        # already exists so we can just load it
    #    doc_emb = np.load(cache_path, allow_pickle=True)
    

    # need a function of :
    # docs = optimized(model, documents, g_id)


    for start_idx in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue
        #print(documents[start_idx:start_idx+batch_size])
        batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        # doc_emb[start_idx] = embeddings
        doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, np.array(embeddings)), axis=0)

        # save the embeddings every 1000 iters, you can adjust this as needed
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
        
    np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    #print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    #print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_bm25(queries,query_ids,documents,doc_ids,excluded_ids,long_context,**kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    all_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        for did, s in zip(doc_ids, similarities):
            all_scores[str(query_id)][did] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                all_scores[str(query_id)].pop(did)
        cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
    return all_scores

@torch.no_grad()
def retrieval_sbert_bge(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='bge':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    elif model_id=='sbert':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    batch_size = kwargs.get('batch_size',128)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_instructor(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='inst-l':
        model = SentenceTransformer('hkunlp/instructor-large')
    elif model_id=='inst-xl':
        model = SentenceTransformer('hkunlp/instructor-xl')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model.set_pooling_include_prompt(False)

    batch_size = kwargs.get('batch_size',4)
    model.max_seq_length = kwargs.get('doc_max_length',2048)
    # queries = add_instruct_list(texts=queries,task=task,instruction=instructions['query'])
    # documents = add_instruct_list(texts=documents,task=task,instruction=instructions['document'])

    query_embs = model.encode(queries,batch_size=batch_size,show_progress_bar=True,prompt=instructions['query'].format(task=task),normalize_embeddings=True)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_embs = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_embs = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True,prompt=instructions['document'].format(task=task))
        np.save(cur_cache_file, doc_embs)
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_grit(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'GritLM/GritLM-7B'
    else:
        print('use',customized_checkpoint)
    model = GritLM(customized_checkpoint, torch_dtype="auto", mode="embedding")
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length',256)
    doc_max_length = kwargs.get('doc_max_length',2048)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file, allow_pickle=True)
    elif ignore_cache:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
    else:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_openai(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q,tokenizer=tokenizer))
    queries = new_queries
    new_documents = []
    for d in documents:
        new_documents.append(cut_text_openai(text=d,tokenizer=tokenizer))
    documents = new_documents
    doc_emb = []
    batch_size = kwargs.get('batch_size',1024)
    # openai_client = OpenAI(api_key=kwargs['key'])
    openai_client = OpenAI()
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            cur_emb = get_embedding_openai(texts=documents[idx:idx + batch_size],openai_client=openai_client,tokenizer=tokenizer)
            with open(cur_cache_file,'w') as f:
                json.dump(cur_emb,f,indent=2)
        doc_emb += cur_emb
    query_emb = []
    for idx in trange(0, len(queries), batch_size):
        cur_emb = get_embedding_openai(texts=queries[idx:idx + batch_size], openai_client=openai_client,
                                       tokenizer=tokenizer)
        query_emb += cur_emb
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_cohere(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    query_emb = []
    doc_emb = []
    batch_size = kwargs.get('batch_size',8192)
    # cohere_client = cohere.Client(kwargs['key'])
    cohere_client = cohere.Client()
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            success = False
            exec_count = 0
            cur_emb = []
            while not success:
                exec_count += 1
                if exec_count>5:
                    print('cohere execute too many times')
                    exit(0)
                try:
                    cur_emb = cohere_client.embed(texts=documents[idx:idx+batch_size], input_type="search_document",
                                                  model="embed-english-v3.0").embeddings

                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(60)
            with open(cur_cache_file, 'w') as f:
                json.dump(cur_emb, f, indent=2)
        doc_emb += cur_emb
    for idx in trange(0, len(queries), batch_size):
        success = False
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('cohere query execute too many times')
                exit(0)
            try:
                cur_emb = cohere_client.embed(queries[idx:idx+batch_size], input_type="search_query",
                                              model="embed-english-v3.0").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                time.sleep(60)
    scores = (torch.tensor(query_emb) @ torch.tensor(doc_emb).T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_voyage(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    new_queries = []
    for q in queries:
        new_queries.append(cut_text(text=q,tokenizer=tokenizer,threshold=16000))
    queries = new_queries
    new_documents = []
    for d in tqdm(documents,desc='preprocess documents'):
        new_documents.append(cut_text(text=d,tokenizer=tokenizer,threshold=16000))
    documents = new_documents

    query_emb = []
    doc_emb = []

    batch_size = kwargs.get('batch_size',1)

    doc_emb = None
    doc_cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(doc_cache_path), exist_ok=True)
    if os.path.isfile(doc_cache_path):
        # already exists so we can just load it
        doc_emb = np.load(doc_cache_path, allow_pickle=True)

    # voyage_client = voyageai.Client(api_key=kwargs['key'])
    voyage_client = voyageai.Client()
    for i in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > i:
            continue
        
        success = False
        threshold = 16000
        cur_texts = documents[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage document too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="document").embeddings
                doc_emb = cur_emb if doc_emb is None else np.concatenate((doc_emb, np.array(cur_emb)), axis=0)
                if (i + 1) % 1000 == 0:
                    np.save(doc_cache_path, doc_emb)
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(5)

    query_emb = []
    for i in trange(0,len(queries),batch_size):
        success = False
        threshold = 16000
        cur_texts = queries[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage query execute too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="query").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(60)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_google(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0409")
    query_emb = []
    # doc_emb = []
    batch_size = kwargs.get('batch_size',8)
    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)

    for start_idx in tqdm(range(0, len(documents), batch_size), desc='embedding'):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue
        
        cur_emb = get_embedding_google(
            texts=documents[start_idx:start_idx + batch_size], task='RETRIEVAL_DOCUMENT',
            model=model
        )
        doc_emb = cur_emb if doc_emb is None else np.concatenate((doc_emb, np.array(cur_emb)), axis=0)
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
    np.save(cache_path, doc_emb)
        
    for start_idx in tqdm(range(0,len(queries), batch_size),desc='embedding'):
        query_emb += get_embedding_google(texts=queries[start_idx:start_idx+ batch_size],task='RETRIEVAL_QUERY',model=model)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


RETRIEVAL_FUNCS = {
    'sf': retrieval_sf_qwen_e5,
    'qwen': retrieval_sf_qwen_e5,
    'qwen2': retrieval_sf_qwen_e5,
    'e5': retrieval_sf_qwen_e5,
    'bm25': retrieval_bm25,
    'sbert': retrieval_sbert_bge,
    'bge': retrieval_sbert_bge,
    'inst-l': retrieval_instructor,
    'inst-xl': retrieval_instructor,
    'grit': retrieval_grit,
    'cohere': retrieval_cohere,
    'voyage': retrieval_voyage,
    'openai': retrieval_openai,
    'google': retrieval_google
}

def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output
