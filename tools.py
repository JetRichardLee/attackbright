
import torch
import torch.nn.functional as F
from math import ceil

from numpy.random import randint

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def target_loss(embeddings, model, tokenizer, target_sequence):
    """
    Computes the loss of the target sequence given the embeddings of the input sequence.
    Args:
        embeddings: Embeddings of the input sequence. Contains embeddings for the prompt and the adversarial sequence.
                    Shape: (batch_size, sequence_length, embedding_dim).
        model: LLM model. Type: AutoModelForCausalLM.
        tokenizer: Tokenizer.
        target_sequence: Target sequence. e.g., "Sure, here is ..."
        first_token_idx: Index of the first token of the target sequence. Used to remove special tokens added
                         by the tokenizer.
        last_token_idx: Index of the last token of the target sequence from the end, i.e., the index from the front
                        is equal to length of the sequence - last_token_idx. Used to remove special tokens added by
                        the tokenizer.
    Returns:
        loss: A tensor containing the loss of the target sequence for each batch item.
    """

    # Get device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device

    # Tokenize target sequence and get embeddings
    target_tokens = tokenizer(target_sequence, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    word_embedding_layer = model.get_input_embeddings()
    target_embeddings = word_embedding_layer(target_tokens)

    # Concatenate embeddings
    target_embeddings = target_embeddings.expand(embeddings.shape[0], -1, -1)
    sequence_embeddings = torch.cat((embeddings, target_embeddings), dim=1)

    # Get logits from model
    sequence_logits = model(inputs_embeds=sequence_embeddings).logits

    # Compute loss for each batch item
    loss_fn = torch.nn.CrossEntropyLoss() # (reduction="sum")
    loss = []
    
    for i in range(embeddings.shape[0]):
        loss.append(loss_fn(sequence_logits[i, embeddings.shape[1]-1:-1, :], target_tokens[0]))

    return torch.stack(loss)

def cosine_similarity_loss_avg(a, bs):
    loss = 0.0
    for b in bs:
        #print("///////")
        #print(F.normalize(a, p=2, dim=1).shape)
        #print(b.shape)
        cos_sim = F.cosine_similarity(F.normalize(a, p=2, dim=1), b, dim=-1)
        loss+=cos_sim.mean()  # maximize cosine similarity
    
    return loss/len(bs)

def cosine_similarity_loss(a, bs):
    loss = 0.0
    for b in bs:
        #print("///////")
        #print(F.normalize(a, p=2, dim=1).shape)
        #print(b.shape)
        cos_sim = F.cosine_similarity(F.normalize(a, p=2, dim=1), b, dim=-1)
        loss+=cos_sim.mean()  # maximize cosine similarity
    
    return loss

def cosine_similarity_loss_multi(a, bs):
    loss = 0.0
    for b in bs:
        #print("///////")
        #print(F.normalize(a, p=2, dim=1).shape)
        #print(b.shape)
        cos_sim = F.cosine_similarity(F.normalize(a, p=2, dim=1).mean(dim=1), b, dim=-1)
        loss+=cos_sim.mean()  # maximize cosine similarity
    
    return loss

def cosine_similarity_loss_sin(a, b):
    #loss = 0.0
    #print(a.shape)
    #print(b.shape)
    cos_sim = F.cosine_similarity(F.normalize(a, p=2, dim=1).mean(dim=0), b, dim=-1)
    loss=cos_sim.mean()  # maximize cosine similarity
    
    return loss

def Q_loss(tensor_list, lambda_center=1.0, lambda_diverse=1.0):
    """
    Custom loss that encourages tensors to stay close to the center (origin) and become diverse between them.

    Parameters:
    - tensors (torch.Tensor): A batch of tensors of shape (N, D), where N is the number of tensors and D is the tensor dimension.
    - lambda_center (float): Weight for the center loss.
    - lambda_diverse (float): Weight for the diversity loss.

    Returns:
    - loss (torch.Tensor): The combined loss.
    """
    tensors = [ten.mean(dim=1).reshape(1,-1) for ten in tensor_list]
    tensors = torch.concat(tensors)
    N, D = tensors.shape
    #print(N,D)
    # Center loss: Encourage tensors to stay close to the origin (zero vector)
    #center_loss = torch.norm(tensors, p=2, dim=1).pow(2).mean()  # Sum of squared L2 norms
    mean_tensor = tensors.mean(dim=0)
    
    # Center loss: Encourage tensors to stay close to the mean tensor
    center_loss = torch.norm(tensors - mean_tensor, p=2, dim=1).pow(2).mean()  # Sum of squared L2 norms

    # Diversity loss: Encourage tensors to become diverse (minimize cosine similarity)
    diversity_loss = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Cosine similarity between tensors[i] and tensors[j]
            cos_sim = F.cosine_similarity(tensors[i].unsqueeze(0), tensors[j].unsqueeze(0))
            diversity_loss += cos_sim
    
    diversity_loss = diversity_loss / (N * (N - 1) / 2)  # Normalize by the number of pairs

    # Total loss
    loss = lambda_center * center_loss + lambda_diverse * diversity_loss  # Subtract cosine similarity for maximization

    return loss

def Q_loss2(tensor_list, lambda_center=0.0, lambda_diverse=1.0):
    """
    Custom loss that encourages tensors to stay close to the center (origin) and become diverse between them.

    Parameters:
    - tensors (torch.Tensor): A batch of tensors of shape (N, D), where N is the number of tensors and D is the tensor dimension.
    - lambda_center (float): Weight for the center loss.
    - lambda_diverse (float): Weight for the diversity loss.

    Returns:
    - loss (torch.Tensor): The combined loss.
    """
    tensors = [ten.mean(dim=1).reshape(1,-1) for ten in tensor_list]
    tensors = torch.concat(tensors)
    N, D = tensors.shape
    #print(N,D)
    # Center loss: Encourage tensors to stay close to the origin (zero vector)
    #center_loss = torch.norm(tensors, p=2, dim=1).pow(2).mean()  # Sum of squared L2 norms
    mean_tensor = tensors.mean(dim=0)
    
    # Center loss: Encourage tensors to stay close to the mean tensor
    center_loss = torch.norm(tensors - mean_tensor, p=2, dim=1).pow(2).mean()  # Sum of squared L2 norms

    # Diversity loss: Encourage tensors to become diverse (minimize cosine similarity)
    diversity_loss = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Cosine similarity between tensors[i] and tensors[j]
            cos_sim = F.cosine_similarity(tensors[i].unsqueeze(0), tensors[j].unsqueeze(0))
            diversity_loss += 1/(1+1e-6-cos_sim)
    
    diversity_loss = diversity_loss / (N * (N - 1) / 2)  # Normalize by the number of pairs

    # Total loss
    loss = lambda_center * center_loss + lambda_diverse * diversity_loss  # Subtract cosine similarity for maximization

    return loss

def gcg_step2(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = len(adv_idxs)
    #print(num_adv)
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data
    print(adv_doc_t.shape)
    # Get word embeddings for input sequence
    doc_embeddings = model(input_ids=adv_doc_t.reshape(1,-1)).last_hidden_state#word_embedding_layer(adv_doc_t)
    print(doc_embeddings.shape)
    doc_embeddings=doc_embeddings[:, -1]
    doc_embeddings.retain_grad()

    query_embeddings = [model(input_ids=query_t.reshape(1,-1)).last_hidden_state[:, -1] for query_t in query_ts]

    # Get loss and gradients
    loss = 0-cosine_similarity_loss(doc_embeddings, query_embeddings)
    loss.backward()  # Minimize loss

    gradients = doc_embeddings.grad
    # Dot product of gradients and embedding matrix
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod[:,forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, top_k).indices)[adv_idxs]

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_item = adv_doc_t.clone().detach()
            rand_adv_idx = randint(0, num_adv)
            #print(rand_adv_idx)
            random_token_idx = randint(0, top_k)
            batch_item[0, adv_idxs[rand_adv_idx]] = top_k_adv[rand_adv_idx, random_token_idx]
            sequence_batch.append(batch_item)
            batch_loss.append(torch.unsqueeze(cosine_similarity_loss(word_embedding_layer(batch_item), query_embeddings),dim=0))

        sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index].unsqueeze(0)

    return adv_seq, min_loss

    
def gcg_step(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = len(adv_idxs)
    #print(num_adv)
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)
    #print(doc_embeddings.requires_grad)# = True
    doc_embeddings.retain_grad()

    #query_embeddings = [word_embedding_layer(query_t)[-1] for query_t in query_ts]
    query_embeddings = [word_embedding_layer(query_t)[-1] for query_t in query_ts]
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-cosine_similarity_loss(doc_embeddings, query_embeddings)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    gradients = doc_embeddings.grad
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod[:,forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, top_k).indices)[adv_idxs]

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_item = adv_doc_t.clone().detach()
            rand_adv_idx = randint(0, num_adv)
            #print(rand_adv_idx)
            random_token_idx = randint(0, top_k)
            batch_item[0, adv_idxs[rand_adv_idx]] = top_k_adv[rand_adv_idx, random_token_idx]
            sequence_batch.append(batch_item)
            batch_loss.append(torch.unsqueeze(cosine_similarity_loss(word_embedding_layer(batch_item), query_embeddings),dim=0))

        sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index].unsqueeze(0)

    return adv_seq, min_loss

def gcg_step_avg(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = len(adv_idxs)
    #print(num_adv)
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)
    #print(doc_embeddings.requires_grad)# = True
    doc_embeddings.retain_grad()

    #query_embeddings = [word_embedding_layer(query_t)[-1] for query_t in query_ts]
    query_embeddings = [word_embedding_layer(query_t)[-1] for query_t in query_ts]
    #print("d shape:",doc_embeddings.shape)
    #print("q shape:",query_embeddings[0].shape)
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-cosine_similarity_loss_avg(doc_embeddings, query_embeddings)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    gradients = doc_embeddings.grad
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod[:,forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, top_k).indices)[adv_idxs]

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_item = adv_doc_t.clone().detach()
            rand_adv_idx = randint(0, num_adv)
            #print(rand_adv_idx)
            random_token_idx = randint(0, top_k)
            batch_item[0, adv_idxs[rand_adv_idx]] = top_k_adv[rand_adv_idx, random_token_idx]
            sequence_batch.append(batch_item)
            batch_loss.append(torch.unsqueeze(cosine_similarity_loss_avg(word_embedding_layer(batch_item), query_embeddings),dim=0))

        sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index].unsqueeze(0)

    return adv_seq, min_loss






def gcg_step_avg_h(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = len(adv_idxs)
    #print(num_adv)
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)
    #print(doc_embeddings.requires_grad)# = True
    doc_embeddings.retain_grad()


    outputs = model(inputs_embeds=doc_embeddings,output_hidden_states=True)  # shape: [batch_size, seq_length, hidden_dim]

    last_hidden_state = outputs.hidden_states[-1]
    doc_hidden = last_hidden_state[:,-1]

    #query_embeddings = [word_embedding_layer(query_t)[-1] for query_t in query_ts]
    query_embeddings = [word_embedding_layer(query_t) for query_t in query_ts]

    query_hiddens = []
    for query_emb in query_embeddings:
        outputs = model(inputs_embeds=query_emb.unsqueeze(0),output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        query_hiddens.append(last_hidden_state[:,-1])


    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-cosine_similarity_loss_avg(doc_hidden, query_hiddens)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    gradients = doc_embeddings.grad
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod[:,forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, top_k).indices)[adv_idxs]

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []
        with torch.no_grad():
            for _ in range(this_batch_size):
                torch.cuda.empty_cache()
                batch_item = adv_doc_t.clone().detach()
                rand_adv_idx = randint(0, num_adv)
                #print(rand_adv_idx)
                random_token_idx = randint(0, top_k)
                batch_item[0, adv_idxs[rand_adv_idx]] = top_k_adv[rand_adv_idx, random_token_idx]
                sequence_batch.append(batch_item)
            
                outputs = model(batch_item,output_hidden_states=True)  # shape: [batch_size, seq_length, hidden_dim]

                # Step 4: Get the final hidden states (output from the last transformer layer)
                #print(outputs.hidden_states)
                last_hidden_state = outputs.hidden_states[-1]
                batch_loss.append(torch.unsqueeze(cosine_similarity_loss_avg(last_hidden_state[:,-1], query_hiddens),dim=0))

        sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index].unsqueeze(0)

    return adv_seq, min_loss




def gcg_step_adq(adv_doc_t,query_t, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = query_t.shape[0]
    query_t = query_t.reshape(1,-1)
    #print(num_adv)
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)[-1]
    #print(doc_embeddings.requires_grad)# = True

    query_embedding = word_embedding_layer(query_t)
    query_embedding.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    gradients = query_embedding.grad
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod[:,forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, top_k).indices)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_item = query_t.clone().detach()
            rand_adv_idx = randint(0, num_adv)
            #print(rand_adv_idx)
            random_token_idx = randint(0, top_k)
            batch_item[0, rand_adv_idx] = top_k_adv[rand_adv_idx, random_token_idx]
            sequence_batch.append(batch_item)
            batch_loss.append(torch.unsqueeze(0-cosine_similarity_loss_sin(doc_embeddings,word_embedding_layer(batch_item)),dim=0))

        sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index].unsqueeze(0)

    return adv_seq[0], min_loss
def gcg_step_adq2(adv_doc_t,query_ts, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    #num_adv = query_t.shape[0]
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data


    query_embeddings = [word_embedding_layer(query_t.reshape(1,-1)) for query_t in query_ts]
    for qe in query_embeddings:

        qe.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-Q_loss(query_embeddings)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    candidates = []


    for query_embedding in query_embeddings:
        gradients = query_embedding.grad
        
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
        #print(query_embedding.shape)
        #print(gradients[0].shape)
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        #print(dot_prod.shape)
        #print("stop")
        #nn = None
        #print(nn.shape)
        # Set dot product of forbidded tokens to -inf
        forbidden_token_ids = forbidden_tokens
        dot_prod[:,forbidden_token_ids] = float("-inf")

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)
        candidates.append(top_k_adv)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_items = []
            for j in range(len(query_ts)):

                batch_item = query_ts[j].clone().detach()
                rand_adv_idx = randint(0, query_ts[j].shape[0])
                #print(rand_adv_idx)
                random_token_idx = randint(0, top_k)
                batch_item[rand_adv_idx] = candidates[j][rand_adv_idx, random_token_idx]
                batch_items.append(batch_item)
            sequence_batch.append(batch_items)

            batch_emb = [word_embedding_layer(batch_item.reshape(1,-1)) for batch_item in batch_items]
            batch_loss.append(torch.unsqueeze(Q_loss(batch_emb),dim=0))

        #sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index]#.unsqueeze(0)

    return adv_seq, min_loss

def doc_q_loss(doc_embeddings,query_embeddings):
    loss = 0.0
    for query_embedding in query_embeddings:

        loss += cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss/= len(query_embeddings)
    return 0-loss

def doc_q_loss2(doc_embeddings,query_embeddings):
    loss = 0.0
    for query_embedding in query_embeddings:

        loss += 1/(1+1e-6-cosine_similarity_loss_sin(doc_embeddings, query_embedding) )
    loss/= len(query_embeddings)
    return 0-loss

def gcg_step_adq3(adv_doc_t,query_ts, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    #num_adv = query_t.shape[0]
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)[-1]
    #print(doc_embeddings.requires_grad)# = True
    
    query_embeddings = [word_embedding_layer(query_t.reshape(1,-1)) for query_t in query_ts]
    for qe in query_embeddings:

        qe.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-Q_loss(query_embeddings,lambda_center=0.0,lambda_diverse=0.0)
    #loss-=doc_q_loss(doc_embeddings,query_embeddings)
    #for query_embedding in query_embeddings:
 
    #    loss += cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    candidates = []


    for query_embedding in query_embeddings:
        gradients = query_embedding.grad
        
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
        #print(query_embedding.shape)
        #print(gradients[0].shape)
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        #print(dot_prod.shape)
        #print("stop")
        #nn = None
        #print(nn.shape)
        # Set dot product of forbidded tokens to -inf
        forbidden_token_ids = forbidden_tokens
        dot_prod[:,forbidden_token_ids] = float("-inf")

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)
        candidates.append(top_k_adv)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_items = []
            for j in range(len(query_ts)):

                batch_item = query_ts[j].clone().detach()
                rand_adv_idx = randint(0, query_ts[j].shape[0])
                #print(rand_adv_idx)
                random_token_idx = randint(0, top_k)
                batch_item[rand_adv_idx] = candidates[j][rand_adv_idx, random_token_idx]
                batch_items.append(batch_item)
            sequence_batch.append(batch_items)

            batch_emb = [word_embedding_layer(batch_item.reshape(1,-1)) for batch_item in batch_items]
            #batch_loss.append(torch.unsqueeze(Q_loss(batch_emb)+doc_q_loss(doc_embeddings,batch_emb),dim=0))
            batch_loss.append(torch.unsqueeze(Q_loss2(batch_emb),dim=0))

        #sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index]#.unsqueeze(0)

    return adv_seq, min_loss


def gcg_step_adq4(adv_doc_t,query_ts, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    #num_adv = query_t.shape[0]
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)[-1]
    #print(doc_embeddings.requires_grad)# = True
    
    query_embeddings = [word_embedding_layer(query_t.reshape(1,-1)) for query_t in query_ts]
    for qe in query_embeddings:

        qe.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-Q_loss(query_embeddings,lambda_center=0.0,lambda_diverse=0.0)
    loss-=doc_q_loss(doc_embeddings,query_embeddings)
    #for query_embedding in query_embeddings:
 
    #    loss += cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    candidates = []


    for query_embedding in query_embeddings:
        gradients = query_embedding.grad
        
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
        #print(query_embedding.shape)
        #print(gradients[0].shape)
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        #print(dot_prod.shape)
        #print("stop")
        #nn = None
        #print(nn.shape)
        # Set dot product of forbidded tokens to -inf
        forbidden_token_ids = forbidden_tokens
        dot_prod[:,forbidden_token_ids] = float("-inf")

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)
        candidates.append(top_k_adv)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_items = []
            for j in range(len(query_ts)):

                batch_item = query_ts[j].clone().detach()
                rand_adv_idx = randint(0, query_ts[j].shape[0])
                #print(rand_adv_idx)
                random_token_idx = randint(0, top_k)
                batch_item[rand_adv_idx] = candidates[j][rand_adv_idx, random_token_idx]
                batch_items.append(batch_item)
            sequence_batch.append(batch_items)

            batch_emb = [word_embedding_layer(batch_item.reshape(1,-1)) for batch_item in batch_items]
            batch_loss.append(torch.unsqueeze(Q_loss2(batch_emb)+doc_q_loss2(doc_embeddings,batch_emb),dim=0))
            #batch_loss.append(torch.unsqueeze(Q_loss(batch_emb),dim=0))

        #sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index]#.unsqueeze(0)

    return adv_seq, min_loss








def gcg_step_adq3_h(adv_doc_t,query_ts, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    #num_adv = query_t.shape[0]
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    #print(doc_embeddings.requires_grad)# = True
    
    #query_embeddings = [word_embedding_layer(query_t.reshape(1,-1)) for query_t in query_ts]
    
    query_embeddings = [word_embedding_layer(query_t) for query_t in query_ts]

    query_hiddens = []
    for query_emb in query_embeddings:
        outputs = model(inputs_embeds=query_emb.unsqueeze(0),output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        query_hiddens.append(last_hidden_state[:,-1])


    for qe in query_embeddings:

        qe.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-Q_loss(query_hiddens,lambda_center=0.0,lambda_diverse=1.0)
    #loss-=doc_q_loss(doc_embeddings,query_embeddings)
    #for query_embedding in query_embeddings:
 
    #    loss += cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    candidates = []


    for query_embedding in query_embeddings:
        gradients = query_embedding.grad
        
        #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
        #print(query_embedding.shape)
        #print(gradients[0].shape)
        dot_prod = torch.matmul(gradients, embedding_matrix.T)

        #print(dot_prod.shape)
        #print("stop")
        #nn = None
        #print(nn.shape)
        # Set dot product of forbidded tokens to -inf
        forbidden_token_ids = forbidden_tokens
        dot_prod[:,forbidden_token_ids] = float("-inf")

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)
        candidates.append(top_k_adv)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        with torch.no_grad():
            for _ in range(this_batch_size):
                batch_items = []
                for j in range(len(query_ts)):

                    batch_item = query_ts[j].clone().detach()
                    rand_adv_idx = randint(0, query_ts[j].shape[0])
                    #print(rand_adv_idx)
                    random_token_idx = randint(0, top_k)
                    batch_item[rand_adv_idx] = candidates[j][rand_adv_idx, random_token_idx]
                    batch_items.append(batch_item)
                sequence_batch.append(batch_items)

                batch_hiddens = []
                for batch_item in batch_items:
                    outputs = model(batch_item.unsqueeze(0),output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    batch_hiddens.append(last_hidden_state[:,-1])

            #batch_loss.append(torch.unsqueeze(Q_loss(batch_emb)+doc_q_loss(doc_embeddings,batch_emb),dim=0))
                batch_loss.append(torch.unsqueeze(Q_loss2(batch_hiddens,lambda_center=0.0,lambda_diverse=1.0),dim=0))

        #sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index]#.unsqueeze(0)

    return adv_seq, min_loss


def gcg_step_adq4_h(adv_doc_t,query_ts, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    #num_adv = query_t.shape[0]
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)
    #print(doc_embeddings.requires_grad)# = True
    doc_embeddings.retain_grad()


    outputs = model(inputs_embeds=doc_embeddings,output_hidden_states=True)  # shape: [batch_size, seq_length, hidden_dim]

    last_hidden_state = outputs.hidden_states[-1]
    doc_hidden = last_hidden_state[:,-1]
    
    #query_embeddings = [word_embedding_layer(query_t.reshape(1,-1)) for query_t in query_ts]
    query_embeddings = [word_embedding_layer(query_t) for query_t in query_ts]
    
    query_hiddens = []
    for query_emb in query_embeddings:
        outputs = model(inputs_embeds=query_emb.unsqueeze(0),output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        query_hiddens.append(last_hidden_state[:,-1])

    for qe in query_embeddings:

        qe.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-Q_loss(query_hiddens,lambda_center=0.0,lambda_diverse=0.0)
    loss-=doc_q_loss(doc_hidden,query_hiddens)
    #for query_embedding in query_embeddings:
 
    #    loss += cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    candidates = []


    for query_embedding in query_embeddings:
        gradients = query_embedding.grad
        
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
        #print(query_embedding.shape)
        #print(gradients[0].shape)
        dot_prod = torch.matmul(gradients, embedding_matrix.T)

        #print(dot_prod.shape)
        #print("stop")
        #nn = None
        #print(nn.shape)
        # Set dot product of forbidded tokens to -inf
        forbidden_token_ids = forbidden_tokens
        dot_prod[:,forbidden_token_ids] = float("-inf")

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)
        candidates.append(top_k_adv)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        with torch.no_grad():
            for _ in range(this_batch_size):
                batch_items = []
                for j in range(len(query_ts)):

                    batch_item = query_ts[j].clone().detach()
                    rand_adv_idx = randint(0, query_ts[j].shape[0])
                    #print(rand_adv_idx)
                    random_token_idx = randint(0, top_k)
                    batch_item[rand_adv_idx] = candidates[j][rand_adv_idx, random_token_idx]
                    batch_items.append(batch_item)
                sequence_batch.append(batch_items)

                batch_hiddens = []
                for batch_item in batch_items:
                    outputs = model(batch_item.unsqueeze(0),output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    batch_hiddens.append(last_hidden_state[:,-1])

                batch_loss.append(torch.unsqueeze(Q_loss2(batch_hiddens)+doc_q_loss2(doc_hidden,batch_hiddens),dim=0))
            #batch_loss.append(torch.unsqueeze(Q_loss(batch_emb),dim=0))

        #sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index]#.unsqueeze(0)

    return adv_seq, min_loss












def gcg_step_adq4_tk(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    #num_adv = query_t.shape[0]
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)[-1]
    #print(doc_embeddings.requires_grad)# = True
    
    query_embeddings = [word_embedding_layer(query_t.reshape(1,-1)) for query_t in query_ts]
    for qe in query_embeddings:

        qe.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-Q_loss(query_embeddings,lambda_center=1.0,lambda_diverse=1.0)
    loss-=doc_q_loss(doc_embeddings,query_embeddings)
    #for query_embedding in query_embeddings:
 
    #    loss += cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    candidates = []


    for query_embedding in query_embeddings:
        gradients = query_embedding.grad
        
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
        #print(query_embedding.shape)
        #print(gradients[0].shape)
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        #print(dot_prod.shape)
        #print("stop")
        #nn = None
        #print(nn.shape)
        # Set dot product of forbidded tokens to -inf
        forbidden_token_ids = forbidden_tokens
        dot_prod[:,forbidden_token_ids] = float("-inf")

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)
        candidates.append(top_k_adv)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_items = []
            for j in range(len(query_ts)):

                batch_item = query_ts[j].clone().detach()
                rand_adv_idx = randint(adv_idxs[j][0],adv_idxs[j][1])
                #print(rand_adv_idx)
                random_token_idx = randint(0, top_k)
                batch_item[rand_adv_idx] = candidates[j][rand_adv_idx, random_token_idx]
                batch_items.append(batch_item)
            sequence_batch.append(batch_items)

            batch_emb = [word_embedding_layer(batch_item.reshape(1,-1)) for batch_item in batch_items]
            batch_loss.append(torch.unsqueeze(Q_loss(batch_emb,lambda_center=1.0,lambda_diverse=1.0)+doc_q_loss(doc_embeddings,batch_emb),dim=0))
            #batch_loss.append(torch.unsqueeze(Q_loss(batch_emb),dim=0))

        #sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index]#.unsqueeze(0)

    return adv_seq, min_loss

def gcg_step_adq3_tk(adv_doc_t,query_ts,adv_idxs, model, forbidden_tokens, top_k, num_samples, batch_size):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    #num_adv = query_t.shape[0]
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)[-1]
    #print(doc_embeddings.requires_grad)# = True
    
    query_embeddings = [word_embedding_layer(query_t.reshape(1,-1)) for query_t in query_ts]
    for qe in query_embeddings:

        qe.retain_grad()
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-Q_loss(query_embeddings,lambda_center=1.0,lambda_diverse=1.0)
    #loss-=doc_q_loss(doc_embeddings,query_embeddings)
    #for query_embedding in query_embeddings:
 
    #    loss += cosine_similarity_loss_sin(doc_embeddings, query_embedding)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    candidates = []


    for query_embedding in query_embeddings:
        gradients = query_embedding.grad
        
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
        #print(query_embedding.shape)
        #print(gradients[0].shape)
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        #print(dot_prod.shape)
        #print("stop")
        #nn = None
        #print(nn.shape)
        # Set dot product of forbidded tokens to -inf
        forbidden_token_ids = forbidden_tokens
        dot_prod[:,forbidden_token_ids] = float("-inf")

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)
        candidates.append(top_k_adv)

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_items = []
            for j in range(len(query_ts)):

                batch_item = query_ts[j].clone().detach()
                rand_adv_idx = randint(adv_idxs[j][0],adv_idxs[j][1])
                #print(rand_adv_idx)
                random_token_idx = randint(0, top_k)
                batch_item[rand_adv_idx] = candidates[j][rand_adv_idx, random_token_idx]
                batch_items.append(batch_item)
            sequence_batch.append(batch_items)

            batch_emb = [word_embedding_layer(batch_item.reshape(1,-1)) for batch_item in batch_items]
            batch_loss.append(torch.unsqueeze(Q_loss(batch_emb,lambda_center=1.0,lambda_diverse=1.0),dim=0))
            #batch_loss.append(torch.unsqueeze(Q_loss(batch_emb),dim=0))

        #sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index]#.unsqueeze(0)

    return adv_seq, min_loss

def gcg_step_show(adv_doc_t,query_ts, adv_idxs, model, forbidden_tokens, top_k, num_samples, batch_size,tokenizer):
    """
    Implements one step of the Greedy Coordinate Gradient (GCG) adversarial attack algorithm for LLMs.
    Args:
        adv_doc: Sequence of tokens to be given as input to the LLM. Contains the document and concanated adversarial tokens
                        Shape: [1, #tokens].
        model: LLM model. Type: AutoModelForCausalLM.
        loss_function: Loss function to be used for the attack. This funtion should take the embeddings of the
                       input sequence and return the loss of the target sequence (a list of loss
                       values for a batch of embeddings). Format: loss_function(embeddings, model).
        forbidden_tokens: Tokens that are not allowed in the adversarial sequence, e.g., non-ascii tokens.
        top_k: Top k adversarial tokens to be considered for each adversarial token.
        num_samples: Number of adversarial sequences to be generated in each iteration.
        batch_size: Batch size for the attack.
        adv_idxs: List of indices of adversarial tokens in the prompt.
    Returns:
        input_sequence: Updated input sequence.
        min_loss: Minimum loss of the batch of adversarial sequences generated by the attack.
    """

    num_adv = len(adv_idxs)
    #print(num_adv)
    
    # Get word embedding layer and matrix
    word_embedding_layer = model.get_input_embeddings()
    embedding_matrix = word_embedding_layer.weight.data

    # Get word embeddings for input sequence
    doc_embeddings = word_embedding_layer(adv_doc_t)
    #print(doc_embeddings.requires_grad)# = True
    doc_embeddings.retain_grad()

    query_embeddings = [word_embedding_layer(query_t)[-1] for query_t in query_ts]
    #for qe in query_embeddings:
    #    qe.requires_grad = False

    # Get loss and gradients
    loss = 0-cosine_similarity_loss(doc_embeddings, query_embeddings)
    loss.backward()  # Minimize loss
    #print(doc_embeddings.requires_grad)# = True
    #print(doc_embeddings.grad)
    #print(query_embeddings[0].grad)
    gradients = doc_embeddings.grad
    #print(gradients.shape)
    # Dot product of gradients and embedding matrix
    #print(embedding_matrix.shape)
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

    # Set dot product of forbidded tokens to -inf
    forbidden_token_ids = forbidden_tokens
    dot_prod[:,forbidden_token_ids] = float("-inf")

    # Get top k adversarial tokens
    top_k_adv = (torch.topk(dot_prod, top_k).indices)[adv_idxs]

    adv_seq = None
    min_loss = float("inf")

    # Create samples of adversarial sequences in batches
    for i in range(ceil(num_samples / batch_size)):
        this_batch_size = min(batch_size, num_samples - i * batch_size)
        # Create a batch of input sequences by uniformly sampling from top k adversarial tokens
        sequence_batch = []
        batch_loss = []

        for _ in range(this_batch_size):
            batch_item = adv_doc_t.clone().detach()
            rand_adv_idx = randint(0, num_adv)
            #print(rand_adv_idx)
            random_token_idx = randint(0, top_k)
            batch_item[0, adv_idxs[rand_adv_idx]] = top_k_adv[rand_adv_idx, random_token_idx]
            sequence_batch.append(batch_item)
            batch_loss.append(torch.unsqueeze(cosine_similarity_loss(word_embedding_layer(batch_item), query_embeddings),dim=0))
            print(f"{tokenizer.batch_decode(batch_item, skip_special_tokens=True)[0]}   Loss:{batch_loss[-1]}")


        sequence_batch = torch.cat(sequence_batch, dim=0)
        batch_loss = torch.cat(batch_loss, dim=0)

        # Compute loss for the batch of sequences

        # Find the index with the minimum loss
        min_batch_loss, min_loss_index = torch.min(batch_loss, dim=0)
        min_batch_loss = min_batch_loss.item()

        # Update minimum loss and adversarial sequence
        if min_batch_loss < min_loss:
            min_loss = min_batch_loss
            adv_seq = sequence_batch[min_loss_index].unsqueeze(0)

    return adv_seq, min_loss



def get_nonascii_toks(tokenizer, device='cpu'):
    """
    Returns the non-ascii tokens in the tokenizer's vocabulary.
    Fucntion obtained from the llm-attacks repository developed as part of the paper
    'Universal and Transferable Adversarial Attacks on Aligned Language Models' by Zou et al.
    Code Reference: https://github.com/llm-attacks/llm-attacks/blob/0f505d82e25c15a83b6954db28191b69927a255d/llm_attacks/base/attack_manager.py#L61
    Args:
        tokenizer: Tokenizer.
    Returns:
        ascii_toks: Non-ascii tokens in the tokenizer's vocabulary.
    """

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)