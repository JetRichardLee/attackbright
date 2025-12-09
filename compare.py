import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"

import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS,calculate_retrieval_metrics,attack_by_surrogate,attack_by_advl,attack_by_original,get_scores_sf_qwen_e5,attack_by_surrogate_emb,attack_by_original_emb
from retrievers import attack_by_adv_emb,attack_by_adv2_emb,attack_by_adiv_emb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
import torch

import matplotlib.pyplot as plt
def Cal_PM(sample_x,D):
    """
    
    Parameters
    ----------
    sample_x : N*D matrix
        sampled matrix of features.
    D : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    XXT = torch.matmul(sample_x.T,sample_x)/sample_x.shape[1]
    _ , U =torch.linalg.eig(XXT)
    return U[:,0:D].to(torch.float64)

def PCA_visual(tensor_list,label,title,time,num):
    #print(tensor_list)
    all_t = torch.stack(tensor_list).to(torch.float64)#.cpu()
    all_t = all_t.reshape(all_t.shape[0],-1)
    M = Cal_PM(all_t,2)
    all_t = torch.matmul(all_t, M).detach().numpy()
    
    plt.figure(figsize=(6,6))
    plt.scatter(all_t[:, 0], all_t[:, 1],c=label,cmap='tab10', s=20)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Visualization of embedding")
    plt.grid(True)
    plt.axis("equal")  # keep aspect ratio equal
    #plt.show()
    plt.savefig(f"fig\{title}_{time}_{num}.png", dpi=300, bbox_inches="tight")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25','cohere','e5','google','grit','inst-l','inst-xl',
                                 'openai','qwen','qwen2','sbert','sf','voyage','bge'])
                                 
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model}_long_{args.long_context}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    score_file_path = os.path.join(args.output_dir,f'score.json')

    if args.input_file is not None:
        with open(args.input_file) as f:
            examples = json.load(f)
    elif args.reasoning is not None:
        examples = load_dataset('xlangai/bright', f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
    else:
        examples = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]
        
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents',cache_dir=args.cache_dir)[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents',cache_dir=args.cache_dir)[args.task]
    doc_ids = []
    documents = []
    o_documents = []
    p_documents = []
    su_documents = []
    di_documents = []
    s_documents = []
    
    query_emb_sur = []
    sample_query_embs = []
    susample_query_embs = []
    disample_query_embs = []
    
    ori_doc_embs_sur = []
    per_doc_embs_sur = []
    sup_doc_embs_sur = []
    div_doc_embs_sur = []
    spe_doc_embs_sur = []


    mapping_doc = {}
    for dp in doc_pairs:
        #print(dp['id'])
        doc_ids.append(dp['id'])
        documents.append(dp['content'])
        mapping_doc[dp['id']]=len(doc_ids)-1

    
    surrogate_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-4B-Chat",
            torch_dtype="auto",
            device_map="auto"
        )
    surrogate_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")
    if True:#not os.path.isfile(score_file_path):
        with open(os.path.join(args.config_dir,args.model,f"{args.task}.json")) as f:
            config = json.load(f)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        queries = []
        query_ids = []
        excluded_ids = {}
        if args.long_context:
            key = 'gold_ids_long'
        else:
            key = 'gold_ids'
        #print(examples)
        #print("Begin to attack")
        for e in tqdm(examples):
            
            queries.append(e["query"])
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            #"""
            cnt =0 
            #print("Original:",e["query"])
            for doc_id in e[key]:
                index = mapping_doc[doc_id]
                o_documents.append(documents[index])
                #if cnt==2:
                    #with open('doc.txt', 'w') as file:
                        # Write a string to the file
                    #    file.write(documents[index])
                p_document,_,qs_embs,pd_emb_s = attack_by_adv_emb(documents[index],surrogate_model,surrogate_tokenizer,num_q=20)
                p_documents.append(p_document)

                su_document,_,suqs_embs,sud_emb_s = attack_by_surrogate_emb(documents[index],surrogate_model,surrogate_tokenizer,num_q=20)
                su_documents.append(su_document)
                
                di_document,_,diqs_embs,did_emb_s =attack_by_adiv_emb(documents[index],surrogate_model,surrogate_tokenizer,num_q=20)
                di_documents.append(di_document)

                s_document,_,q_emb_s,od_emb_s,sd_emb_s = attack_by_original_emb(documents[index],e["query"],surrogate_model,surrogate_tokenizer,num_q=1)
                s_documents.append(s_document)
                #documents[index],_ = attack_by_original(documents[index],e["query"],surrogate_model,surrogate_tokenizer,num_q=1)
                #documents[index],_ = attack_by_advl(documents[index],surrogate_model,surrogate_tokenizer,num_q=30)
                
                query_emb_sur.append(q_emb_s)
                sample_query_embs.append(qs_embs)
                susample_query_embs.append(suqs_embs)
                disample_query_embs.append(diqs_embs)

                ori_doc_embs_sur.append(od_emb_s)
                per_doc_embs_sur.append(pd_emb_s)
                sup_doc_embs_sur.append(sud_emb_s)
                div_doc_embs_sur.append(did_emb_s)
                spe_doc_embs_sur.append(sd_emb_s)

                cnt+=1
                #print(documents[index])
                #break
                if cnt>=3:
                    break
                    
            #break
            #"""
            #print("//////////////////////////")
            #print("id ",len(queries),":")
            #print(e["query"])
            #print(e[key])
            
            #print(mapping_doc[e[key][0]])
            #print(documents[mapping_doc[e[key][0]]])
            #if cnt%10==0:
            #    print(cnt)
            #    break
            assert len(overlap)==0
        assert len(queries)==len(query_ids), f"{len(queries)}, {len(query_ids)}"
        #print(None.shape)
        #print("attack_ended")
        if not os.path.isdir(os.path.join(args.cache_dir, 'doc_ids')):
            os.makedirs(os.path.join(args.cache_dir, 'doc_ids'))
        if os.path.isfile(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")):
            with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")) as f:
                cached_doc_ids = json.load(f)
            for id1,id2 in zip(cached_doc_ids,doc_ids):
                assert id1==id2
        else:
            with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json"),'w') as f:
                json.dump(doc_ids,f,indent=2)
        assert len(doc_ids)==len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")


        #documents = documents[:5]
        #doc_ids = doc_ids[:5]

        if args.debug:
            documents = documents[:30]
            doc_paths = doc_ids[:30]
        kwargs = {}
        if args.query_max_length>0:
            kwargs = {'query_max_length': args.query_max_length}
        if args.doc_max_length>0:
            kwargs.update({'doc_max_length': args.doc_max_length})
        if args.encode_batch_size>0:
            kwargs.update({'batch_size': args.encode_batch_size})
        if args.key is not None:
            kwargs.update({'key': args.key})
        if args.ignore_cache:
            kwargs.update({'ignore_cache': args.ignore_cache})

        """
        Here to perform document perturbation
        implement a function that 
        docs = perform_attack(doc, gids)

        doc-> pdoc
        """
        q_embs_tar,od_embs_tar,o_scores = get_scores_sf_qwen_e5(
            queries=queries, query_ids=query_ids, documents=o_documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )
        _,pd_embs_tar,p_scores = get_scores_sf_qwen_e5(
            queries=queries, query_ids=query_ids, documents=p_documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )
        _,sud_embs_tar,su_scores = get_scores_sf_qwen_e5(
            queries=queries, query_ids=query_ids, documents=su_documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )
        _,did_embs_tar,di_scores = get_scores_sf_qwen_e5(
            queries=queries, query_ids=query_ids, documents=di_documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )
        _,sd_embs_tar,s_scores = get_scores_sf_qwen_e5(
            queries=queries, query_ids=query_ids, documents=s_documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )
        cntq =0 
        cntd =0 
        num1 = 0
        num2 = 0
        num=0
        for e in tqdm(examples):
            #print("Original:",e["query"])
            cnt = 0
            for doc_id in e[key]:
                #index = mapping_doc[doc_id]
                if o_scores[cntq][cntd]>p_scores[cntq][cntd] and p_scores[cntq][cntd]<s_scores[cntq][cntd] and p_scores[cntq][cntd]<su_scores[cntq][cntd] and p_scores[cntq][cntd]<di_scores[cntq][cntd]:
                    
                    num+=1
                    #num2+=1
                    torch.save([query_emb_sur[cntd],ori_doc_embs_sur[cntd],spe_doc_embs_sur[cntd]],f"fig/specif_before_{num}.pt")
                    torch.save([q_embs_tar[cntq],od_embs_tar[cntd],sd_embs_tar[cntd]],f"fig/specif_after_{num}.pt")

                    torch.save([sample_query_embs[cntd],query_emb_sur[cntd],ori_doc_embs_sur[cntd],per_doc_embs_sur[cntd]],f"fig/sample_before_{num}.pt")

                    torch.save([q_embs_tar[cntq],od_embs_tar[cntd],pd_embs_tar[cntd]],f"fig/sample_after_{num}.pt")
                    
                    torch.save([susample_query_embs[cntd],query_emb_sur[cntd],ori_doc_embs_sur[cntd],sup_doc_embs_sur[cntd]],f"fig/susample_before_{num}.pt")

                    torch.save([q_embs_tar[cntq],od_embs_tar[cntd],sud_embs_tar[cntd]],f"fig/susample_after_{num}.pt")


                    torch.save([disample_query_embs[cntd],query_emb_sur[cntd],ori_doc_embs_sur[cntd],div_doc_embs_sur[cntd]],f"fig/divsample_before_{num}.pt")
                    #torch.save([susample_query_embs[cntd],query_emb_sur[cntd],ori_doc_embs_sur[cntd],sup_doc_embs_sur[cntd]],f"fig/susample_before_{num}.pt")

                    torch.save([q_embs_tar[cntq],od_embs_tar[cntd],did_embs_tar[cntd]],f"fig/divsample_after_{num}.pt")
                cnt+=1
                cntd+=1
                if cnt>=3:
                    break

                if num>50:
                    break
                
            if num>50:
                break
            cntq+=1
        #print(num1)
        #print(num2)
        #print(scores)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path,'exists')


