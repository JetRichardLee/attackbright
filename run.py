import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
import torch
import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS,calculate_retrieval_metrics,attack_by_surrogate,attack_by_advl,attack_by_original,attack_by_advl_tokens,attack_by_surrogate_div,attack_by_advl_ft
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        #cnt =0 
        for e in tqdm(examples):
            queries.append(e["query"])
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            #"""
            cnt=0
            #print("Original:",e["query"])
            for doc_id in e[key]:
                index = mapping_doc[doc_id]
                #if cnt==2:
                    #with open('doc.txt', 'w') as file:
                        # Write a string to the file
                    #    file.write(documents[index])
                #documents[index],_ = attack_by_surrogate(documents[index],surrogate_model,surrogate_tokenizer,num_q=20)
                #documents[index],_ = attack_by_original(documents[index],e["query"],surrogate_model,surrogate_tokenizer,num_q=1)
                #documents[index],_ = attack_by_advl(documents[index],surrogate_model,surrogate_tokenizer,num_q=10,q_rate=0.3)
                documents[index],_ = attack_by_advl_ft(documents[index],surrogate_model,surrogate_tokenizer,num_q=10,N_FT=30)
                #documents[index],_ = attack_by_surrogate_div(documents[index],surrogate_model,surrogate_tokenizer,num_q=10)
                #documents[index],_ = attack_by_advl_tokens(documents[index],surrogate_model,surrogate_tokenizer,num_q=20,q_rate=0.3)
                cnt+=1
                #print(documents[index])
                #break
                torch.cuda.empty_cache()
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
        scores = RETRIEVAL_FUNCS[args.model](
            queries=queries, query_ids=query_ids, documents=documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )

        #print(scores)
        with open(score_file_path,'w') as f:
            json.dump(scores,f,indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path,'exists')
    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in scores[e['id']]
            assert not did in ground_truth[e['id']]

    print(args.output_dir)
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
