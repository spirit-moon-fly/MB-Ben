import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

accelerator = Accelerator()
device = accelerator.device
logger.info(f"Using device: {device}")

#model name
MODEL_PATH = "Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

logger.info("Loading model with Accelerate...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16, 
    device_map="auto",
    offload_folder="offload_dir",  
    offload_state_dict=True
)
model.eval()

LABELS = ["entailment", "neutral", "contradiction"]

#few-shot-example1
shots = ''' 
Examine the pair of sentence and determine if they exhibit entailment, neutral or contradicition. Answer with either 'entailment', 'neutral', or 'contradiction': 
Premise: uh i don't know i i have mixed emotions about him uh sometimes i like him but at the same times i love to see somebody beat him.
Hypothesis: I like him for the most part, but would still enjoy seeing someone beat him.
Answer: 'entailment'

Premise: The new rights are nice enough. 
Hypothesis: Everyone really likes the newest benefits.
Answer: 'neutral'

Premise: This site includes a list of all award winners and a searchable database of Government Executive articles.
Hypothesis: The Government Executive articles housed on the website are not able to be searched.
Answer: 'contradiction'

'''

#few-shot-example2
# ''' 
# Examine the pair of sentence and determine if they exhibit entailment, neutral or contradicition. Answer with either 'entailment', 'neutral', or 'contradiction': 
# Premise: I am asserting my membership in the club of Old Geezers.
# Hypothesis: I am proclaiming that I am now a member of the club of Old Geezers.
# Answer: 'entailment'

# Premise: Jon's feeling of age and weariness must have shown.
# Hypothesis: Jon had traveled longer than his body could handle.
# Answer: 'neutral'

# Premise: oh really it wouldn't matter if we plant them when it was starting to get warmer.
# Hypothesis: It is better to plant when it is colder.
# Answer: 'contradiction'

# '''

#few-shot-example3
# shots = ''' 
# Examine the pair of sentence and determine if they exhibit entailment, neutral or contradicition. Answer with either 'entailment', 'neutral', or 'contradiction': 
# Premise: New York Times columnist Bob Herbert asserts that managed care has bought Republican votes and that patients will die as a result.
# Hypothesis: Managed care bought Republican votes and patients will end up dead because of this.
# Answer: 'entailment'

# Premise: Dirt mounds surrounded the pit so that the spectators stood five or six people deep around the edge of the pit.
# Hypothesis: The hole is seven feet deep.
# Answer: 'neutral'

# Premise: There are many homes built into the hillsides; some have been converted into art galleries and shops selling collectibles.
# Hypothesis: All of the homes in the hillside have been converted into art galleries and shops selling collectibles.
# Answer: 'contradiction'

# '''

def get_target_token_ids(tokenizer, labels):
    prefix = "Answer: '"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    
    tok_ids = {}
    logger.info("--- Re-calculating Target Token IDs based on new prefix ---")
    for l in labels:
        full_text = prefix + l
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        
        if len(full_ids) > len(prefix_ids):
            target_id = full_ids[len(prefix_ids)]
            tok_ids[l] = target_id
            logger.info(f"Label '{l}' under prefix '{prefix}' maps to Token ID: {target_id} (Decoded: '{tokenizer.decode(target_id)}')")
        else:
            raise ValueError(f"Tokenization failed for label: {l}")
    logger.info("---------------------------------------------------------")
    return tok_ids

TARGET_TOK_IDS = get_target_token_ids(tokenizer, LABELS)

def infer_probs(premise, hypothesis):
    prompt = shots + f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer: '"
    
    logger.info("\n" + "="*50 + " FULL PROMPT START " + "="*50)
    logger.info(prompt)
    logger.info("="*51 + " FULL PROMPT END " + "="*51 + "\n")

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inp)
        logits = outputs.logits[:, -1, :]
        print(f"Logits: {logits}")
        print(f"Predicted token id: {logits.argmax(dim=-1)}")
        print(f"Decoded response: {tokenizer.decode(logits.argmax(dim=-1)[0])}")
    logit_values = {l: logits[0, TARGET_TOK_IDS[l]].item() for l in LABELS}
    vec = torch.tensor(list(logit_values.values()), device=device)
    probs = torch.softmax(vec, dim=-1).cpu().numpy().tolist()


    top_k = 5
    top_k_logits, top_k_indices = torch.topk(logits[0], top_k)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    
    logger.info(">>> Diagnostic Prediction Output <<<")
    for i in range(top_k):
        tok_id = top_k_indices[i].item()
        tok_str = tokenizer.decode(tok_id)
        tok_prob = top_k_probs[i].item()
        tok_logit = top_k_logits[i].item()
        logger.info(f"  Rank {i+1} | Token ID: {tok_id:<6} | Str: {repr(tok_str):<10} | Logit: {tok_logit:.4f} | Local Prob: {tok_prob:.4f}")
    
    logger.info("Target Label Extraction Result:")
    for l in LABELS:
        logger.info(f"  {l:<15} -> Token ID: {TARGET_TOK_IDS[l]:<6} | Logit: {logit_values[l]:.4f} | Softmax Prob: {probs[LABELS.index(l)]:.4f}")

    return [float(x) for x in probs]

def add_orig_probs(data_file, out_file="augmented_dataset.jsonl"):
    with open(data_file) as f:
        data = [json.loads(l) for l in f]

    for i, e in enumerate(data):
        if "idx" not in e:
            e["idx"] = i + 1

    augmented_data = []

    for idx, e in enumerate(data):
        p_orig = infer_probs(e["premise"], e["hypothesis"])
        logger.info(f"sample{e['idx']:03d} orig_p={p_orig}")

        e["orig_probs"] = p_orig
        augmented_data.append(e)

    Path(Path(out_file).parent).mkdir(exist_ok=True, parents=True)

    with open(out_file, "w") as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved augmented dataset with orig_probs to {out_file}")

if __name__ == "__main__":
    add_orig_probs("../data/mnli.jsonl", out_file="mnli-with-orig_probs.jsonl")
    add_orig_probs("../data/hans.jsonl", out_file="hans-with-orig_probs.jsonl")
    add_orig_probs("../data/5bias.jsonl", out_file="5bias-with-orig_probs.jsonl")
    add_orig_probs("../data/3bias.jsonl", out_file="3bias-with-orig_probs.jsonl")
