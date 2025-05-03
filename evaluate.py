from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import torch.nn.functional as F

def evaluate_sst2(model, tokenizer=None, device='cuda' if torch.cuda.is_available() else 'cpu', num_samples=500):
    dataset = load_dataset("glue", "sst2", split="validation[:{}]".format(num_samples))
    correct = 0
    total = 0

    model.eval()
    for example in dataset:
        inputs = tokenizer(example['sentence'], return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            prediction = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            correct += (prediction.cpu().item() == example['label'])
            total += 1
    model.train()
    return correct / total if total > 0 else 0.0

def evaluate_squad(model, tokenizer=None, device='cuda' if torch.cuda.is_available() else 'cpu', num_samples=100):
    dataset = load_dataset("squad", split="validation[:{}]".format(num_samples))
    f1_total = 0.0
    em_total = 0.0
    total = 0

    model.eval()
    for example in dataset:
        inputs = tokenizer(
            example['question'], example['context'],
            return_tensors='pt', padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            start = torch.argmax(start_logits, dim=-1).cpu().item()
            end = torch.argmax(end_logits, dim=-1).cpu().item()
            predicted_answer = tokenizer.decode(inputs['input_ids'][0][start:end+1])

            ground_truth = example['answers']['text'][0]
            f1_total += compute_f1(predicted_answer, ground_truth)
            em_total += compute_em(predicted_answer, ground_truth)
            total += 1
    model.train()
    return f1_total / total if total > 0 else 0.0, em_total / total if total > 0 else 0.0

def compute_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_em(prediction, ground_truth):
    return float(prediction.strip().lower() == ground_truth.strip().lower())
