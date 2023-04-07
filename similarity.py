# similarity.py

import re
import torch
import heapq
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering


def filter_paragraphs(paragraphs, min_length=5):
    # There must be at least 5 words in the paragraph ( eventually 30, but we'll fix that later )
    return [(i, p) for i, p in enumerate(paragraphs) if len(p.split()) >= min_length]


def calculate_similarity(paragraphs, question):
    print("[!] WARNING MAY RUN OUT OF RAM")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    scores = []
    for p in paragraphs:
        if p is None:
            return None
        #print(f"{paragraphs}, {question}")
        inputs = tokenizer(text=question, text_pair=p, return_tensors='pt', truncation=True, padding=True)
        #print(f"Inputs: {inputs}")
        output = model(**inputs)
        #print(f"Output: {output}")
        score = torch.max(output.start_logits).item() + torch.max(output.end_logits).item()
        scores.append(score)

    return scores


def find_top_k_similar(paragraphs, question, k=None, word_limit=2250):
    if paragraphs is None or len(paragraphs) == 0 or question is None or len(question) == 0:
        return []
    
    filtered_indices = [i for i, p in filter_paragraphs(paragraphs)]
    filtered_paragraphs = [paragraphs[i] for i in filtered_indices]

    print(f"Filtered Paragraphs: {filtered_paragraphs}")
    similarities = calculate_similarity(filtered_paragraphs, question)
    #print(f"Similarities: {similarities}")

    if k is not None:
        top_k_indices = heapq.nlargest(k, range(len(similarities)), key=lambda i: similarities[i])
    else:
        top_k_indices = heapq.nlargest(len(similarities), range(len(similarities)), key=lambda i: similarities[i])

        total_words = 0
        filtered_indices_list = []
        for idx in top_k_indices:
            original_idx = filtered_indices[idx]
            paragraph = paragraphs[original_idx]
            words_in_paragraph = len(paragraph.split())
            if total_words + words_in_paragraph <= word_limit:
                total_words += words_in_paragraph
                filtered_indices_list.append(original_idx)
            else:
                break

        top_k_indices = filtered_indices_list

    top_k_paragraphs = [paragraphs[i] for i in top_k_indices]
    return top_k_paragraphs
