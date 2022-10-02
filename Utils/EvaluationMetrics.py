from nltk.translate.bleu_score import sentence_bleu


def calculate_bleu_score(reference=None, prediction=None, ngrams=None):
    reference_1 = reference[0].split()
    reference_2 = reference[1].split()
    reference_3 = reference[2].split()
    reference_4 = reference[3].split()
    reference_5 = reference[4].split()
    reference = [reference_1, reference_2, reference_3, reference_4, reference_5]
    if ngrams == 1:
        return sentence_bleu(reference, prediction, weights=(1, 0, 0, 0))
    return None
