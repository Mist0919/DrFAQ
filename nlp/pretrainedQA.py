import torch
from transformers import BertTokenizer, BertForMaskedLM
from sentence_transformers import util
import pandas as pd

class PretrainedQA:
    
    def __init__(self, excel_file):
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained("model/vocab.txt",local_files_only=True)
        self.model = BertForMaskedLM.from_pretrained('model/pytorch_model.bin',config='model/config.json', local_files_only=True)
        self.df = pd.read_csv(excel_file)
        self.token_list = []
        c = 0
        for s in self.df['Question']:
            print(c)
            c += 1
            self.token_list.append(self.model(**self.tokenizer(s, return_tensors="pt")))
        #self.token_list = [self.model(**self.tokenizer(s, return_tensors="pt")) ]

    def ask(self, question, threshold=0.6):
        """Ask question to QA."""
        score, answer = self.query(question)
        print("NLP score:", score)
        print("Answer:", answer)

        if score > threshold:
            return answer
        else:
            return None

    def query(self, question):
        max_score = 0
        question_idx = 0
        token_text = self.model.encode(**self.tokenizer(question, return_tensors="pt"))
        temp = []
        for i, q in enumerate(self.token_list):
            s = self.compare(token_text, q)
            temp.append((q, i, s))
            if s > max_score:
                max_score = s
                question_idx = i
        return max_score, self.df['Answer'][question_idx]

    def get_ranks(self, question, threshold=0.8):
        """Returns FAQ answer if similarity score exceeds threshold"""
        max_score = 0
        token_text = self.model.encode(question)
        temp = []
        for i, q in enumerate(self.token_list):
            s = self.compare(token_text, q)
            temp.append((i, s))
            if s > threshold:
                temp.append((i, s))
            if s > max_score:
                max_score = s       
        rank = sorted(temp, key=lambda i: i[-1], reverse=True)
        # print(rank[:3], "FAQ score:", max_score)
        if max_score > threshold:
            return rank
        else:
            return None
    
    def compare(self, embedding1, embedding2):
        return util.pytorch_cos_sim(embedding1, embedding2).item()
        

if __name__ == "__main__":
    """Example"""
    qa = PretrainedQA("../chat/trainDataFinal.csv")
    print("waiting")
    #score, answer = qa.query("What is Covid?")
    #print("Answer:", answer)
    #print("Score:", score)
    print(qa.query("What is Covid?"))
    