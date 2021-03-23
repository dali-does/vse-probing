#from transformers import BertTokenizerFast ,BertForMaskedLM
import torch
import math
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

# Load pre-trained model (weights)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    predictions=model(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data
    return math.exp(loss)

class BertScorer():
    def __init__(self, weight_name='bert-base-uncased'):
        self.tokenizer = BertTokenizerFast.from_pretrained(weight_name,
                                                           do_lower_case=True)
        self.model = BertForMaskedLM.from_pretrained(weight_name)
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.device = self.get_device()
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda') 
        return torch.device('cpu')
        
    def get_score(self,sentence):
        with torch.no_grad():
            tensor_input = torch.tensor([self.tokenizer.encode(sentence.lower())])
            tensor_input = tensor_input.to(self.device)
            predictions=self.model(tensor_input)
        loss = self.loss_fct(predictions[0].squeeze(),tensor_input.squeeze())
        return torch.exp(loss).data

    def get_scores(self,sentences):
        with torch.no_grad():
            sentences = [sentence.lower() for sentence in sentences]
            tokenized_sentences = self.tokenizer.batch_encode_plus(sentences,pad_to_max_length=True)

            tensor_sentences = torch.tensor(tokenized_sentences['input_ids'])
            tensor_sentences = tensor_sentences.to(self.device)
            predictions=self.model(tensor_sentences)

            losses = torch.empty(len(sentences))
            for i in range(len(sentences)):
                losses[i] = self.loss_fct(predictions[0][i],tensor_sentences[i])
            del tensor_sentences
            del tokenized_sentences
            del predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        #print(torch.cuda.memory_stats())
        return torch.exp(losses).data

