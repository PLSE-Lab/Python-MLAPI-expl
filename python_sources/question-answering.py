# %% [code]
import pandas as pd
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, BasicTokenizer
from transformers.data.metrics.squad_metrics import _get_best_indexes


class QuestionAnswering(object):
    def __init__(self, pretrained_model):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.model = BertForQuestionAnswering.from_pretrained(pretrained_model)

    def compute_predictions(self, start_logits, end_logits, input_ids, context, nbest, max_answer_length):
        """
        Given the start and end logits for the model return the answers
        """
        start_indexes = _get_best_indexes(start_logits, nbest)
        end_indexes = _get_best_indexes(end_logits, nbest)
        selected_answers = []
        #Get position of [SEP]
        sep_index = input_ids.index(103)
        answers = []
        #Add null prediction
        answer = {'text': 'CANNOTANSWER',
                'score': start_logits[0] + end_logits[0],
                 'start_index': -1,
                 'end_index': -1}
        answers.append(answer)
        for start_index in start_indexes:
            for end_index in end_indexes:
                #Avoid invalid predictions
                answer_len = end_index - start_index + 1
                if end_index < start_index:
                    continue
                if max_answer_length < answer_len:
                    continue
                #If answer includes question
                if start_index < sep_index:
                    continue
                text = self.tokenizer.decode(input_ids[start_index:start_index + answer_len],
                                             clean_up_tokenization_spaces=False)
                if '[SEP]' in text or'[CLS]' in text or '##' in text or '[PAD]' in text:
                    continue
                #Avoid selecting same answer twice in different subspans
                if text in selected_answers:
                    continue
                selected_answers.append(text)
                #If there is any problem when looking for the answer in the text
                try:
                    original_start_index = context.index(text)
                except:
                    continue
                original_end_index = original_start_index + len(text)
                answer = {'text': text.capitalize(),
                         'score': start_logits[start_index] + end_logits[end_index],
                         'start_index': original_start_index,
                         'end_index': original_end_index}
                answers.append(answer)
        return answers

    def run_qa(self, question, context, nbest, max_answer_len):
        """
        Given a question and a context retrieve the nbest answers
        """
        # Simple sliding window approach for max context cases
        tokenizer_dict = self.tokenizer.encode_plus(text=question, text_pair=context, max_length=384, stride=120,
                                                    return_overflowing_tokens=True, truncation_strategy='only_second')
        input_ids = [tokenizer_dict['input_ids']]
        input_type_ids = [tokenizer_dict['token_type_ids']]

        while 'overflowing_tokens' in tokenizer_dict.keys():
            tokenizer_dict = self.tokenizer.encode_plus(text=self.tokenizer.encode(question, add_special_tokens=False),
                                                        text_pair=tokenizer_dict['overflowing_tokens'],
                                                        max_length=384, stride=120, return_overflowing_tokens=True,
                                                        truncation_strategy='only_second',
                                                        is_pretokenized=True, pad_to_max_length=True)
            input_ids.append(tokenizer_dict['input_ids'])
            input_type_ids.append(tokenizer_dict['token_type_ids'])

        outputs = self.model(torch.tensor(input_ids), token_type_ids=torch.tensor(input_type_ids))
        answers = []
        for i in range(len(input_ids)):
            start_logits, end_logits = [output[i].detach().cpu().tolist() for output in outputs]
            answers += self.compute_predictions(start_logits, end_logits, input_ids[i], context.lower(), nbest,
                                                max_answer_len)

        answers.sort(key=lambda x: x['score'], reverse=True)
        return answers[0:nbest]

    def extract_answers(self, qstring, df_results, nbest, max_answer_len):
        """
        Given a question and the results from the ir experiments add the nbest answers for the questions.
        Each answer will contain the text and the score of it.
        """
        answers = []
        for i, context in enumerate(df_results['text']):
            context = ' '.join(self.basic_tokenizer.tokenize(context))
            answers.append(self.run_qa(qstring, context, nbest, max_answer_len))
            df_results['text'][i] = context
        df_results['qa_answers'] = answers
        return df_results


if __name__ == '__main__':
    pretrained_model = '/kaggle/input/scibertqasquad/checkpoint-31500'
   
    ids = ['77yma44s##sect10']
    texts = ["Seasonal influenza Seasonal influenza A and B illness in humans ranges \
    from subclinical or mild upper respiratory tract symptoms to more severe illness, \
    including laryngotracheitis and pneumonia, or less commonly, death from respiratory system failure.\
    The most common presenting symptoms are cough, high temperature, joint pain and general malaise. \
    The rapid onset and short incubation period are characteristic, though incubation can last up to 4 days. \
    Individuals at greatest risk of complications are those with pre-existing cardiac and respiratory disease, \
    the elderly, and those with impaired immunity. The severity of illness reflects pre-existing host immunity \
    and the prevailing virus strain."]
    
    df_ir_results = pd.DataFrame({'id': ids,
                                  'text': texts})

    qa_model = QuestionAnswering(pretrained_model)

    qstring = "Range of incubation periods for the disease in humans"
    top_answers = 1
    max_answer_length = 30
    answers = qa_model.extract_answers(qstring, df_ir_results, top_answers, max_answer_length)
    print(answers)