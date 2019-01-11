import json

SQUAD_TRAIN, SQUAD_TEST = 'data/train-v1.1.json', 'data/dev-v1.1.json'

class DataLoader:
    def __init__(self, inputs):
        self.data = None
        self.title =None
        self.context = None
        self.question = None
        self.answer_text= None
        self.answer_start = None
        self.answer_end = None
        self.load_data(inputs)
        
    def article_to_cqa(self, data):
        '''
        structure:
        article > paragraphs > sub paragarph -- context
                                             -- q_a -- question
                                                    -- answers -- text
                                                               -- answer_start
        article = a pair of 'title', 'paragraphs'
        paragraphs = set of paragraphs
        sub_paragraph = a paragraph, that includes 'qas', 'context' as a key of dictionary
        sub_paragraph['qas'] = lots of question-answer pairs in a sub_paragraph
        '''
        title =[]
        context = []
        question = []
        answer_text = []
        answer_start = []
        answer_end =[]
        
        for i, article in enumerate(data):
            paragraphs = article['paragraphs'] 
            for sub_paragraph in paragraphs:
                for q_a in sub_paragraph['qas']:
                    title.append(article['title'])
                    context.append(sub_paragraph['context'])
                    question.append(q_a['question'])
                    answer_text.append(q_a['answers'][0]['text'])
                    answer_start.append(q_a['answers'][0]['answer_start'])
                    answer_end.append(q_a['answers'][0]['answer_start'] + len(q_a['answers'][0]['text']) - 1)
        
        return title, context, question, answer_text, answer_start, answer_end
        
    def load_data(self, inputs):
        with open(inputs) as f:
            data = json.load(f)
            self.data = data['data']
            
        self.title, self.context, self.question, self.answer_text, self.answer_start, self.answer_end = self.article_to_cqa(self.data)
    
if __name__ == "__main__":
    train_loader = DataLoader(SQUAD_TRAIN)
    test_loader = DataLoader(SQUAD_TEST)
