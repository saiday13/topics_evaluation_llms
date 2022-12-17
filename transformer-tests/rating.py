import time
import numpy as np
from gpt3 import gpt3
from scipy.stats import spearmanr, pearsonr

class Rating():
    def __init__(self, topics, num_topics, path_save, file_path):
        super(Rating, self).__init__()

        ## define hyperparameters
        self.topics = topics
        self.num_topics = num_topics
        self.path_save = path_save
        self.file_path = file_path
        self.prompt_name = 'rating_p3_v1'


    def run_gpt3(self, prompts):
        # run GPT-3 for all topics
        with open(self.path_save + self.prompt_name + '_gpt3.txt', 'w') as f:
            for i in range(len(prompts)):
                prompt = prompts[i]
                response = str(gpt3(prompt)).replace('\n', '')
                f.write(response + '\n')
                print(prompt, " ", response)
                time.sleep(3)
            f.close()


    def create_prompt(self):
        '''
        p1: Rate how related the following terms are to each other as 'very related', 'somewhat related' or 'not related'
        p2: Rate how related the following terms are to each other in a range from 1 to 3
        '''

        prompts = []
        topics = [topic.split() for topic in self.topics]
        for i in range(len(self.topics)):
            list1 = "['file', 'window', 'problem', 'run', 'system', 'program', 'font', 'work', 'win', 'change']"
            list2 = "['chip', 'clipper', 'phone', 'key', 'encryption', 'government', 'system', 'write', 'nsa', 'communication']"
            list_ten_terms = str(topics[i][:10])

            # list1 = list1.replace("[", "").replace("]", "")
            # list2 = list2.replace("[", "").replace("]", "")
            # list_ten_terms = list_ten_terms.replace("[", "").replace("]", "")

            # list1 = list1.replace("'", "")
            # list2 = list2.replace("'", "")
            # list_ten_terms = list_ten_terms.replace("'", "")

            # list1 = 'file, window, problem, run, system, program, font, work, win, change'
            # list2 = 'chip, clipper, phone, key, encryption, government, system, write, nsa, communication'
            # list_ten_terms = list_ten_terms.replace("'", "").replace("[", "").replace("]", "")

            prompts.append("Rate how related the following terms are to each other " \
                           "as '3-very related', '2-somewhat related' or '1-not related': " + list1 + "\n" \
                           "Answer: 3\n\nRate how related the following terms are to each other " \
                           "as '3-very related', '2-somewhat related' or '1-not related': " + list2 + "\n"
                           "Answer: 2\n\nRate how related the following terms are to each other " \
                           "as '3-very related', '2-somewhat related' or '1-not related': " + list_ten_terms + "\nAnswer: ")



        with open(self.path_save + self.prompt_name + '.txt', 'w') as f:
            for i in range(self.num_topics):
                f.write(prompts[i] + '\n')
            f.close()

        print("Running GPT-3...")
        self.run_gpt3(prompts)


    def load_npmis(self):
        with open(self.file_path + 'npmis.txt') as file:
            npmis = file.readlines()
        npmis = [float(term.replace('\n', '')) for term in npmis]
        return npmis


    def load_responses(self):
        with open(self.path_save + self.prompt_name + '_gpt3.txt') as file:
            responses = file.readlines()
        numeric_response = [int(term.replace('\n', '').replace(' ', '')) for term in responses]

        return numeric_response


    def calculate_metrics(self):
        responses = self.load_responses()
        # calculate accuracy
        acc = np.mean(responses)
        variance = np.var(responses)
        spear_rho, spear_p = spearmanr(self.load_npmis(), responses)
        pear_rho, pear_p = pearsonr(self.load_npmis(), responses)
        correlations = {
            "spear_rho": spear_rho,
            "spear_p": spear_p,
            "pear_rho": pear_rho,
            "pear_p": pear_p
        }

        with open(self.path_save + self.prompt_name + '_counts.txt', 'w') as f:
            f.write("Counts: {}\nMean: {}\nVariance: {}\nCorrelations: {}".format(responses, acc, variance, correlations))
        print("\nMean: {}\nVariance: {}\nCorrelations: {}\n".format(acc, variance, correlations))


    def run_rating(self):
        print("Creation of prompts for GPT-3...")
        self.create_prompt()
        self.calculate_metrics()


