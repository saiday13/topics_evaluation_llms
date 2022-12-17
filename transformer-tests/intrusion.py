import os
import time
import random
import numpy as np
from gpt3 import gpt3
from scipy.stats import spearmanr, pearsonr

class Intrusion():
    def __init__(self, topics, num_topics, path_save, file_path):
        super(Intrusion, self).__init__()

        ## define hyperparameters
        self.topics = topics
        self.num_topics = num_topics
        self.path_save = path_save
        self.file_path = file_path
        self.prompt_name = 'intr_p2_v6'


    def create_intruder(self, num_terms=5, sample_top_topic_terms=False):
        """
        topics_list: format is a list of dicts [
            {"topic_id": 1, "terms": ["a","b","c"]},
            {"topic_id": 2, "terms": ["a","b","c"]}
        ]

        n: The number of topics to sample
        """
        # sample_top_terms = 10
        # topic_intruder = []
        # # Generate n random ints for the selection of topics we'll conduct intrusion on
        # topic_idxs = random.sample(range(len(self.topics)), self.num_topics)
        # selected_intruders = set()
        #
        # for topic_idx in topic_idxs:
        #     # select another topic from which to grab a term, exclude the current topic
        #     random_topic_idx = random.choice([idx for idx in range(0, len(self.topics))
        #                                       if (idx != topic_idx and idx not in selected_intruders)])
        #     print(random_topic_idx)
        #     selected_intruders.add(random_topic_idx)
        #
        #     # assert that the new word is not in the top 5 words of the original topic
        #     correct_words = [word for word in self.topics[topic_idx]["terms"][:num_terms]]
        #     # collect top 10 words of the current topic
        #     top_topic_words = [word for word in self.topics[topic_idx]["terms"][:sample_top_terms]]
        #     print(top_topic_words)
        #     # collect top words of the 'intruder' topics that do NOT overlap with any of the top 10 words of the other topic
        #     if sample_top_terms:
        #         top_random_words = random.sample(
        #             [word for word in self.topics[random_topic_idx]["terms"][:sample_top_terms]
        #              if word not in top_topic_words], num_terms)
        #     else:
        #         top_random_words = [word for word in self.topics[random_topic_idx]["terms"][:num_terms]
        #                             if word not in top_topic_words]
        #
        #     print(top_random_words)
        #     # select the intruder word
        #     selected_intruder = random.choice(top_random_words)
        #
        #     topic_intruder.append(
        #         {
        #             "topic_id": self.topics[topic_idx]["topic_id"],
        #             "intruder_id": self.topics[random_topic_idx]["topic_id"],
        #             "intruder_term": selected_intruder,
        #             "topic_terms": correct_words
        #         }
        #     )
        # print('Done!')
        # sorted_by_topic = sorted(topic_intruder, key=lambda d: d['topic_id'])
        #
        # # Write the topic intruders to a file
        # if not os.path.isdir(self.path_save):
        #     os.system('mkdir -p ' + self.path_save)
        #
        # with open(self.path_save + 'intruders.txt', 'w') as f:
        #     for i in range(len(sorted_by_topic)):
        #         f.write(str(sorted_by_topic[i]) + '\n')

        intruder_list = []
        selected_intruders = set()
        for topic_idx in range(self.num_topics):
            # select another topic from which to grab a term, exclude the current topic
            random_topic_idx = random.choice(
                [idx for idx in range(0, self.num_topics) if (idx != topic_idx and idx not in selected_intruders)])
            selected_intruders.add(random_topic_idx)
            # take the top 5 words of the current topic and ONE of the top 5 terms from the top of the other topic
            # assert that the new word is not in the top 50 words of the original topic
            correct_words = [word for word in self.topics[topic_idx]["terms"][:num_terms]]

            # This collects the top 50 words of the current topic
            top_topic_words = [word for word in self.topics[topic_idx]["terms"][:50]]

            # This collects the top words of the 'intruder' topics that do NOT overlap with any of the top
            # 10 words of the other topic
            if sample_top_topic_terms:
                top_random_words = random.sample([word for word in self.topics[random_topic_idx]["terms"][:10] \
                                                  if word not in top_topic_words], num_terms)
            else:
                top_random_words = [word for word in self.topics[random_topic_idx]["terms"][:4] \
                                    if word not in top_topic_words]

            # EDGE-CASE - The top 50 words of the selected topic may overlap heavily with the
            # 'intruder' topics's top words. In this case, narrow down the set of excluded terms
            # for the current topic to just the top 10. If that doesn't work, then..... skip??
            if not top_random_words:
                top_topic_words = [word for word in self.topics[topic_idx]["terms"][:10]]
                top_random_words = [word for word in self.topics[random_topic_idx]["terms"][:5] \
                                    if word not in top_topic_words]

                if not top_random_words:
                    print(f"Skipping word intrusion for topic {topic_idx} with intruder {random_topic_idx}")
                    continue
            # select the intruder word
            selected_intruder = random.choice(top_random_words)

            # The last word in each list is the 'intruder', this should be randomized before showing
            # [topics_list[topic_idx]["topic_id"]] + correct_words + [selected_intruder]
            intruder_list.append(
                {
                    "topic_id": self.topics[topic_idx]["topic_id"],
                    "intruder_id": self.topics[random_topic_idx]["topic_id"],
                    "intruder_term": selected_intruder,
                    "topic_terms": correct_words
                }
            )
        # Write the topic intruders to a file
        if not os.path.isdir(self.path_save):
            os.system('mkdir -p ' + self.path_save)

        with open(self.path_save + 'intruders.txt', 'w') as f:
            for i in range(len(intruder_list)):
                f.write(str(intruder_list[i]) + '\n')

        return intruder_list


    def create_prompt(self, topic_intruder):
        '''  p1: Show the least related term
             p2: Select which term is the least related to all other terms
             p3: What is the intruder term in the following terms?
             p4: Which word does not belong?
             p5: Which one of the following words does not belong?
             p6: Find the intruder term
        '''

        list_of_terms = []
        topic_intruder_prompt = []
        for i in range(len(topic_intruder)):
            list_of_terms.append(topic_intruder[i]['topic_terms'] + topic_intruder[i]['intruder_term'].split())

        for i in range(len(list_of_terms)):
            random.shuffle(list_of_terms[i])
            # list_terms = str(list_of_terms[i])
            list_terms = str(list_of_terms[i]).replace("[", "").replace("]", "")
            # list_terms = str(list_of_terms[i]).replace("'", "")
            # list_terms = str(list_of_terms[i]).replace("[", "").replace("]", "").replace("'", "")

            # prompt = 'What is the intruder term in the following terms? ' + list_terms
            prompt = 'Select which term is the least related to all other terms\nTerms: ' + list_terms + '\nAnswer: '

            topic_intruder_prompt.append(
                {
                    "topic_id": topic_intruder[i]["topic_id"],
                    "topic_terms": topic_intruder[i]['topic_terms'],
                    "intruder_term": topic_intruder[i]['intruder_term'],
                    "prompt": prompt
                }
            )
        with open(self.path_save + self.prompt_name + '.txt', 'w') as f:
            for i in range(len(topic_intruder_prompt)):
                f.write(str(topic_intruder_prompt[i]) + '\n')
            f.close()

        return topic_intruder_prompt


    def run_gpt3(self, tip_dict):
        # run GPT-3 for all topics
        response_list = []
        with open(self.path_save + self.prompt_name + '_gpt3.txt', 'w') as f:
            for i in range(len(tip_dict)):
                prompt = tip_dict[i]['prompt']
                response = str(gpt3(prompt)).replace('\n', '')
                response_list.append(response)
                f.write(response + '\n')
                print(prompt, " ", response)
                time.sleep(3)
            f.close()
        return response_list


    def load_npmis(self):
        with open(self.file_path + 'npmis.txt') as file:
            npmis = file.readlines()
        npmis = [float(term.replace('\n', '')) for term in npmis]
        return npmis


    def compare_intruder(self, tip_dict):
        real_intruder = [tip_dict[i]['intruder_term'] for i in range(len(tip_dict))]
        topic_terms = [tip_dict[i]['topic_terms'] for i in range(len(tip_dict))]

        with open(self.path_save + self.prompt_name + '_gpt3.txt') as file:
            gpt3_intruders = file.readlines()
        gpt3_intruders = [term.replace('\n', '').lower() for term in gpt3_intruders]

        compared_resp_topics = [[term for term in topic_terms[idx] if(term in gpt3_intruders[idx])] for idx in range(len(tip_dict))]
        # put 1 if topic terms are not in gpt3 response
        int_transform_topic = [1 if len(list) == 0 else 0 for list in compared_resp_topics]
        # put 1 if intruder term in gpt3 response
        compared_resp_intruder = [[1 if(real_intruder[idx] in gpt3_intruders[idx]) else 0] for idx in range(len(tip_dict))]
        int_transform_intruder = sum(compared_resp_intruder, [])
        counts = int_transform_intruder and int_transform_topic

        # calculate accuracy
        acc = np.mean(counts)
        variance = np.var(counts)
        spear_rho, spear_p = spearmanr(self.load_npmis(), counts)
        pear_rho, pear_p = pearsonr(self.load_npmis(), counts)
        correlations = {
            "spear_rho": spear_rho,
            "spear_p": spear_p,
            "pear_rho": pear_rho,
            "pear_p": pear_p
        }

        with open(self.path_save + self.prompt_name + '_counts.txt', 'w') as f:
            f.write("Counts: {}\nAccuracy: {}\nVariance: {}\nCorrelations: {}".format(counts, acc, variance, correlations))
        print("\nAccuracy: {}\nVariance: {}\nCorrelations: {}\n".format(acc, variance, correlations))

        return counts


    def run_intrusion(self):
        print("Creation of intruder words...")
        topic_intruder = self.create_intruder()
        print("Creation of prompts for GPT-3...")
        topic_intruder_prompt = self.create_prompt(topic_intruder)
        print("Running GPT-3...")
        self.run_gpt3(topic_intruder_prompt)
        print("Comparing intruder terms...")
        self.compare_intruder(topic_intruder_prompt)

