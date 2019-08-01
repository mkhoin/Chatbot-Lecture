"""
마르코프 체인 클래스가 정의된 markov_engine.py 파일입니다.
마르코프체인(Markov Chain)은 마르코프성질(Markov Property)을 지닌
이산 확률과정(discrete-time stochastic process)을 의미합니다.

말이 조금 어려운데, 이산 확률과정이란 어떠한 확률분포에 의해 일어나는
일련의 연속 현상을 모델링하는 것을 말하며, 마르코프 성질이란 이 연속 현상이
앞서 일어났던 n 개의 현상에게만 영향을 받음을 의미합니다.

즉, n차 마르코프 체인은 앞에서 일어난 n개의 현상에 의해서 다음 현상이
결정되는 것을 의미하게 됩니다. 아래 코드에서 볼수 있는 예시처럼
우리가 자연어 생성 (NLG : Natural Language Generation)에 3차 마르코프 체인을
적용한다면 앞에서 나왔던 3개의 단어에 의해서만 다음 단어가 결정됨을 의미합니다.

Author : Hyunwoong
When : 7/31/2019
Homepage : https://github.com/gusdnd852
"""

# 패키지에 빨간줄이 그어지면
# ALT + Enter 를 입력한뒤 install package 를 클릭하세요.
import json
import os
import random
from collections import Counter

import pandas as pd
from konlpy.tag import Okt

from src.util.spell_checker import fix


class MarkovEngine:  # 마르코프 체인 클래스 정의
    def __init__(self):
        self.dic = {}
        self.dic = self.ready()  # 단어가 들어갈 딕셔너리 정의

    def ready(self):
        """
        우선 딕셔너리를 준비합니다. 딕셔너리는 {키 : 값}의 자료구조입니다.
        딕셔너리에 대해 잘 모르신다면 아래를 참조하세요.
        https://wikidocs.net/16 (컨트롤 누르고 클릭하면 이동합니다.)

        이미 단어를 학습했다면 학습된 단어를 로드하고
        단어를 학습하지 않았다면, 새로 학습시키고 로드합니다.

        :return: 로드된 딕셔너리
        """
        if os.path.exists("markov_data/markov.json"):  # 학습된 파일이 이미 존재하면
            dics = json.load(open("markov_data/markov.json", "r"))  # 로드합니다.
            return dics  # 로드한 딕셔너리 리턴
        else:  # 존재하지 않으면
            self.train_markov()  # 데이터를 새로 학습하고 (미리 만들어둔 7천 쌍의 대화 데이터입니다)
            dics = json.load(open("markov_data/markov.json", "r"))  # 로드합니다.
            return dics  # 로드한 딕셔너리 리턴

    def register_dic(self, words: str, train: bool = False, percent: float = 0):
        if train:  # 학습중일때만 출력
            print("현재 {0}% 학습 : {1}".format(round(percent, 2), words))

        """
        문장을 딕셔너리에 등록하는 함수입니다.
        문장을 입력받고 3단어 단위로 지속적으로 잘라낸뒤 이것을
        위에서 선언했던 dic 이라는 딕셔너리에 추가합니다.

        :param words: 등록할 문장입니다.
        """
        if len(words) == 0:
            return  # 문장의 길이가 0 (아무것도 입력되지 않았을때) 함수를 종료합니다.
        tmp = ["@"]  # 임시공간 선언, @는 문장의 시작점임을 의미합니다.
        for i in words:  # 입력받은 문장을 단어단위로 쪼개서 아래를 반복합니다
            word = i[0] # 품사 제거, 단어 추출
            if word == "" or word == "\r\n" or word == "\n":
                # 단어가 빈칸이거나 줄바꿈 일경우 (\n 은 줄바꿈문자를 의미합니다)
                continue  # for문의 시작지점으로 돌아갑니다. (이번 단어를 건너 뜁니다)
            tmp.append(word)  # 위 조건에 포함되지 않은(빈칸, 줄바꿈이 아닐때) 임시리스트에 단어를 담고
            if len(tmp) < 3:  # 만약에 3개 이하의 단어가 담겼다면
                continue  # for 문의 시작지점으로 돌아갑니다 (무조건 3개 이상의 단어가 들어올때 까지)

            if len(tmp) > 3:  # 만약 단어의 수가 3개를 넘었다면 (4개의 경우 걸림)
                tmp = tmp[1:]  # 가장 앞단어를 잘라낸 리스트를 만듭니다
                # 때문에 리스트의 사이즈가 반드시 3 이하로 유지됩니다.
                # 아래 예시를 참고하세요.

                # "안녕 반가워 정말 뭐 하고 지냈니"는 다음과 같이 변형됩니다.

                # 1) [@]
                # 1) [@, 안녕]
                # 2) [@, 안녕, 반가워]
                # 3) [@, 안녕, 반가워, 정말] => [안녕, 반가워, 정말]
                # 4) [안녕, 반가워, 정말, 뭐] => [반가워, 정말, 뭐]
                # 5) [반가워, 정말, 뭐, 하고] => [정말, 뭐, 하고]
                # 6) [정말, 뭐, 하고, 지냈니] => [뭐, 히고, 지냈니]

            self.set_word3(tmp)  # 단어를 딕셔너리에 저장합니다. => 아래 함수 참고 !
            if word == "." or word == "?":  # 만약 문장에서 마침표나 물음표를 발견했다면
                tmp = ["@"]  # 그것은 문장의 끝을 의미합니다. 임시공간을 초기화시킵니다.
                continue
        # 딕셔너리가 변경될 때마다 JSON(챗봇의 머릿속이라고 생각해도 좋습니다)
        # 파일에 새로운 딕셔너리들을 저장합니다.
        if not train:
            json.dump(self.dic, open("markov_data/markov.json", "w", encoding="utf-8"))

    def set_word3(self, s3):

        """
        3개의 단어를 입력받아서 다음과 같은3차원의 딕셔너리를 만듭니다.

        단어1 : {
                    단어2 : {
                            단어3: 횟수
                            }
                }

        예시는 다음과 같습니다.
         입력된 데이터들 :

        [진짜, 싫어, 어떻게] x 5
        [진짜, 싫어, 정말] x 3
        [진짜, 싫어, 아니] x 6
        [진짜, 반갑다, 요즘] x 2
        [진짜, 반갑다, 그치] x 5
        [진짜, 반갑다, 그런데] x 1
        [진짜, 좋아 ,사랑해] x 4
        [진짜, 좋아, 정말로] x 2
        [진짜, 좋아, 정말] x 1
        [나는, 너를 정말] x 1
        [나는, 어제, 그러니까] x 2


        만들어진 3차원 딕셔너리 :

        self.dic = {
            진짜 : {         <= 첫번째 나온 단어를 의미힙니다.
                    싫어 : {           <= 두번째 나온 단어를 의미합니다
                            어떻게: 3         <= 세번째 나온 단어와 출현 빈도를 의미합니다.
                            정말 : 3
                            아니 : 6
                            }
                    반갑다 : {
                            요즘 : 2
                            그치 : 5
                            그런데 : 1
                            }
                    좋아 : {
                            사랑해 : 4
                            정말로 : 2
                            정말 : 1
                            }
                    }

            나는 : {         <= 첫번째 나온 단어를 의미힙니다.
                너를 : {           <= 두번째 나온 단어를 의미합니다
                        정말 : 1      <= 세번째 나온 단어와 출현 빈도를 의미합니다.
                    }
                어제 : {
                        그러니까 : 1
                    }
                }
        }

        :param s3: 3개의 연속된 단어가 담긴 리스트 입니다.
        (위 register_dict 함수에서 3개씩 잘라낸 문장들이 여기로 들어옵니다)
        """
        w1, w2, w3 = s3  # 단어 3개를 꺼냅니다
        if w1 not in self.dic:  # 단어1번이 메인 딕셔너리에 없다면
            self.dic[w1] = {}  # 단어 1번에 빈 딕셔너리 부여
        if w2 not in self.dic[w1]:  # 단어 2번이 단어 1번의 딕셔너리에 없다면
            self.dic[w1][w2] = {}  # 단어 2번에 빈 딕셔너리 부여
        if w3 not in self.dic[w1][w2]:  # 단어 3번이 단어 2번의 딕셔너리에 없다면
            self.dic[w1][w2][w3] = 0  # 0번 횟수 지정
        self.dic[w1][w2][w3] += 1  # 만약 있었다면 횟수를 1 올립니다.

    def make_sentence(self, head):
        """
        문장을 만드는 함수힙니다.
        만들 문장의 가장 첫단어만 입력받습니다.

        :param head: 만들 문장의 첫 단어
        :return: 만들어진 문장을 반환합니다
        """

        if head not in self.dic:  # 만약 만들 단어의 첫 단어가 딕셔너리에 없다면
            return ""  # 그냥 빈문장을 반환(대답)합니다
        ret = []  # 반환(대답)할 문장이 들어갈 리스트입니다 (return 의 ret 입니다)
        if head != "@":  # 만약 첫 단어가 @가 아니라면
            ret.append(head)  # 첫번째 단어로 추가합니다.
            # @는 우리가 알기 쉽게 문장의 시작임을 표기하는 기호로만 사용되어야 합니다.
            # 때문에 이를 대답할 리스트에서 제외해줍니다.

        top = self.dic[head]
        # 대답할 가장 첫번째 단어는 딕셔너리의 head 부터 입니다.

        # 우리가 만든 딕셔너리는 다음과 같습니다.
        # 안녕 : {
        #         단어 : {
        #                 단어 : 빈도수
        #                 }
        #        }
        # 헬로 : {
        #          단어 : {
        #                   단어 : 빈도수
        #                   }
        #       }
        # ... 더 많은 단어들
        #
        # 이중에서 만약 "안녕"이 첫번째 단어로 입력되었다면
        # "헬로 : {단어들 ...}" 부터는 고려하지 않고 "안녕 : {단어들 ...}" 부터만 고려한다는 이야기입니다.
        w1 = self.word_choice(top)  # 두번째 단어를 선정합니다 (방법은 아래 함수를 참고하세요)
        w2 = self.word_choice(top[w1])  # 세번째 단어를 선정합니다 (방법은 아래 함수를 참고하세요)
        ret.append(w1)  # 단어를 담습니다 [head, w1]
        ret.append(w2)  # 단어를 담습니다 [head, w1, w2]

        while True:  # 그 뒤로 같은 방법으로 쭉쭉 단어를 붙입니다.
            # 만약 방금 문장 리스트 [head, w1, w2] 에서
            # head 를 제외하고 w1를 첫번쨰 단어로 보고 다시 dic를 참조합니다
            # 이제 w1 이 첫 단어이므로 w1 -> w2 -> w3 로 이어지는 문장을 만듭니다.
            if w1 in self.dic and w2 in self.dic[w1]:
                w3 = self.word_choice(self.dic[w1][w2])
            else:
                w3 = ""  # 만약에 못찾았다면 그냥 w3는 빈칸으로 둡니다 [head, w1, w2]만 이용하는거겠죠?
            ret.append(w3)  # 찾았다면 해당 단어를 넣고 못찾았다면 빈칸을 문장에 넣어줍니다.
            if w3 == "." or w3 == "？ " or w3 == "":  # 만약 찾은 3번째 단어가 . 이나 ? 였다면
                # 문장의 끝을 의미하기 때문에 더이상 단어를 찾이 않고 종료합니다.
                break
            w1, w2 = w2, w3  # w2와 w3가 이제 w1과 w2가 됩니다.
            # 만약 head가 "안녕"이였고 [안녕, 지금, 뭐] 를 찾았고
            # 이 while 문의 시작에서 다시 "지금"을 시작으로 [지금, 뭐, 하고] 를 찾았다면
            # 이제는 "뭐"가 문장의 가장 첫번째가 되어 [뭐, 하고, 있어] 를 찾아냅니다.
            # 각 단어는 계속해서 한칸씩 당겨지면서 다음에 나올 단어를 찾고
            # 단어를 찾던도중 마침표나 물음표를 찾았거나, 단어를 찾지 모했다면
            # 반복을 종료합니다.
        ret = " ".join(ret)  # 반복이 종료되면 모든 단어를 연결합니다.
        # [안녕, 지금, 뭐, 하고, 있어] => 안녕지금뭐하고있어
        return fix(ret)  # 맞춤법을 교정합니다. (띄어쓰기 포함)

    @staticmethod
    def word_choice(sel):  # 단어를 고르는 함수입니다.
        keys = list(sel.keys())  # 입력받은 딕셔너리에서 키들만 뽑아냅니다.
        counts = Counter(keys)
        max_count = max(counts.values())
        result = [x_i for x_i, count in counts.items() if count == max_count]
        return random.choice(result) if type(result) == list else result
        # 뽑아낸 키들 중에서 랜덤하게 원소들을 뽑아냅니다.
        # (책의 소스는 빈도수를 고려하지 않고 랜덤으로 추출합니다.)

        # 예를들어 "안녕" 이 가장 첫단어로 입력되었고
        # 안녕 : {
        #       진짜 : { ... }
        #       반갑다 : { ... }
        #       요즘 : { ... }
        # } 이런식으로 딕셔너리가 구성되어있다면
        # 안녕 다음에 올 단어로 진짜, 반갑다, 요즘 중에서
        # 랜덤으로 단어를 선택하게 됩니다.

    def train_markov(self):
        """
        가장 초반에는 대화할 데이터가 아예 없기 때문에 미리 입력된 말들을 딕셔너리에 넣습니다.
        그러나 고전적인 머신러닝이나 딥러닝에서의 학습(가중치를 구하는 과정)과는 조금 다릅니다.
        초반에 부족한 딕셔너리를 채우기 위해서 많은 데이터를 입력해서 챗봇이 참조할 단어가 담긴
        JSON 파일을 구축하는 것이 목표입니다.
        """
        data = 'ChatbotData.csv'  # 문장이 저장된 데이터파일입니다.
        data_df = pd.read_csv(data)  # 파일을 읽습니다.
        question = data_df['Q'].values  # 질문쌍입니다.
        answer = data_df['A'].values  # 대답쌍입니다.
        data_set = []  # 만들어놓을 데이터셋입니다.
        for i in zip(question, answer):  # 데이터의 문장들을 꺼내서 데이터셋에 담습니다.
            data_set.append(i[0])
            data_set.append(i[1])
        if not (os.path.isdir('markov_data/')):  # 만약 데이터 폴더가 없다면
            os.makedirs(os.path.join('markov_data/'))  # 데이터 폴더를 만듭니다.

        fp_q = open('markov_data/markov.txt', 'a', encoding="utf-8")
        # 데이터 폴더에 텍스트파일을 저장합니다.

        for message in data_set:
            if message:
                fp_q.write(message.replace(':', '').replace(',', '') + '\n')

        full_length = len(data_set)
        print(data_set)
        for idx, i in enumerate(data_set):
            words = Okt().pos(i)
            percent = (idx / full_length) * 100
            print(words)
            self.register_dic(words, train=True, percent=percent)

        json.dump(self.dic, open("markov_data/markov.json", "w", encoding="utf-8"))


    # 응답
    def apply_markov(self, text):
        token = Okt()
        words = token.pos(text)
        self.register_dic(words)
        # 사전에 단어가 있다면 그것을 기반으로 문장 만들기
        for word in words:
            face = word[0]
            if face in self.dic:
                return self.make_sentence(face)
        return self.make_sentence("@")
