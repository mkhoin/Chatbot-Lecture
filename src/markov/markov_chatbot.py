# Author : Hyunwoong
# When : 7/31/2019
# Homepage : github.com/gusdnd852
import random
from collections import Counter

from src.markov.markov_engine import MarkovEngine
from src.util.spell_checker import fix


def generate_answer(engine, text):
    """
    마르코프 체인을 이용해서 답변을 생성합니다.
    총 5번 답변을 출력하여 그 중 가장 자주 나온 문장을 출력합니다.
    모두 1번씩만 나왔다면 아니라면 랜덤으로 출력합니다.
    같은 입력으로 같은문장이 여러번 나온다면, 그 입력이 나올 확률이
    높다고 생각해서 다음과 같은 구현을 작성하였습니다.


    :param engine:  마르코프 체인 클래스
    :param text: 사용자의 입력
    :return: 인공지능의 출력
    """
    counter = 0  # 루프 갯수를 세줄 변수
    markov_list = []
    while True:
        counter += 1
        markov = engine.apply_markov(text)
        if markov != text:
            markov_list.append(markov)
        if len(markov_list) == 5:
            break
        if counter > 5:
            while True:
                markov_list.append(markov)
                if len(markov_list) >= 5:
                    break

    answer = [fix(i) for i in markov_list]
    counts = Counter(answer)
    max_count = max(counts.values())
    result = [x_i for x_i, count in counts.items() if count == max_count]

    if len(answer) == 1:
        result_answer = text + '?' if result[0] == text else result[0]  # 최빈 값 출력
    else:  # 데이터와 마르코프 최빈값의 결과 수 가 같을때
        result_answer = random.choice(result)
        result = text + '?' if result_answer == text else result_answer  # 최빈 값 출력
    return fix(result)

if __name__ == '__main__':
    print('말을 배우는 중입니다... (시간이 조금 걸립니다)')
    engine = MarkovEngine()
    print('단어를 모두 배웠습니다. ')
    print('초반에는 띄어쓰기로 구분된 2단어 이상의 말을 해서 말을 가르쳐주세요.')
    while True:
        print('사용자 : ', end='', )
        req = fix(input())
        answer = generate_answer(engine, req)
        print('인공지능 : {0}'.format(answer))