# ==========================================
# 유튜브 댓글 감정분석 AI (최종 안정화 버전)
# ==========================================

# ==========================================
# 필요한 라이브러리 설치
# ==========================================

!pip install google-api-python-client transformers torch konlpy wordcloud -q

# ==========================================
# 라이브러리 불러오기
# ==========================================

from googleapiclient.discovery import build
from transformers import pipeline
from collections import Counter
from konlpy.tag import Okt
import re

# ==========================================
# YouTube API 키 입력
# ==========================================

API_KEY = "AIzaSyAV-V1l1Tsb8MVY4KtLCF1C5c6Apl3UmMQ"

# ==========================================
# 유튜브 API 연결
# ==========================================

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

# ==========================================
# 감정분석 모델 로드
# ==========================================

print("AI 감정분석 모델 로딩 중...")

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

print("모델 로딩 완료!")

# ==========================================
# 영상 ID 추출 함수
# ==========================================

def extract_video_id(url):

    patterns = [
        r"v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"shorts/([a-zA-Z0-9_-]+)",
        r"embed/([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None

# ==========================================
# 댓글 가져오기 함수
# ==========================================

def get_comments(video_id, max_comments=100):

    comments = []

    try:

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )

        response = request.execute()

        while response and len(comments) < max_comments:

            for item in response["items"]:

                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

                # 공백 제거
                comment = comment.strip()

                if comment:
                    comments.append(comment)

                if len(comments) >= max_comments:
                    break

            if "nextPageToken" in response:

                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=response["nextPageToken"],
                    textFormat="plainText"
                )

                response = request.execute()

            else:
                break

    except Exception as e:
        print("댓글을 가져오는 중 오류 발생:")
        print(e)

    return comments

# ==========================================
# 감정 분석 함수
# ==========================================

def analyze_sentiment(comments):

    results = []

    for comment in comments:

        try:

            # 너무 긴 댓글 자르기
            comment = comment[:512]

            result = sentiment_analyzer(comment)[0]

            label = result['label']

            # 별점 기반 감정 분류
            if label in ['4 stars', '5 stars']:
                emotion = '긍정'

            elif label == '3 stars':
                emotion = '중립'

            else:
                emotion = '부정'

            results.append(emotion)

        except:
            results.append("중립")

    return results

# ==========================================
# 결과 요약 함수
# ==========================================

def summarize_results(sentiments, comments):

    counter = Counter(sentiments)

    total = sum(counter.values())

    # 댓글 없는 경우
    if total == 0:
        print("\n댓글이 없는 영상입니다.")
        return

    positive = counter['긍정']
    neutral = counter['중립']
    negative = counter['부정']

    positive_percent = (positive / total) * 100
    neutral_percent = (neutral / total) * 100
    negative_percent = (negative / total) * 100

    print("\n==============================")
    print("유튜브 댓글 감정분석 결과")
    print("==============================\n")

    print(f"총 댓글 수: {total}")
    print(f"긍정 😊 : {positive}개 ({positive_percent:.1f}%)")
    print(f"중립 😐 : {neutral}개 ({neutral_percent:.1f}%)")
    print(f"부정 😡 : {negative}개 ({negative_percent:.1f}%)")

    # =====================================
    # 키워드 분석
    # =====================================

    okt = Okt()

    nouns = []

    # 불용어 제거
    stopwords = [
        "영상", "댓글", "진짜", "그냥", "사람",
        "생각", "이거", "이번", "부분", "느낌",
        "너무", "약간", "진심", "요즘", "처음",
        "이건", "저는", "제가", "완전"
    ]

    for comment in comments:

        try:

            words = okt.nouns(comment)

            for word in words:

                if len(word) > 1 and word not in stopwords:
                    nouns.append(word)

        except:
            pass

    keyword_counter = Counter(nouns)

    top_keywords = keyword_counter.most_common(5)

    print("\n==============================")
    print("댓글 핵심 키워드")
    print("==============================\n")

    if len(top_keywords) == 0:
        print("추출된 키워드가 없습니다.")

    else:
        for word, count in top_keywords:
            print(f"{word} ({count}회)")

    # =====================================
    # AI 3줄 요약
    # =====================================

    keywords = [word for word, count in top_keywords]

    print("\n==============================")
    print("AI 3줄 요약")
    print("==============================\n")

    # 첫 번째 줄
    if positive_percent >= 60:
        line1 = "시청자들은 영상에 대해 전반적으로 긍정적인 반응을 보이고 있습니다."

    elif negative_percent >= 50:
        line1 = "영상에 대한 비판적 의견과 부정적인 반응이 비교적 많이 나타납니다."

    else:
        line1 = "긍정과 부정 의견이 혼합된 반응을 보이고 있습니다."

    # 두 번째 줄
    if len(keywords) >= 3:
        line2 = f"댓글에서는 '{keywords[0]}', '{keywords[1]}', '{keywords[2]}' 관련 언급이 많이 등장했습니다."

    elif len(keywords) > 0:
        line2 = f"주요 키워드로는 '{keywords[0]}' 등이 자주 언급되었습니다."

    else:
        line2 = "특정 키워드 분석이 충분하지 않았습니다."

    # 세 번째 줄
    if positive > negative:
        line3 = "전체적으로 영상의 내용과 분위기에 만족하는 시청자가 많은 편입니다."

    else:
        line3 = "영상 내용에 대해 다양한 의견과 평가가 나타나고 있습니다."

    print("1.", line1)
    print("2.", line2)
    print("3.", line3)

# ==========================================
# 메인 실행
# ==========================================

print("\n==============================")
print("유튜브 댓글 감정분석 AI")
print("==============================\n")

url = input("유튜브 영상 링크를 입력하세요: ")

video_id = extract_video_id(url)

if not video_id:

    print("\n올바른 유튜브 링크가 아닙니다.")

else:

    print("\n댓글 수집 중...\n")

    comments = get_comments(video_id, max_comments=100)

    print(f"{len(comments)}개의 댓글 수집 완료!")

    if len(comments) == 0:
        print("댓글이 없거나 댓글 사용이 중지된 영상입니다.")

    else:

        print("\n감정 분석 중...\n")

        sentiments = analyze_sentiment(comments)

        summarize_results(sentiments, comments)
