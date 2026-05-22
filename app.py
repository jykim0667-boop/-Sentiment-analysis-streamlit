import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
from collections import Counter
from konlpy.tag import Okt
import re

# ==========================================
# API KEY
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

@st.cache_resource
def load_model():

    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

sentiment_analyzer = load_model()

# ==========================================
# 영상 ID 추출
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
# 댓글 가져오기
# ==========================================

def get_comments(video_id, max_comments=100):

    comments = []

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

            if comment.strip():
                comments.append(comment)

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

    return comments

# ==========================================
# 감정 분석
# ==========================================

def analyze_sentiment(comments):

    results = []

    for comment in comments:

        try:

            result = sentiment_analyzer(comment[:512])[0]

            label = result['label']

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
# Streamlit UI
# ==========================================

st.title("🎬 유튜브 댓글 감정분석 AI")

url = st.text_input("유튜브 영상 링크 입력")

if st.button("분석 시작"):

    video_id = extract_video_id(url)

    if not video_id:

        st.error("올바른 유튜브 링크가 아닙니다.")

    else:

        with st.spinner("댓글 분석 중..."):

            comments = get_comments(video_id)

            sentiments = analyze_sentiment(comments)

            counter = Counter(sentiments)

            total = sum(counter.values())

            if total == 0:

                st.warning("댓글이 없는 영상입니다.")

            else:

                positive = counter['긍정']
                neutral = counter['중립']
                negative = counter['부정']

                st.subheader("📊 감정 분석 결과")

                st.write(f"총 댓글 수: {total}")
                st.write(f"긍정 😊 : {positive}")
                st.write(f"중립 😐 : {neutral}")
                st.write(f"부정 😡 : {negative}")

                # 키워드 분석

                okt = Okt()

                nouns = []

                stopwords = [
                    "영상", "댓글", "진짜", "그냥",
                    "사람", "생각", "이거"
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

                st.subheader("🔥 핵심 키워드")

                for word, count in top_keywords:
                    st.write(f"{word} ({count}회)")

                keywords = [word for word, count in top_keywords]

                st.subheader("🤖 AI 3줄 요약")

                if positive > negative:
                    st.write("1. 시청자 반응은 전반적으로 긍정적입니다.")
                else:
                    st.write("1. 다양한 의견이 나타나는 영상입니다.")

                if len(keywords) >= 3:
                    st.write(
                        f"2. '{keywords[0]}', '{keywords[1]}', '{keywords[2]}' 관련 언급이 많았습니다."
                    )

                st.write("3. 댓글 반응을 통해 높은 관심도를 확인할 수 있습니다.")
