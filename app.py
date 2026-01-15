import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from transformers import XLMRobertaTokenizer

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Multilingual YouTube Sentiment Analyzer")

st.title("🌍 Multilingual YouTube Comment Sentiment Analyzer")
st.caption("Understands English, Hindi, Hinglish, emojis and more")

# -------------------------------
# Load sentiment model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # 🔒 Force SLOW tokenizer to avoid fast-tokenizer crash
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation = True,
        max_length = 512
    )

sentiment_model = load_model()

# -------------------------------
# UI
# -------------------------------
st.write("Paste a YouTube video link to analyze comment sentiment.")
video_url = st.text_input("YouTube Video Link")

if st.button("Analyze"):
    if not video_url:
        st.error("Please enter a YouTube link.")
        st.stop()

    # -------------------------------
    # Fetch comments
    # -------------------------------
    with st.spinner("Fetching comments..."):
        downloader = YoutubeCommentDownloader()
        comments = []

        try:
            for comment in downloader.get_comments_from_url(video_url, sort_by=0):
                comments.append(comment["text"])
                if len(comments) >= 200:   # limit for speed
                    break
        except Exception:
            st.error("Could not fetch comments. The video may be private or comments may be disabled.")
            st.stop()

    if len(comments) == 0:
        st.warning("No comments found.")
        st.stop()

    # -------------------------------
    # Sentiment analysis (BATCHED)
    # -------------------------------
    with st.spinner("Analyzing sentiment..."):
        predictions = sentiment_model(comments, batch_size=8)

    results = []
    confidences = []

    for pred in predictions:
        label = pred["label"].lower()
        score = pred["score"]

        if "pos" in label:
            final_label = "Positive"
        elif "neg" in label:
            final_label = "Negative"
        else:
            final_label = "Neutral"

        results.append(final_label)
        confidences.append(round(score, 3))

    # -------------------------------
    # Build dataframe
    # -------------------------------
    df = pd.DataFrame({
        "Comment": comments,
        "Sentiment": results,
        "Confidence": confidences
    })

    counts = df["Sentiment"].value_counts()

    # -------------------------------
    # Results UI
    # -------------------------------
    st.success("Analysis Complete!")

    st.subheader("📊 Sentiment Summary")
    st.bar_chart(counts)

    st.subheader("💬 Sample Comments")
    st.dataframe(df.head(20))

    # -------------------------------
    # Download option
    # -------------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download results as CSV",
        data=csv,
        file_name="youtube_sentiment_results.csv",
        mime="text/csv"
    )
