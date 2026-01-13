import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Download VADER lexicon once
nltk.download('vader_lexicon')

st.set_page_config(page_title="YouTube Sentiment Analyzer")

st.title("📊 YouTube Comment Sentiment Analyzer")
st.write("Paste a YouTube video link to analyze comment sentiment.")

video_url = st.text_input("YouTube Video Link")

if st.button("Analyze"):
    if not video_url:
        st.error("Please enter a YouTube link.")
    else:
        with st.spinner("Fetching comments..."):
            downloader = YoutubeCommentDownloader()
            comments = []

            try:
                for comment in downloader.get_comments_from_url(video_url, sort_by=0):
                    comments.append(comment["text"])
                    if len(comments) >= 200:   # limit for speed
                        break
            except:
                st.error("Failed to fetch comments.")
                st.stop()

        if len(comments) == 0:
            st.warning("No comments found.")
            st.stop()

        sia = SentimentIntensityAnalyzer()

        results = []
        for c in comments:
            score = sia.polarity_scores(c)["compound"]
            if score >= 0.05:
                label = "Positive"
            elif score <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"
            results.append(label)

        df = pd.DataFrame({
            "Comment": comments,
            "Sentiment": results
        })

        counts = df["Sentiment"].value_counts()

        st.success("Analysis Complete!")

        st.subheader("📈 Sentiment Summary")
        st.bar_chart(counts)

        st.subheader("💬 Sample Comments")
        st.dataframe(df.head(20))