from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import re

app = Flask(__name__)

# Load summarizer once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def clean_summary(summary: str) -> str:
    """Ensure summary ends with a proper sentence."""
    summary = summary.strip()
    if not summary.endswith(('.', '!', '?')):
        summary += "..."
    return summary

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.json
        text = data.get("text", "")
        mode = data.get("mode", "brief")

        if not text.strip():
            return jsonify({"error": "No text provided."}), 400

        # Set length limits
        if mode == "brief":
            max_len, min_len = 120, 50
        elif mode == "detailed":
            max_len, min_len = 250, 100
        elif mode == "bullet":
            max_len, min_len = 180, 70
        else:
            max_len, min_len = 150, 60

        # Handle long input by chunking
        words = text.split()
        if len(words) > 500:
            chunks = [words[i:i+400] for i in range(0, len(words), 400)]
            summaries = []
            for chunk in chunks:
                chunk_text = " ".join(chunk)
                result = summarizer(chunk_text, max_length=max_len, min_length=min_len, do_sample=False)
                summaries.append(result[0]['summary_text'])

            combined = " ".join(summaries)
            final = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
            summary = clean_summary(final[0]['summary_text'])
        else:
            result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            summary = clean_summary(result[0]['summary_text'])

        # For bullet mode → split into points
        if mode == "bullet":
            points = re.split(r'(?<=[.!?]) +', summary)
            summary = "\n• " + "\n• ".join([p.strip() for p in points if p.strip()])

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
