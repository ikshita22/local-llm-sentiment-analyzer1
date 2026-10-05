"""Local sentiment and custom zero-shot classification dashboard."""
import hashlib
import math
import time

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError, ParserError
from skllm.datasets import get_classification_dataset
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier

from src.app_helpers import SENTIMENTS, empty_text_mask, filter_results, generate_explanation, make_excel_download, normalise_sentiment, quality_flags, read_csv_bytes
from src.history_store import list_runs, load_run, save_run

DEFAULT_MODEL = "llama3.2:3b"
MODELS = [DEFAULT_MODEL, "llama3.2:1b"]
st.set_page_config(page_title="Local LLM Sentiment Analyzer", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_classifier(model, labels, temperature):
    classifier = ZeroShotOllamaClassifier(model=model, options={"temperature": temperature})
    if list(labels) == [x.lower() for x in SENTIMENTS]:
        x, y = get_classification_dataset()
        classifier.fit(x, y)
    else:
        classifier.fit(None, list(labels))
    return classifier


def analyse(data, text_column, label_column, labels, model, temperature, batch_size, explanations):
    started = time.perf_counter()
    result = data.copy()
    empty = empty_text_mask(data, text_column)
    result[label_column] = "Neutral" if label_column == "Sentiment" else "Not analyzed"
    rows = data.loc[~empty, text_column].astype(str)
    classifier = get_classifier(model, tuple(x.lower() for x in labels), temperature)
    status = st.status("Preparing local analysis…", expanded=True)
    predicted = []
    total = math.ceil(len(rows) / batch_size) if len(rows) else 0
    for number, start in enumerate(range(0, len(rows), batch_size), 1):
        status.update(label=f"Analyzing batch {number} of {total}…", state="running")
        predicted += classifier.predict(rows.iloc[start:start + batch_size].tolist()).tolist()
    if label_column == "Sentiment":
        predicted = [normalise_sentiment(item) for item in predicted]
    result.loc[~empty, label_column] = predicted
    if explanations:
        status.update(label="Generating local explanations…", state="running")
        result["Explanation"] = "Not generated for empty text."
        for index in result.index[~empty]:
            result.loc[index, "Explanation"] = generate_explanation(model, str(result.at[index, text_column]), str(result.at[index, label_column]), temperature)
    status.update(label="Local analysis complete.", state="complete", expanded=False)
    return result, time.perf_counter() - started


st.title("Local LLM Sentiment Analyzer")
st.write("Analyze CSV text locally with Ollama, Llama 3.2, and Scikit-LLM. Your data stays on your machine.")
with st.sidebar:
    st.header("Local model settings")
    model = st.selectbox("Ollama model", MODELS)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    batch_size = st.select_slider("Batch size", [1, 5, 10, 25], value=5)
    st.caption("The model must be installed locally.")
    st.code(f"ollama pull {model}", language="bash")
    st.divider()
    st.subheader("Saved local runs")
    history = list_runs()
    if history.empty:
        st.caption("Completed analyses appear here.")
    else:
        choices = {f"#{r.id} · {r.created_at} · {r.model}": r.id for r in history.itertuples()}
        choice = st.selectbox("Load history", ["Choose a run", *choices])
        if choice != "Choose a run" and st.button("Load saved results"):
            st.session_state.results, st.session_state.metadata = load_run(choices[choice])
            st.rerun()

st.info(f"Selected model: `{model}` · Ollama backend · Local only")
upload = st.file_uploader("Upload a CSV file", type="csv")
if upload is not None:
    content = upload.getvalue()
    identifier = (upload.name, upload.size, hashlib.sha256(content).hexdigest())
    if st.session_state.get("upload_id") != identifier:
        st.session_state.upload_id = identifier
        st.session_state.pop("results", None)
        st.session_state.pop("metadata", None)
    try:
        data = read_csv_bytes(content)
    except (EmptyDataError, ParserError, UnicodeDecodeError) as error:
        st.error(f"We could not read that CSV: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Unexpected CSV error: {error}")
        st.stop()
    if data.empty:
        st.warning("The CSV has headers but no rows to analyze.")
        st.stop()

    st.subheader("Upload settings")
    columns = list(data.columns.astype(str))
    text_column = st.selectbox("Text column", columns, index=columns.index("Review") if "Review" in columns else 0)
    mode, group_column, date_column = st.columns(3)
    mode = mode.selectbox("Analysis mode", ["Sentiment", "Custom categories"])
    group_column = group_column.selectbox("Break down by", ["None", *[x for x in columns if x != text_column]])
    date_column = date_column.selectbox("Date trend", ["None", *[x for x in columns if x != text_column]])
    if mode == "Custom categories":
        labels = [x.strip() for x in st.text_input("Custom labels", "Question, Complaint, Praise").split(",") if x.strip()]
        label_column = "Classification"
        if len(labels) < 2:
            st.error("Enter at least two comma-separated custom labels.")
            st.stop()
    else:
        labels, label_column = SENTIMENTS, "Sentiment"
    explanations = st.checkbox("Generate a short local explanation for each result")
    flags = quality_flags(data, text_column)
    st.caption("Data quality: " + " · ".join(f"{key}: {value}" for key, value in flags.sum().items()))
    with st.expander("Review data-quality flags"):
        st.dataframe(pd.concat([data, flags], axis=1), use_container_width=True, hide_index=True)
    st.dataframe(data, use_container_width=True, hide_index=True)
    if empty_text_mask(data, text_column).any():
        st.warning("Empty text is not sent to the model.")
    if st.button("Analyze locally", type="primary"):
        try:
            results, duration = analyse(data, text_column, label_column, labels, model, temperature, batch_size, explanations)
            st.session_state.results = results
            st.session_state.metadata = {"model": model, "text_column": text_column, "label_column": label_column, "labels": labels, "duration": duration, "group": group_column, "date": date_column}
            save_run(results, model, text_column, label_column, labels, duration)
            st.success(f"Analysis complete in {duration:.1f} seconds and saved locally.")
        except Exception as error:
            st.error(f"Analysis failed. Ensure Ollama is running and `{model}` is installed. Details: {error}")

if "results" not in st.session_state:
    st.stop()
results, meta = st.session_state.results, st.session_state.metadata
text_column, label_column, labels = meta["text_column"], meta["label_column"], meta["labels"]
st.subheader("Analysis dashboard")
st.caption(f"Model: {meta['model']} · {len(results)} rows · {meta['duration']:.1f}s · {len(results) / max(meta['duration'], .01):.1f} rows/sec")
counts = results[label_column].value_counts()
for column, label in zip(st.columns(len(labels)), labels):
    column.metric(label, f"{counts.get(label, 0) / len(results):.0%}", help=f"{counts.get(label, 0)} rows")
chart = pd.DataFrame({label_column: labels}).assign(Reviews=lambda x: x[label_column].map(counts).fillna(0)).set_index(label_column)
st.bar_chart(chart, y="Reviews")
if meta.get("group") != "None" and meta["group"] in results:
    st.subheader(f"Breakdown by {meta['group']}")
    st.bar_chart(pd.crosstab(results[meta['group']].fillna("Missing"), results[label_column]))
if meta.get("date") != "None" and meta["date"] in results:
    dated = results.assign(_date=pd.to_datetime(results[meta['date']], errors="coerce")).dropna(subset=["_date"])
    if not dated.empty:
        st.subheader("Classification trend")
        st.line_chart(pd.crosstab(dated["_date"].dt.date, dated[label_column]))

st.subheader("Search and filter")
search, label_filter, group_filter = st.columns(3)
search = search.text_input("Search text")
label_filter = label_filter.selectbox(label_column, ["All", *labels])
if meta.get("group") != "None" and meta["group"] in results:
    group_filter = group_filter.selectbox(meta["group"], ["All", *sorted(results[meta['group']].fillna("Missing").astype(str).unique())])
else:
    group_filter = "All"
filtered = filter_results(results, text_column, label_column, search, label_filter)
if group_filter != "All":
    filtered = filtered[filtered[meta["group"]].fillna("Missing").astype(str) == group_filter]
st.caption(f"Showing {len(filtered)} of {len(results)} rows. Filtering does not re-run the model.")
st.dataframe(filtered, use_container_width=True, hide_index=True)
col1, col2, col3 = st.columns(3)
col1.download_button("Download all CSV", results.to_csv(index=False).encode(), "sentiment_results.csv", "text/csv")
col2.download_button("Download filtered CSV", filtered.to_csv(index=False).encode(), "filtered_results.csv", "text/csv")
col3.download_button("Download Excel report", make_excel_download(results, label_column), "sentiment_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
