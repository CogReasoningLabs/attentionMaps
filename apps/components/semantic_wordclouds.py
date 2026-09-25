"""Render only script-produced cluster word clouds and term frequencies."""

import json
import shlex


def render_cluster_wordcloud(st, root, cluster, *, embedding_root=None):
    st.subheader(f"Word cloud · Cluster {cluster}")
    summary = root / "wordclouds" / "summary.json"
    embedding_root = embedding_root or root
    command = ["venv/bin/python", "scripts/cluster_dataset.py", "wordclouds", "--run", str(embedding_root)]
    if root != embedding_root:
        command.extend(["--clustering", root.name])
    if not summary.is_file():
        st.info("This result has no saved word clouds yet. Generate them for these exact cluster assignments with the command below, then refresh this page.")
        st.code(shlex.join(command), language="bash")
        return
    report = json.loads(summary.read_text(encoding="utf-8"))
    from attention_maps.explorer.semantic_wordclouds import configured_stopwords
    if frozenset(report.get("stopwords", [])) != configured_stopwords():
        st.warning("These saved clouds use a different stopword list. Run the command below to apply the current list, then refresh this page.")
        st.code(shlex.join([*command, "--refresh"]), language="bash")
    item = report["clusters"].get(str(cluster))
    if not item or not item["top_words"]:
        st.info("No words remain after filtering in this cluster.")
        return
    st.caption(f"All {item['records']:,} cluster records · {item['counted_tokens']:,} counted word occurrences · up to {report['max_displayed_words']} displayed words. Size represents frequency, not semantic importance.")
    if item.get("image"):
        image = root / "wordclouds" / f"cluster-{cluster}.png"
        st.image(str(image), width="stretch")
        st.download_button("Download cluster word cloud", image.read_bytes(), file_name=f"{root.name}-cluster-{cluster}.png", mime="image/png")
    elif item.get("note"):
        st.info(item["note"])
    with st.expander("Cluster word frequencies and counting rules"):
        st.dataframe([{"Word": word, "Occurrences": count} for word, count in item["top_words"]], hide_index=True, width="stretch")
        st.caption(report["tokenization"])
        st.caption(report["content"])
        st.write({"Excluded stopwords": report["stopwords"]})
