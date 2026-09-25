"""Display the entire saved instance and the exact input behind its vector."""

import json


def render_instance(st, row, definition, *, key):
    instance = row.get("instance")
    st.caption(f"Source row {row['source_row']} · {row['chunks']} token chunk(s)")
    if instance is not None and definition:
        st.caption(f"Instance ID: {instance['id']} · Split: {instance.get('split') or 'unknown / not assigned'}")
        schema = definition["schema"]
        if schema in {"pretraining", "task_specific_supervised"}:
            if schema == "task_specific_supervised":
                st.write({"Task": instance["task"], "Label": instance["label"]})
            st.text_area("Complete instance text", value=instance["text"], height=280, disabled=True, key=key + ":text")
        elif schema == "instruction_finetuning":
            for index, message in enumerate(instance["messages"]):
                st.caption(f"Turn {index + 1} · {message['role']}")
                if isinstance(message["content"], str):
                    st.text(message["content"])
                else:
                    st.json(message["content"], expanded=True)
        else:
            for name in ("prompt", "chosen", "rejected"):
                st.caption(name.capitalize())
                if isinstance(instance[name], str):
                    st.text(instance[name])
                else:
                    st.json(instance[name], expanded=True)
        with st.expander("Complete canonical instance and metadata", expanded=False):
            st.json(instance, expanded=True)
    else:
        st.text_area("Saved instance text", value=row["text"], height=280, disabled=True, key=key + ":text")
    with st.expander("Full original source record", expanded=instance is None):
        st.json(row["record"], expanded=True)
    with st.expander("Exact text used for the similarity embedding"):
        st.text(row["text"])
    st.download_button("Download complete instance", data=json.dumps(
        {"instance": instance, "original_record": row["record"], "embedding_text": row["text"],
         "source_row": row["source_row"], "embedding_id": row["embedding_id"]}, ensure_ascii=False, indent=2),
        file_name=f"instance-{row['embedding_id']}.json", mime="application/json", key=key + ":download")
