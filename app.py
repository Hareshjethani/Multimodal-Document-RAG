import os
import json
import gzip
import zipfile
import shutil

import torch
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel

# ==========================================
# CONFIG — apna HF dataset repo yahan daalo
# ==========================================
HF_DATASET_REPO = "haresh8765/visual-rag-data"
INDEX_ZIP_NAME = "colpali_index.zip"
IMAGES_ZIP_NAME = "docvqa_images.zip"

BYALDI_DIR = ".byaldi"
INDEX_NAME = "colpali_docvqa_index"
IMAGES_DIR = "docvqa_images"


# ==========================================
# STEP 1: Download + extract index & images
# (Sirf pehli baar — agar already extracted hai to skip)
# ==========================================
def setup_data():
    if not os.path.exists(os.path.join(BYALDI_DIR, INDEX_NAME)):
        print("⬇️  Downloading ColPali index from Hugging Face...")
        index_zip_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=INDEX_ZIP_NAME,
            repo_type="dataset",
        )
        print("📦 Extracting index...")
        with zipfile.ZipFile(index_zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ Index ready.")
    else:
        print("✅ Index already extracted, skipping download.")

    if not os.path.exists(IMAGES_DIR) or len(os.listdir(IMAGES_DIR)) == 0:
        print("⬇️  Downloading document images from Hugging Face...")
        images_zip_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=IMAGES_ZIP_NAME,
            repo_type="dataset",
        )
        print("📦 Extracting images...")
        with zipfile.ZipFile(images_zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ Images ready.")
    else:
        print("✅ Images already extracted, skipping download.")


setup_data()

# ==========================================
# STEP 2: Load doc_id -> filename mapping
# ==========================================
def load_id_mapping():
    index_file = os.path.join(BYALDI_DIR, INDEX_NAME, "doc_ids_to_file_names.json.gz")
    with gzip.open(index_file, "rt", encoding="utf-8") as f:
        raw_map = json.load(f)

    mapping = {}
    for k, v in raw_map.items():
        # v could already be a full/relative path, or just a filename
        filename = os.path.basename(v)
        mapping[int(k)] = os.path.join(IMAGES_DIR, filename)
    return mapping


print("📖 Loading doc_id -> filename mapping...")
id_mapping = load_id_mapping()
print(f"✅ Mapping loaded: {len(id_mapping)} documents")

# ==========================================
# STEP 3: Load ColPali (from existing index — NO re-indexing)
# ==========================================
print("🔍 Loading ColPali (RAGMultiModalModel) from existing index...")
RAG = RAGMultiModalModel.from_index(INDEX_NAME)
print("✅ ColPali loaded.")

# ==========================================
# STEP 4: Load Qwen2-VL (CPU, NO quantization — Render CPU-only)
# ==========================================
print("🧠 Loading Qwen2-VL-2B-Instruct (CPU mode, this may take a while)...")
QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
print("✅ Qwen2-VL loaded.")


# ==========================================
# STEP 5: RAG search + answer function
# ==========================================
def rag_search_and_answer(user_query):
    if not user_query.strip():
        return "⚠️ Baraye meharbani pehle sawaal likhein!", None

    results = RAG.search(user_query, k=1)
    if not results:
        return "❌ Pre-built index mein koi matching document nahi mila.", None

    top_result = results[0]
    byaldi_id = int(
        top_result["doc_id"]
        if isinstance(top_result, dict)
        else getattr(top_result, "doc_id", top_result.doc_id)
    )

    retrieved_img_path = id_mapping.get(byaldi_id, list(id_mapping.values())[0])
    retrieved_img = Image.open(retrieved_img_path).convert("RGB")
    retrieved_img.thumbnail((1024, 1024))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": retrieved_img},
                {"type": "text", "text": f"Answer briefly based on the document: {user_query}"},
            ],
        }
    ]

    text_prompt = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(qwen_model.device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=40)
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        ans = qwen_processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

    del inputs, generated_ids, trimmed_ids, image_inputs, video_inputs

    return ans, retrieved_img_path


# ==========================================
# STEP 6: Gradio UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="Automated Visual RAG System") as demo:
    gr.Markdown("# 🔍 Automated Document Visual RAG System")
    gr.Markdown("Pehle se Bane hue ColPali Index se Live Search aur Answering.")

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                placeholder="E.g., What is the invoice number? / What date is mentioned?",
                label="Aapka Sawaal (Query)",
            )
            search_btn = gr.Button("Search Pre-Indexed Database 🚀", variant="primary")

            gr.Examples(
                examples=[
                    ["What is the main title of the document?"],
                    ["What date is mentioned in the form?"],
                    ["What is the total amount or price listed?"],
                ],
                inputs=query_input,
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="🤖 Qwen2-VL Answer", interactive=False)
            output_image = gr.Image(label="📌 ColPali Auto-Retrieved Document Page")

    search_btn.click(
        fn=rag_search_and_answer,
        inputs=[query_input],
        outputs=[output_text, output_image],
    )

# Render $PORT env variable use karna zaroori hai
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
