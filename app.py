import gradio as gr
from tiktoken import encoding_for_model
import random
import html
import os  # Needed for deployment environments

def tokenize_text_colored(text, model_name="gpt-3.5-turbo"):
    if not text.strip():
        return "No text provided.", 0

    try:
        # Get the tokenizer for the selected model
        enc = encoding_for_model(model_name)
        token_ids = enc.encode(text)
        tokens = [enc.decode_single_token_bytes(t).decode('utf-8', errors='replace') for t in token_ids]

        # Generate random pastel colors for each token
        colors = []
        for _ in tokens:
            r = random.randint(150, 255)
            g = random.randint(150, 255)
            b = random.randint(150, 255)
            colors.append(f"rgb({r},{g},{b})")

        # Wrap tokens in colored spans
        colored_tokens = ""
        for t, c in zip(tokens, colors):
            safe_t = html.escape(t).replace(" ", "\u00A0")  # preserve spaces
            colored_tokens += f'<span style="background-color:{c};padding:2px;margin:1px;border-radius:4px;display:inline-block;">{safe_t}</span>'

        total_tokens = len(tokens)
        return colored_tokens, total_tokens

    except Exception as e:
        return f"Error: {str(e)}", 0

# -------------------
# Gradio Interface
# -------------------
with gr.Blocks() as demo:
    gr.Markdown("# Nahla Ali Tokenizer Playground")
    gr.Markdown(
        "This playground is part of the Applied AI course by @Nahla Ali.\n\n"
        "Enter text below to see its tokenization with colors, similar to OpenAI Playground. "
        "Each token is highlighted in a random color."
    )

    # Input row
    with gr.Row():
        text_input = gr.Textbox(label="Input Text", placeholder="Enter text to tokenize...", lines=5)
        model_input = gr.Dropdown(
            label="Model",
            choices=[
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-32k",
                "text-davinci-003",
                "text-davinci-002",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "code-davinci-002"
            ],
            value="gpt-3.5-turbo"
        )

    # Output row
    with gr.Row():
        tokens_output = gr.HTML(label="Colored Tokens")
        summary_output = gr.Textbox(label="Total Tokens", interactive=False, lines=2)

    # Connect function
    text_input.change(fn=tokenize_text_colored, inputs=[text_input, model_input], outputs=[tokens_output, summary_output])
    model_input.change(fn=tokenize_text_colored, inputs=[text_input, model_input], outputs=[tokens_output, summary_output])

if __name__ == "__main__":
    demo. launch(
        server_name="0.0.0.0",                  # required for deployment platforms
        server_port=int(os.environ.get("PORT", 7860)),  # Render or other platform provides PORT
        share=True                               # generates temporary public URL
    )
