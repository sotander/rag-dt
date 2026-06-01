import ipywidgets as widgets
from IPython.display import display, Markdown


class RagUI:
    def __init__(self, store, llm_model, llm_tokenizer, reranker, top_k=20, rerank_k=5):
        self.store = store
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_k = rerank_k

    def launch(self):
        question_box = widgets.Textarea(
            placeholder="Ask a question...",
            layout=widgets.Layout(width="100%", height="100px"),
        )

        button = widgets.Button(description="Ask")
        output = widgets.Output()

        def on_click(_):
            question = question_box.value.strip()
            if not question:
                return

            with output:
                output.clear_output()

                answer = ask(
                    question,
                    self.store,
                    self.llm_model,
                    self.llm_tokenizer,
                    self.reranker,
                    top_k=self.top_k,
                    rerank_k=self.rerank_k,
                )

                display(Markdown(f"### Answer\n\n{answer}"))

        button.on_click(on_click)

        display(question_box)
        display(button)
        display(output)
