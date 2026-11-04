"""Interface de terminal do Mestre: mesmo agente e memória do Streamlit."""

import time
import uuid
import warnings

from dotenv import load_dotenv

load_dotenv()  # antes de importar src: tools.py lê USE_DUMMY no import

from src.agent import TOOL_CATEGORY, build_mestre_with_memory

# APIs classic deprecated por escolha (rubrica) e aviso do torch.jit: ruído no chat, não erro.
# Depois dos imports: o langchain_core registra o próprio filtro de deprecation ao ser importado.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    mestre = build_mestre_with_memory()
    session = f"cli-{uuid.uuid4().hex[:8]}"
    print("🥋 Mestre (Ctrl+C ou 'sair' para encerrar | 'nova' reinicia a conversa)\n")

    while True:
        try:
            question = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            break
        if question.lower() == "nova":
            session = f"cli-{uuid.uuid4().hex[:8]}"
            print("(nova conversa)\n")
            continue

        t0 = time.perf_counter()
        try:
            r = mestre.invoke(
                {"input": question}, config={"configurable": {"session_id": session}}
            )
        except Exception as e:  # 429 do Groq, rede etc.
            print(f"[erro] {type(e).__name__}: {e}\n")
            continue

        cats = sorted({TOOL_CATEGORY.get(t, "?") for t in r["tools"]})
        rate = "-" if r["support_rate"] is None else f"{r['support_rate']:.0%}"
        print(f"\nMestre: {r['output']}\n")
        print(
            f"  [tools={r['tools']} | categorias={cats} | veredito={r['verdict']} "
            f"| suporte={rate} | revisada={r['revised']} | {time.perf_counter() - t0:.1f}s]\n"
        )
    print("Oss! 🥋")


if __name__ == "__main__":
    main()
