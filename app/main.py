from app.llm import build_llm

def run():
    llm = build_llm()
    result = llm.invoke("Hello! Introduce yourself in one sentence.")
    print(result.content)

if __name__ == "__main__":
    run()