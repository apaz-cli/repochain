#!./bin/python3
key_path = ".api_key"
repo_url = "https://git.savannah.gnu.org/git/nano.git"
repo_path = "/tmp/nano"

# Read OAI api key
import os
if not os.getenv("OPENAI_API_KEY"):
    with open(key_path, "r") as f:
        os.environ["OPENAI_API_KEY"] = str(f.read().strip())


# Clone repo
from git import Repo
from os.path import isdir
def load_repo(repo_path):
    if not isdir(repo_path):
        return Repo.clone_from(repo_url, to_path=repo_path)
    else:
        return Repo(repo_path)
repo = load_repo(repo_path)


# Create langchain loader
from langchain.document_loaders import GitLoader
loader = GitLoader(repo_path=repo_path, branch=repo.head.reference)


# Load/filter documents
extensions = ['.c', '.h', '.cpp', '.py', '.sh', '.java', '.js', '.mjs', '.html', '.css', '.php', '.pl', '.pm', '.rb', '.lua', '.rs', '.go' '.md']
documents = [f for f in loader.load() if f.metadata['file_type'] == '.c']
print(f"Loaded {len(documents)} documents.")


# Split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)
print(f"Split into {len(texts)} texts.")


# Embed documents
from langchain.embeddings import OpenAIEmbeddings
embed_model = OpenAIEmbeddings()
embeddings = None
import pickle
if os.path.exists("embeddings.pkl"):
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
else:
    embeddings = embed_model.embed_documents([t.page_content for t in texts])
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
        print("Saved embeddings.")
print(f"Loaded {len(embeddings)} embeddings.")

# Create vector db
from langchain.vectorstores import Annoy
docsearch = Annoy.from_embeddings(
                [(t.page_content, e) for t, e in zip(texts, embeddings)],
                embed_model,
                metadatas=[t.metadata for t in texts])

# Create LLM
from langchain.llms import OpenAI
llm = OpenAI()

# Create prompt
from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of code to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer: """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": prompt}


# Create QA chain (document vector DB -> LLM)
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=docsearch.as_retriever(),
                                    chain_type_kwargs=chain_type_kwargs,
                                    return_source_documents=True)

# Execute the langchain
result = chain.run(prompt)

text_response = result["result"]
docs_used = result["source_documents"]

print(f"Response:\n{text_response.page_content}")


