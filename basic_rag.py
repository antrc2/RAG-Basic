import llama_cpp
import pandas as pd
from tabulate import tabulate
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document as shemaDocument
from PyPDF2 import PdfReader
import json
import faiss
import numpy as np
import os
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

headers = {"Content-Type": "application/json"}
embedding_model_path = "nomic-embed-text-v1.5.f16.gguf"
faiss_name = "faiss_index.index"
database_json = 'database.json'
api_url = "https://0b22-34-141-165-187.ngrok-free.app/v1"
dataset_path = 'datasets'
llm_chat = OpenAI(base_url=api_url,api_key="hehe")
model_name = llm_chat.models.list().data[0].id
llm_embedding = llama_cpp.Llama(embedding_model_path,embedding=True,n_ctx=2048,n_gpu_layers=-1)

def embeddingText(text):
    response = (llm_embedding.create_embedding(text)['data'][0]["embedding"])
    return response
def text_split(docs):
    text_splitter = CharacterTextSplitter(
        chunk_size=2048, 
        chunk_overlap=500,
        separator="\n",
        length_function=len
    )
    texts = text_splitter.split_documents(docs)
    return texts
def excel_to_object(excel_path):
    sheets = pd.read_excel(excel_path, sheet_name=None)  # Đọc tất cả các sheet
    markdown_tables = {}
    i=0
    for sheet_name, df in sheets.items():
        markdown_tables[i] = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
        i+=1
    document = []
    for i in range(len(markdown_tables)):
        markdown = markdown_tables[i]
        document.append(
            shemaDocument(
                page_content=markdown,
                metadata= {"source": excel_path, "sheet_index": i}
            )
        )
    return document
def pdf_to_object(documents):
    documents = [
    shemaDocument(page_content=document['page_content'], metadata=document['metadata']) 
    for document in documents
]
    return documents
def pdf_to_raw_text(pdf_path):
    documents = []
    pdf_reader = PdfReader(pdf_path)
    
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        
        document = {
            'page_content': page_text,
            'metadata': {
                'source': pdf_path,
                'page': page_num
            }
        }
        documents.append(document)
    
    return documents
def dataset_path_to_database(path):
    files = os.listdir(path)
    datasets_files = []
    for file in files:
        # Lấy phần mở rộng của file
        file_extension = os.path.splitext(file)[1].lower()  # Chuyển thành chữ thường để so sánh chính xác
        
        if file_extension == '.pdf':
            datasets_files.append({'type': 'pdf', 'file_name': file})
        elif file_extension in ['.xls', '.xlsx', '.xlsm']:  # Các định dạng Excel phổ biến
            datasets_files.append({'type': 'excel', 'file_name': file})
    db_json = []
    for file in datasets_files:
        if file['type'] == "pdf":
            raw_text = pdf_to_object(pdf_to_raw_text(path + "/" + file['file_name']))
            response = text_split(raw_text)
            for res in response:
                db_json.append(
                    {
                        "page_content": res.page_content,
                        "metadata": {'source': path + "/" + file['file_name']}
                        
                    }
                )
        elif file['type'] == "excel":
            markdowns = excel_to_object(path + "/" + file['file_name'])
            # print(markdowns)
            for i in range(len(markdowns)):
                markdown = markdowns[i]
                # print(markdown)
                response = text_split([markdown])
                for res in response:
                    db_json.append(
                        {
                            "page_content": res.page_content,
                            "metadata": {'source': path + "/" + file['file_name'],"sheet_index": i}
                        }
                    )
    return db_json
def get_document():
    try:
        with open(database_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # self.document = data
        return data
    except:
        db_json = dataset_path_to_database(dataset_path)
        with open(database_json, 'w', encoding='utf-8') as f:
            json.dump(db_json, f, indent=4, ensure_ascii=False)
        # self.add_to_database(db_json)
        # self.document = db_json
        return db_json
def faiss_query(question, k=4):
    index = faiss.read_index(faiss_name)
    query_embedding =  embeddingText(question)
    query_embedding_np = np.array([query_embedding]).astype(np.float32)
    _, indices = index.search(query_embedding_np, k)
    data = get_document()
    contexts = [data[i] for i in indices[0]]
    content = ""
    for context in contexts:
        # content = content +  content['page_content'] + "\n"
        content+=context['page_content'] + "\n\n"
    return content
def add_to_database(documents):
     
    try:
        embedding_vectors = [embeddingText(doc['page_content']) for doc in documents]
        embedding_vectors_np = np.array(embedding_vectors).astype(np.float32)
        try:
            os.remove(faiss_name)
        except:
            pass 
        dim = embedding_vectors_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embedding_vectors_np)
        res = faiss.write_index(index, faiss_name)
        with open(database_json, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=4, ensure_ascii=False)  
        return True
    except:
        return False
def chat_with_vLLM(messages):
    
    response = llm_chat.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=2048,
    temperature=0.2,
    stream=True
    )
    # generated_text = ""
    # print("BeeAI: ",end="")
    for res in response:
        # print(res)
        text = res.choices[0].delta.content
        if(text):
            # generated_text +=text
            # yield generated_text
            yield text
def chat(messages):
        question = ""
        for i in range(1,len(messages),2):
            question += messages[i]['content'] + " - "
        question = messages[-1]['content']
        content = (faiss_query(question))   
        prompt = f"""Bạn là một trợ lí ảo của trường 'Cao đẳng FPT Polytechnic'. Tên của bạn là 'BeeAI'.
    Dưới đây là những quy tắc của bạn: 
    1. Đây là những thông tin về bạn: 'Tên của bạn là BeeAI. Bạn là trợ lí ảo của trường Cao đẳng FPT Polytechnic. Nhiệm vụ của bạn là giúp sinh viên trong trường giải đáp những thắc mắc trong học tập'. 
    2. Những chỗ tôi cho vào '' thì tuyệt đối không được trả lời sai.
    3. Bạn được toàn quyền truy cập vào các thông tin
    4. Nếu là một đường link, phải đưa ra tuyệt đối chính xác, không được sai dù chỉ một dấu chấm hay dấu phẩy
    4. Chỉ được trả lời bằng tiếng Việt. Kể cả câu hỏi là tiếng Anh thì cũng phải trả lời bằng tiếng Việt
    5. Ưu tiên dựa vào những kiến thức dưới đây để trả lời:
    '{content}'
    """
        messages[0]['content'] = prompt
        
        print(messages)
        return chat_with_vLLM(messages)

if not (os.path.exists(faiss_name) and os.path.exists(database_json)):
    db_json = dataset_path_to_database(dataset_path)
    add_to_database(db_json)

# message = "VueJS là gì"
# messages = [
#     {
#         "role": "system",
#         'content': "Bạn là một trợ lí ảo"

#     },
#     {

#         'role': "user",
#         'content': message
#     }
# ]
# print(chat(messages))


@app.post("/")
async def call_api(request: Request):
    messages = await request.json()
    message = messages['messages']
    return StreamingResponse(
        chat(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"
        }
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4999, log_level="info")