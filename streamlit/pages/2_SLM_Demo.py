import streamlit as st
import time
import datetime
import torch
import os
import json
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import login

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()

huggingface_api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config("Küçük Dil Modeli Demo", page_icon="https://img.icons8.com/?size=100&id=pnUfrhjBRScY&format=png&color=000000")

# Veriler ve Modellerin Tanımlanması
data_path = "./Data/iot_readable_last_en.txt"
encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
encoder_model_lib = "transformer"
decoder_name = "microsoft/Phi-3-mini-4k-instruct"
vectorstore_path = "./Data/iot_data_faiss_en_new"
local_decoder_path = "./Models/model-phi-3-4k-mini-instruct"
local_tokenizer_path = "./Models/tokenizer-phi-3-4k-mini-instruct"

# CHAT HİSTORY MANTIĞINI KURMAYI DÜŞÜNEBİLİRSİN DAHA MANTIKLI BİR YAKLAŞIM OLABİLİR.

# Fonksiyonlar

# Vektör Veritabanının Hazırlanmasını Sağlayan Fonksiyon:
@st.cache_resource
def initialize_vectorstore():
    if not os.path.exists(vectorstore_path):
        # LOAD -> SPLIT -> EMBED -> STORE -> RETRIEVE
        loader = TextLoader(data_path, encoding="utf-8")
        text_file = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size = 1,
            chunk_overlap = 0
        )

        # Metadata Çıkarma İşlemi
        print("Metadata Çıkarılıyor...")
        metadata = []
        for line in text_file[0].page_content.split("\n"):
            parts = [p.strip() for p in line.split(",")]

            if len(parts) <= 1:
                break
            
            date = parts[0].split(":")[1].strip()
            hour = parts[1].split(":", 1)[1].strip()
            room = parts[2].split(":")[1].strip()
            sensor = parts[3].split(":")[1].strip()

            dtime = datetime.datetime.strptime(f"{date} {hour}", "%Y-%m-%d %H:%M")
            timestamp = dtime.timestamp()

            metadata.append({"Date": date, "Hour": hour, "Room": room, "Sensor": sensor})
        print("Metadata Hazır!\n")

        print("Metinler Ayrıştırılıyor...")
        splitted_texts = text_splitter.split_text(text_file[0].page_content)
        print(f"Metinden şu kadar satır çıkarıldı: {len(splitted_texts)}")
        splitted_texts = splitted_texts[:-1]
        print(f"Veri olarak seçilen satır sayısı: {len(splitted_texts)}")
        print(f"Metadata listesi uzunluğu: {len(metadata)}\n")

        docs = [Document(page_content=text, metadata=meta) for text, meta in zip(splitted_texts, metadata)]

        print(f"Embedding Modeli Yükleniyor... Model Kütüphanesi: {encoder_model_lib}")
        if encoder_model_lib == "sentence-transformer":
            pass
        elif encoder_model_lib == "transformer":
            embeddings = HuggingFaceEmbeddings(
                model_name=encoder_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": False}
            )
        elif encoder_model_lib == "google-genai":
            pass
        print("Embedding Modeli Hazır!\n")

        print("Metadatalı Vektör Veritabanı Başlatılıyor...")
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        print("Metadatalı Vektör Veritabanı Hazır!")
    
    else:
        print(f"Metadatalı Vektör Veritabanı {vectorstore_path} Klasöründen Yükleniyor...")
        embeddings = HuggingFaceEmbeddings(
            model_name=encoder_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
        
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        print("Metadatalı Vektör Veritabanı Başarıyla Yüklendi!")

    return vectorstore

# Kullanıcı mesajını alıp mesajın veritabanı sorgusuna ihtiyaç olmadan cevaplanıp cevaplanamayacağına karar veren fonksiyon:
def route_model(user_query : str):
    route_msg_template = f"""
    You are an expert at routing a user's question to the correct data source.

    Based on the user's question, decide whether to route them to the 'vectorstore' or 'general_knowledge'.
    Make sure you only return decision that you made.

    'vectorstore': This is for questions that ask for specific, historical, or current data points from the smart home sensors. 
    Examples:
    - What is the current temperature in the living room?
    - What was the humidity in the bedroom yesterday at 5 PM?
    - Show me the sensor readings for the last hour.

    'general_knowledge': This is for questions that ask for advice, explanations, how-to guides, or general conversation that does not require specific sensor data.
    Examples:
    - How can I reduce the temperature in a room?
    - Why is humidity important?
    - Hello, how are you today?

    User Question:
    {user_query}
    """

    messages = [
        {
            "role" : "user",
            "content" : route_msg_template
        }
    ]

    return messages

# Vektör Veritabanından Dökümanları Bulmak İçin Kullanılan Fonksiyon:
def fetch_documents(structured_query, vectorstore, metadatas):
    if metadatas['Hour'] == "all":
        retriever = vectorstore.as_retriever(search_type = "mmr", search_kwargs={"k":20, "fetch_k": 200})
        search_result = retriever.invoke(structured_query)
    else:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 300, "filter": {"Hour": metadatas["Hour"]}})
        search_result = retriever.invoke(structured_query)
    
    return search_result

# Cevap Verici Modelin Tokenizerini Hazırlayan Fonksiyon:
@st.cache_resource
def initialize_decoder_tokenizer():
    if os.path.exists(local_tokenizer_path):
        print(f"Decoder Tokenizeri {local_tokenizer_path} Klasöründen Yükleniyor...")
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
        print(f"Decoder Tokenizeri {local_tokenizer_path} Klasöründen Başarıyla Yüklendi!\n")
    else:
        print("Local Tokenizer Bulunamadı Online Olarak İndiriliyor...")
        login(token=huggingface_api_key)
        tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        print("Tokenizer Başarıyla İndirildi!\n")
    return tokenizer

# Cevap Verici Modeli Hazırlayan Fonksiyon:
@st.cache_resource
def initialize_decoder():
    if torch.cuda.is_available():
        print(f"Sistemdeki aktif GPU sayısı: {torch.cuda.device_count()}")
        print(f"Sistemdeki Ana GPU'nun Adı: {torch.cuda.get_device_name(0)}")

        print("BNB Config Hazırlanıyor...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16
        )
        print("BNB Config Hazır!\n")
    else:
        print("Sistemde çalışan GPU Bulunamıyor.")
        print("Model CPU üzerinde koşturulacak.\n")
        quantization_config = None

    if os.path.exists(local_decoder_path):
        if quantization_config:
            print(f"Decoder Modeli {local_decoder_path} Klasöründen Quantize Edilerek Yükleniyor...")
            print(f"Modelin Yükleneceği GPU: {torch.cuda.get_device_name(0)}")  
            model = AutoModelForCausalLM.from_pretrained(local_decoder_path, quantization_config=quantization_config, device_map="auto")
            print(f"Decoder Modeli {local_decoder_path} Klasöründen Başarıyla Yüklendi!")
        else:
            print(f"Decoder Modeli {local_decoder_path} Klasöründen Tam Olarak Yükleniyor...")
            print("Decoder Modeli CPU Üzerinden Koşturulacak.")
            model = AutoModelForCausalLM.from_pretrained(local_decoder_path, device_map="auto")
            print(f"Decoder Modeli {local_decoder_path} Klasöründen Başarıyla Yüklendi!")
    else:
        print("Local Model Bulunamadı Online Olarak İndiriliyor...")
        login(token=huggingface_api_key)
        
        if quantization_config:
            print(f"Decoder modeli {torch.cuda.get_device_name(0)} isimli GPU üzerine yükleniyor...")
            model = AutoModelForCausalLM.from_pretrained(decoder_name, quantization_config=quantization_config, device_map="auto")
        else:
            print(f"Decoder modeli CPU üzerine yükleniyor...")
            model = AutoModelForCausalLM.from_pretrained(decoder_name, device_map="auto")
        print("Decoder Modeli Başarıyla Yüklendi!")
    
    return model

# Kullanıcının Sorusunu İşleyebilmek İçin Prompt Hazırlayan Fonksiyon:
def create_query_prompt(user_query : str):
    current_year = datetime.datetime.now().year
    
    messages = [
        {
            "role": "user",
            "content": f"""
    Your task is to extract specific fields from the given input sentence and return the result strictly in JSON format.
    
    Rules:
    - Always include these 4 fields in the JSON: "date", "time", "room", "query".
    - If the sentence has a specific calendar date (e.g., "16 August 2025"), extract it in ISO format YYYY-MM-DD if possible.
      If the year is not given, return only the given part (e.g., "16 August").
    - If no explicit date is mentioned, return "not mentioned".
    - If year is not mentioned use current year, Current Year: {current_year}
    - For the "time" field:
        * If the query has explicit time (e.g., "9 A.M."), convert it to 24-hour format (e.g., "09:00").
        * If the query uses relative expressions like "yesterday", "today", "current", "last time", keep them as they are.
        * If no time expression exists, return "not mentioned".
        * If no room information exists, return "not mentioned".
        * If no query information exists, return "not mentioned"
    - "room" should always be extracted from the input (e.g., "living room", "bedroom" or "not mentioned").
    - "query" should capture what is being asked (e.g., "temperature", "humidity" or "not mentioned").
    """
        },
        {
            "role": "assistant",
            "content": "Understood. I will extract the fields from the input sentence and provide the output in JSON format. Please provide the first input."
        },
        {
            "role": "user",
            "content": "Input: What is the current temperature in the living room?"
        },
        {
            "role": "assistant",
            "content": '{"date": "current", "time": "not mentioned", "room": "living room", "query": "temperature"}'
        },
        {
            "role": "user",
            "content": "Input: What was the temperature in the bedroom yesterday?"
        },
        {
            "role": "assistant",
            "content": '{"date": "yesterday", "time": "not mentioned", "room": "bedroom", "query": "temperature"}'
        },
        {
            "role": "user",
            "content": "Input: What was the temperature in the bedroom last time?"
        },
        {
            "role": "assistant",
            "content": '{"date": "last time", "time": "not mentioned", "room": "bedroom", "query": "temperature"}'
        },
        {
            "role": "user",
            "content": "Input: What is the current humidity level in the sitting room?"
        },
        {
            "role": "assistant",
            "content": '{"date": "current", "time": "not mentioned", "room": "sitting room", "query": "humidity"}'
        },
        {
            "role": "user",
            "content": "Input: What was the humidity level in living room yesterday at 5 P.M"
        },
        {
            "role": "assistant",
            "content": '{"date": "yesterday", "time": "17:00", "room": "living room", "query": "humidity"}'
        },
        {
            "role": "user",
            "content": "Input: What was the temperature in bathroom yesterday"
        },
        {
            "role": "asisstant",
            "content": '{"date": "yesterday", "time": "not mentioned", "room": "bathroom", "query": "temperature"}'
        },
        {
            "role": "user",
            "content": "Input: What was the temperature in the living room at the 16 August 2025 9 A.M?"
        },
        {
            "role": "assistant",
            "content": '{"date": "2025-08-16", "time": "09:00", "room": "living room", "query": "temperature"}'
        },
        {
            "role": "user",
            "content": "Input: What was the humidity level in bedroom at 20 August"
        },
        {
            "role": "assistant",
            "content": '{"date": "2025-08-16, "time": "not mentioned", "room": "bedroom", "query": "humidity"}'
        },
        {
            "role":"user",
            "content": "Input: Which day was the hottest for the living room?"
        },
        {
            "role":"assistant",
            "content":'{"date": "last week", "time": "all", "room": "living room", "query": "temperature"}'
        },
        {
            "role":"user",
            "content":"Input: Which day was the hottest day?"
        },
        {
            "role":"assistant",
            "content":'{"date": "last week", "time": "all", "room": "not mentioned", "query": "temperature"}'
        },
        {
            "role":"user",
            "content":"Input: today temperature"
        },
        {
            "role":"assistant",
            "content":'{"date": "current", "time": "not mentioned", "room": "not mentioned", "query": "temperature"}'
        },
        {
            "role":"user",
            "content":"Input: on 15 August 5 PM temperature"
        },
        {
            "role":"assistant",
            "content":'{"date": "2025-08-15", "time": "17:00", "room": "not mentioned", "query": "temperature"}'
        },
        {
            "role": "user",
            "content": "Input: living room temperature on 14 august"
        },
        {
            "role": "assistant",
            "content": '{"date": "2025-08-14", "time": "not mentioned", "room": "living room", "query": "temperature"}'
        },
        {
            "role": "user",
            "content": f"Input: {user_query}"
        },
    ]

    return messages

# Kullanıcının Sorusunu Yeniden Düzenleyen ve Sorudaki Metadataları Çıkartan Fonksiyon: (key value alma kısmını değiştirdim.)
def rewrite_query_and_extract_metadata(model_output : str, user_query : str):
    """
    Gelecek stringde 4 alan var, bu alanlar şunlar:
    - date
    - time
    - room
    - query

    date: ['current', 'yesterday', 'now', 'last time' or date format e.g '2025-08-16']
    time: ['not mentioned' or 24 hour format e.g '09:00']
    room: ['kitchen', 'bedroom', 'living room', 'bathroom' ...]
    query: ['temperature', 'humidity']
    """
    match = re.search(r'\{.*\}', model_output, re.DOTALL)
    if not match:
        print(f"Girdide geçerli bir JSON yapısı bulunamadı: {model_output}")
        # Hatalı durumda, açıklama isteyen bir statü ile çıkalım
        return user_query, [], "clarification_needed" 
    
    clean_output = match.group(0)
    
    try:
        temp = json.loads(clean_output)
    except:
        print(f"Şu çıktıyı JSON objesine çevirmeye çalışırken hata oluştu:\n{model_output}")

    try:
        date = temp['date']
        time = temp['time']
        room = temp['room']
        query = temp['query']
    except:
        print(f"JSON objesinden valueleri almaya çalışırken hata meydana geldi {temp}")

    current = datetime.datetime.now()

    current_date = current.strftime("%Y-%m-%d")
    
    current_hour = current.strftime("%H:%M")
    current_hour = datetime.datetime.strptime(current_hour, "%H:%M")
    new_min = (current_hour.minute // 5) * 5
    rounded_hour = current_hour.replace(minute=new_min, second=0, microsecond=0)

    rounded_hour = rounded_hour.strftime("%H:%M")
    
    yesterday = current - datetime.timedelta(days=1)
    yesterday_date = yesterday.strftime("%Y-%m-%d")

    if((room.lower() == "not mentioned") or (query.lower() == "not mentioned")):
        content = f"{user_query}"
        metadata = []
        status = "clarification_needed"
        return content, metadata, status

    if ((date.lower() == "not mentioned") or (date.lower() == "current") or (date.lower() == "now") or (date.lower() == "last time")):
        new_date = current_date
    elif (date.lower() == "yesterday"):
        new_date = yesterday_date
    elif (date.lower() == "last week"):
        end_time = current_date
        start_time = (datetime.datetime.strptime(current_date, "%Y-%m-%d") - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        new_date = f"{start_time} {end_time}"
    else:
        new_date = date

    if (time.lower() == "not mentioned"):
        new_time = rounded_hour
    elif (time.lower() == "all"):
        new_time = "all"
    else:
        new_time = time

    if ((time.lower() == "yesterday") or (not(time.lower() == "not mentioned"))):
        content = f"What was the {query} in {room} on {date} at {new_time}"
    else:
        content = f"What is the {query} in {room} on {date} at {new_time}"

    metadata = {"Date": new_date, "Hour": new_time, "Room": room.capitalize(), "Sensor":  query.capitalize()+" Sensor" }
    status = "no_clarification_needed"
    return content, metadata, status

# Modelin Kullanıcı Sorusuna Cevap Vermesini Sağlamak İçin Prompt Üreten Fonksiyon:
def create_answering_prompt(user_query : str, model_decision : str, status = None, docs_from_faiss=None, metadata_info=None):
    if "general_knowledge" in model_decision:
        answering_prompt_template = f"""
        You are an smart home assistant that answers user questions clearly and directly using general knowledge. 
        Keep your responses concise and focused without unnecessary small talk such as "Would you like to know more?" or "Should I also explain this?". 
        Avoid giving identical answers every time by slightly varying your wording or adding small creative touches, while still keeping the response accurate and relevant to the user’s question. 
        You can provide information on the following topics: smart home sensors, general knowledge, or the current measurement values from sensors in specific parts of the house.
        You are an assistant that only answers user questions. You only have access to the database. **You cannot control or change smart devices, like turning them on or off.**
        If you do not know the answer to a question, state that you do not have the information rather than guessing.
        **Make sure you just return the answer that you made.**

        Now answer the user's question with your general knowledge:
        User Question: {user_query}
        """

        messages = [
            {
                "role" : "user",
                "content" : answering_prompt_template 
            }
        ]
    else:
        if ((status) and (status == "clarification_needed")):
            answering_prompt_template = """
            Your goal is to act as an intelligent assistant. 
            You must analyze the user's request provided in the [User Question] section. 
            Your task is to determine if the user has supplied all the necessary information to answer their question using the data available in the [Provided Information] section.
            Based on your analysis, you will either provide a direct answer or ask for the missing information.

            ### Rules
            1. Your system language is English, you must answer in English.
            2. **Make sure return only answer that you made.**


            ### Required Information Fields:
                1. date
                2. time
                3. room (e.g., living room, kitchen, bedroom)
                4. query (the specific metric the user is asking for, e.g., temperature, humidity)
                
            ### Examples

            ## Example 1:
            [User Question]
            what about the living room?
            The user did not provide the required informations for the query.

            [Provided Information]

            [Metadata]
                
            [Your Answer]
            I can help you with information about the living room, but I need a few more details to give you a specific answer. 
            Please provide the following missing information: date, time, and query (e.g., temperature or humidity).
                
            ## Example 2:
            [User Question]
            temperature today at 3 PM

            [Provided Information]

            [Metadata]

            [Your Answer]
            I can provide the temperature for today at 3 PM, but I need to know which room you are asking about. Please specify the room.

            ## Now, answer the user's actual question:
            [User Question]
            {user_question_placeholder}

            [Provided Information]
            {context_from_faiss_placeholder}

            [Metadata]
            {metadata_info_placeholder}

            [Your Answer]
            """

            answering_prompt_final = answering_prompt_template.format(
                user_question_placeholder = user_query,
                context_from_faiss_placeholder = docs_from_faiss,
                metadata_info_placeholder = metadata_info
            )

            messages = [
                {
                    "role":"user",
                    "content":answering_prompt_final
                }
            ]
        else:
            answering_prompt_template = """
            You are a helpful and concise smart home assistant. Your primary goal is to answer the user's question based strictly on the context provided under "[Provided Information]" and "[Metadata]". Follow all rules precisely.

            Here are the rules and examples for how you must respond.

            ### Rules
            1. Your main goal is to answer the user's question using only the informations within the '[Provided Information]' and '[Metadata]' blocks.
            2. If the information is sufficient to answer the question, synthesize it into a complete, natural, and helpful English sentence. Do not just state the number.
            3. If the '[Provided Information]' block is empty, does not contain relevant details, or is insufficient to answer the question, you must respond with the exact phrase: "I don't have enough information on this topic, please try again later."
            4. If the [User Question] asks for a maximum value (using words like 'hottest', 'maximum', or 'most'), locate that specific maximum value in the [Provided Information]. Verify that this value answers the user's question and use it to write your response.
            5. After providing the direct answer based on the information, you are allowed to add one short, relevant, and interesting fact in English if you are highly confident about it. This extra fact must start with the prefix "Info: ". Do not add this if you have no information.
            6. **Make sure you only return answer that you made.**

            ### Examples

            ## Example 1: (Sufficient Information)
            [User Question]
            What was the temperature in the sitting room yesterday?

            [Provided Information]
            Date: 2025-08-15, Time: 9:00, Room: Sitting Room, Sensor: Temperature Sensor, Sensor Value: 23.4, Sensor Unit: °C, Status: Low Temperature

            [Metadata]
            {{"Date": "2025-08-15", "Hour": "09:00", "Room": "Sitting room", "Sensor": "Temperature Sensor"}}

            [Your Answer]
            The temperature recorded in the sitting room yesterday at 9:00 was 23.4°C, which was considered low.

            ## Example 2: (Insufficient Information)
            [User Question]
            What is the orbit of Mars?

            [Provided Information]
            "No information found this topic."

            [Metadata]
                
            [Your Answer]
            I don't have enough information on this topic, please try again later.

            ## Example 3: (Sufficient Information + Extra Fact)
            [User Question]
            What is the current temperature in the living room?

            [Provided Information]
            Date: 2025-08-16, Time: 15:00, Room: Living Room, Sensor: Temperature Sensor, Sensor Value: 25.4, Sensor Unit: °C, Status: Normal

            [Metadata]
            {{"Date": 2025-08-16, "Hour": 15:00, "Room": Living room, "Sensor": "Temperature Sensor"}}

            [Your Answer]
            The current temperature in the living room at 15:00 is 25.4°C, which is a normal value. Info: The ideal room temperature for humans is generally considered to be between 20-22°C.
            
            ## Example 4:  
            [User Question]
            Which day was the hottest day in kitchen?

            [Provided Information]
            Date: 2025-08-16, Time: 09:00, Room: Kitchen, Sensor: Temperature Sensor, Sensor Value: 25.4, Sensor Unit: °C, Status: Normal
            Date: 2025-08-15, Time: 15:00, Room: Kitchen, Sensor: Temperature Sensor, Sensor Value: 27.4, Sensor Unit: °C, Status: High Temperature
            Date: 2025-08-14, Time: 10:00, Room: Kitchen, Sensor: Temperature Sensor, Sensor Value: 26.4, Sensor Unit: °C, Status: High Temperature
            Date: 2025-08-15, Time: 13:00, Room: Kitchen, Sensor: Temperature Sensor, Sensor Value: 21.4, Sensor Unit: °C, Status: Normal
            Date: 2025-08-14, Time: 11:00, Room: Kitchen, Sensor: Temperature Sensor, Sensor Value: 22.4, Sensor Unit: °C, Status: Normal

            [Metadata]
            {{"Date": "2025-08-16 2025-08-09", "Hour": "all", "Room": "Kitchen", "Sensor": "Temperature Sensor"}}

            [Your Answer]
            The hottest temperature recorded in the kitchen was 27.0°C on 2025-08-15 at 15:00, which was considered high temperature. Info: The temperature in a kitchen can often be higher than other rooms due to heat generated from cooking appliances.

            ## Now, answer the user's actual question based on the provided information and metadata information:
            ## **Make sure you only return answer that you made.**

            [User Question]
            {user_question_placeholder}

            [Provided Information]
            {context_from_faiss_placeholder}

            [Metadata]
            {metadata_info_placeholder}

            [Your Answer]
            """

            if docs_from_faiss:
                context = "\n".join([doc.page_content for doc in docs_from_faiss])
            else:
                context = "No document infortmation found this topic."
            
            if not metadata_info:
                metadata_info = "No metadata information found for this topic."

            answering_prompt_final = answering_prompt_template.format(
                user_question_placeholder = user_query,
                context_from_faiss_placeholder = context,
                metadata_info_placeholder = metadata_info
            )

            messages = [
                {"role": "user", "content": answering_prompt_final}
            ]

    return messages

# Modelin Ürettiği Cevabı Veren Fonksiyon (hem metadata için hem de normal cevap için):
def model_invoke(messages : list, tokenizer, model):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    answer = model.generate(inputs, max_new_tokens=150, temperature=0.3, eos_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(answer[0][inputs.shape[-1]:], skip_special_tokens=True)

    return result

# Modelin Ürettiği Cevabı Animasyonlu Yazdırmak İçin Gerekli Fonksiyon:
def response_generator(model_response : str):
    for chars in model_response:
        yield chars + ""
        time.sleep(0.03)


if "local_slm_initialized" not in st.session_state:
    st.session_state.local_slm_initialized = True
    # Vektör veritabanını kur
    info = st.text("Vektör Veritabanı Başlatılıyor...")
    st.session_state.slm_vectorstore = initialize_vectorstore()
    info.text("Vektör Veritabanı Hazır!")
    time.sleep(1)

    # Decoder modelinin hazırlanması
    info.text("Decoder Hazırlanıyor...")
    st.session_state.decoder_model = initialize_decoder()
    info.text("Decoder Hazır!")
    time.sleep(1)

    # Tokenizerin hazırlanması
    info.text("Tokenizer Hazırlanıyor...")
    time.sleep(0.5)
    st.session_state.decoder_tokenizer = initialize_decoder_tokenizer()
    info.text("Tokenizer Hazır!")
    time.sleep(0.5)
    info.empty()

if "local_slm_messages" not in st.session_state:
    st.session_state.local_slm_messages = []

for message in st.session_state.local_slm_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
if prompt := st.chat_input("Bir Şeyler Sorun"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.local_slm_messages.append({"role" : "user", "content" : prompt})

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor...", show_time=True):
            route_prompt = route_model(user_query=prompt)

            model_decision = model_invoke(messages=route_prompt, tokenizer=st.session_state.decoder_tokenizer, model=st.session_state.decoder_model)

            if "general_knowledge" in model_decision:
                answering_prompt_template = create_answering_prompt(user_query=prompt, model_decision=model_decision)

                model_response = model_invoke(messages=answering_prompt_template, tokenizer=st.session_state.decoder_tokenizer, model=st.session_state.decoder_model)
            else:
                query_prompt_template = create_query_prompt(prompt)

                model_output = model_invoke(query_prompt_template, tokenizer=st.session_state.decoder_tokenizer, model=st.session_state.decoder_model)

                structured_query, metadatas, status = rewrite_query_and_extract_metadata(model_output, user_query=prompt)

                if not (status == "clarification_needed"):
                    docs_from_faiss = fetch_documents(structured_query=structured_query, vectorstore=st.session_state.slm_vectorstore, metadatas=metadatas)
                else:
                    docs_from_faiss = []
                
                answering_prompt_template = create_answering_prompt(prompt, model_decision=model_decision, status=status, docs_from_faiss=docs_from_faiss, metadata_info=metadatas)

                model_response = model_invoke(answering_prompt_template, tokenizer=st.session_state.decoder_tokenizer, model=st.session_state.decoder_model)

        response = st.write_stream(response_generator(model_response=model_response))
    st.session_state.local_slm_messages.append({"role" : "assistant", "content" : response})