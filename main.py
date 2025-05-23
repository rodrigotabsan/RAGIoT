import streamlit as st
import json
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar variables de entorno
load_dotenv()

def cargar_datos_sensores(ruta_json: str):
    """Carga el JSON y convierte los datos en documentos"""
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        documentos = []
        granja = datos["granja_datos"]
        
        # Procesar cada sensor
        for sensor in granja["sensores"]:
            contenido_sensor = f"""
            Sensor ID: {sensor['id']}
            Tipo: {sensor['tipo']}
            Ubicación: {sensor['ubicacion']}
            Configuración: Umbral mínimo {sensor['configuracion']['umbral_minimo']}, 
            máximo {sensor['configuracion']['umbral_maximo']}
            """
            #Aquí se crea cada documento a partir de la información troceada del json de cada sensor y 
            #se almacena en un array de documentos.
            documentos.append(Document(
                page_content=contenido_sensor.strip(),
                metadata={
                    "sensor_id": sensor['id'],
                    "tipo_sensor": sensor['tipo'],
                    "ubicacion": sensor['ubicacion']
                }
            ))
            
            # Procesar lecturas
            for lectura in sensor['lecturas']:
                contenido_lectura = f"""
                Sensor {sensor['id']} ({sensor['tipo']}) en {sensor['ubicacion']}:
                Valor: {lectura['valor']} {lectura['unidad']}
                Estado: {lectura['estado']}
                Timestamp: {lectura['timestamp']}
                """
                #Aquí se crea cada documento a partir de la información troceada del json de cada lectura del sensor y 
                #se almacena en un array de documentos.
                documentos.append(Document(
                    page_content=contenido_lectura.strip(),
                    metadata={
                        "sensor_id": sensor['id'],
                        "valor": lectura['valor'],
                        "estado": lectura['estado']
                    }
                ))
        
        return documentos
    
    except FileNotFoundError:
        st.error("No se encontró el archivo sensores_iot.json en la carpeta data/")
        return []
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return []

def crear_vector_store(documentos):
    """Crea el vector store con ChromaDB"""
    if not documentos:
        return None
    
    try:
        embeddings = OpenAIEmbeddings()
        #Chroma procesa cada documento para convertirlo cada uno en un vector tras haber utilizado OpenAIEmbeddings que traducirá cada palabra
        #del documento a un formato numérico. El resultado de cada vector lo almacenará en un directorio llamado chroma_db y solo lo hará una vez.
        vector_store = Chroma.from_documents(
            documents=documentos,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creando vector store: {str(e)}")
        return None

#Aquí se crea la interfaz web con streamlit
def main():
    st.title("🌱 RAG IoT Agricultura")
    st.write("Consulta inteligente sobre datos de sensores agrícolas")
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("No se encontró OPENAI_API_KEY en las variables de entorno")
        st.write("Crea un archivo .env con tu API key:")
        st.code("OPENAI_API_KEY")
        return
    
    # Cargar datos
    if 'vector_store' not in st.session_state:
        with st.spinner("Cargando datos de sensores..."):
            datos = cargar_datos_sensores("data/sensores_iot.json")
            if datos:
                st.session_state.vector_store = crear_vector_store(datos)
                
                if st.session_state.vector_store:
                    # Crear la cadena RAG con la temperatura a 0,7 que es el estándar.
                    llm = ChatOpenAI(temperature=0.7)
                    st.session_state.rag_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    st.success("✅ Sistema RAG inicializado correctamente")
                else:
                    st.error("❌ Error inicializando el vector store")
                    return
            else:
                st.error("❌ No se pudieron cargar los datos")
                return
    
    # Interfaz de consulta
    st.write("### Haz una pregunta sobre los sensores:")
    
    # Ejemplos de preguntas
    ejemplos = [
        "¿Qué sensores tienen alertas activas?",
        "¿Cuál es el estado de los sensores de humedad?",
        "¿Qué sensores están en el Sector A?",
        "¿Hay algún sensor con valores fuera del rango normal?"
    ]
    
    ejemplo_seleccionado = st.selectbox("O selecciona un ejemplo:", [""] + ejemplos)
    
    pregunta = st.text_input(
        "Tu pregunta:",
        value=ejemplo_seleccionado,
        placeholder="Escribe tu pregunta aquí..."
    )
    
    if pregunta and 'rag_chain' in st.session_state:
        with st.spinner("Procesando consulta..."):
            try:
                respuesta = st.session_state.rag_chain.invoke({"query": pregunta})
                
                st.write("### 🤖 Respuesta:")
                st.write(respuesta["result"])
                
                # Mostrar fuentes
                if "source_documents" in respuesta and respuesta["source_documents"]:
                    st.write("### 📄 Fuentes consultadas:")
                    for i, doc in enumerate(respuesta["source_documents"]):
                        with st.expander(f"Fuente {i+1}"):
                            st.write(doc.page_content)
                            if doc.metadata:
                                st.write("**Metadatos:**", doc.metadata)
            
            except Exception as e:
                st.error(f"Error procesando la consulta: {str(e)}")

if __name__ == "__main__":
    main()