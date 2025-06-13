# Quranic-QA-Chatbot
RAG based Quranic QA Chatbot


# RAG-based Tafseer System for Surah Al-Baqarah

A Retrieval-Augmented Generation (RAG) system that provides AI-powered Tafseer (interpretation) of Surah Al-Baqarah, the second chapter of the Holy Quran. This system combines the power of large language models with authentic Islamic scholarship to deliver accurate, contextually grounded answers about Quranic verses.

## 🎯 Overview

This project addresses the limitations of traditional LLMs when dealing with Islamic scholarship by implementing a RAG framework that:
- Retrieves relevant context from trusted Tafseer sources
- Generates accurate, grounded responses about Surah Al-Baqarah
- Ensures factual consistency and reduces hallucinations
- Provides contextual references to classical Islamic scholarship

## 👥 Team Members

- **Huzaifa Bin Tariq** - 25133
- **Hazim Ghulam Farooq** - 25148  
- **Muhammad Wasay** - 24497

## 🏗️ System Architecture

```
User Query → Query Processing → Semantic Search → Context Retrieval → 
Answer Generation → Response with Citations
```

### Core Components
- **Document Processor**: Handles English Tafseer text preprocessing and chunking
- **Vector Store**: Chroma/FAISS for semantic similarity search
- **Embedding Model**: BAAI/bge-base-en for document representation
- **LLM**: Qwen-QWQ-32B via Groq API for answer generation
- **Evaluation Framework**: Custom metrics for faithfulness and relevance

## 📊 Performance Results

Our best configuration achieved:

| Metric | Score | Description |
|--------|-------|-------------|
| **Latency** | 1.99s | Average response time |
| **Semantic Similarity** | 0.90 | Answer quality vs ground truth |
| **Relevance** | 0.31 | Context utilization score |
| **Faithfulness** | 0.65 | Factual consistency score |

### Best Configuration
- **Chunk Size**: 500 tokens (100 overlap)
- **Vector Store**: Chroma
- **Embedding**: BAAI/bge-base-en
- **LLM**: qwen-qwq-32b (Groq API)
- **Search**: Keyword-based with dense ranking
- **Processing**: Summarization before LLM

## 🚀 Quick Start

### Prerequisites
```bash
# Required dependencies
pip install chromadb
pip install sentence-transformers
pip install groq
pip install ragas
```

### Installation
```bash
git clone https://github.com/your-username/rag-tafseer-baqarah.git
cd rag-tafseer-baqarah
pip install -r requirements.txt
```

### Setup
1. **Get Groq API Key**: Sign up at [Groq Console](https://console.groq.com/)
2. **Set Environment Variable**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```
3. **Prepare Data**: The cleaned Tafseer PDF is included in `/data/`

### Usage
```python
from rag_system import TafseerRAG

# Initialize the system
rag = TafseerRAG(
    chunk_size=500,
    chunk_overlap=100,
    vector_store="chroma",
    embedding_model="BAAI/bge-base-en",
    llm_model="qwen-qwq-32b"
)

# Load the Tafseer data
rag.load_documents("data/al-baqarah-eng.pdf")

# Ask a question
question = "What is the significance of the 'forbidden tree' in the story of Adam (A.S)?"
response = rag.query(question)
print(response)
```

## 📖 Data Source

**Primary Source**: English Tafseer of Surah Al-Baqarah (406 pages)
- **Content**: English translation of Quranic verses with detailed commentary
- **Preprocessing**: 
  - Removed Arabic symbols and random characters
  - Standardized character encoding (ĥ→h, ū→u, ā→a)
  - Manual cleaning and verification
- **Format**: PDF converted to structured text for RAG processing

## 🔬 Evaluation Methodology

### Ground Truth Generation
- **6 carefully curated questions** across easy, medium, and hard difficulty levels
- **LLaMA 3 70B model** used for ground truth generation with strict constraints
- **Manual verification** to ensure alignment with Surah Al-Baqarah content

### Test Questions
| Difficulty | Question Examples |
|------------|-------------------|
| **Easy** | Who is Adam (A.S)? |
| **Medium** | Why did the Jews of Madinah oppose Islam? |
| **Hard** | What lessons does the story of Adam (A.S) teach about free will? |

### Metrics
1. **Semantic Similarity**: Cosine similarity between generated and ground truth answers
2. **Relevance Score**: ROUGE-based overlap between answer and retrieved context
3. **Faithfulness Score**: Cross-encoder model measuring factual consistency

## ⚙️ Configuration Options

### Chunking Strategies
- **500-100**: Best for speed and faithfulness
- **1000-200**: Balanced performance
- **1500-300**: Maximum context, higher latency

### Vector Stores
- **Chroma**: Faster, simpler setup
- **FAISS**: Better for large-scale deployments

### LLM Options
- **llama3-8b-8192**: Fastest responses
- **qwen-qwq-32b**: Best overall performance (recommended)
- **meta-llama/llama-4-maverick-17b**: Highest quality, slower

## 📁 Project Structure

```
rag-tafseer-baqarah/
├── data/
│   ├── al-baqarah-eng.pdf          # Source Tafseer document
│   └── ground_truth.json           # Evaluation questions
├── src/
│   ├── rag_system.py               # Main RAG implementation
│   ├── evaluation.py               # Metrics and evaluation
│   └── preprocessing.py            # Data cleaning utilities
├── results/
│   ├── rag_config_results_1.json   # Experiment results
│   ├── rag_config_results_2.json
│   └── rag_config_results_3.json
├── notebooks/
│   └── evaluation_analysis.ipynb   # Results analysis
├── requirements.txt
└── README.md
```

## 🔍 Key Findings

### Performance Insights
- **Smaller chunks (500 tokens)** provide better faithfulness scores
- **BGE embeddings** consistently outperform alternatives
- **Keyword search with dense ranking** offers optimal precision
- **Summarization preprocessing** significantly improves answer quality

### Limitations
- Currently limited to Surah Al-Baqarah only
- English-only Tafseer sources
- Performance varies with question complexity
- Requires careful prompt engineering for cultural sensitivity

## 🔮 Future Work

1. **Expand Coverage**: Include complete Quran Tafseer
2. **Multilingual Support**: Add Arabic and other languages
3. **Enhanced Evaluation**: Larger ground truth dataset
4. **Islamic Scholar Review**: Validation by religious authorities
5. **Web Interface**: User-friendly chat interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions from the community! Please:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Cultural Sensitivity Guidelines
When contributing to this Islamic studies project, please:
- Maintain respectful language and tone
- Verify accuracy against authentic Islamic sources
- Acknowledge the sacred nature of the Quranic content
- Consult with Islamic scholars when in doubt

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **Huzaifa Bin Tariq**: [huzaifa.tariq@example.com]
- **Hazim Ghulam Farooq**: [hazim.farooq@example.com]
- **Muhammad Wasay**: [muhammad.wasay@example.com]

## 🙏 Acknowledgments

- Thanks to the Islamic scholars whose Tafseer work made this project possible
- Groq for providing high-speed LLM inference
- The open-source community for the foundational tools and libraries
- Our instructors for guidance throughout this research project

---

*This project is developed for educational and research purposes. All Quranic interpretations are based on established scholarly sources and should be verified with qualified Islamic authorities for religious guidance.*
