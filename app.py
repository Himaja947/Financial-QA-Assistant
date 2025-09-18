import streamlit as st
import pandas as pd
import PyPDF2
import re
import json
from io import BytesIO
import requests
import openpyxl
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Document Q&A Assistant",
    page_icon="📊",
    layout="wide"
)

class DocumentProcessor:
    """Handle document processing for PDF and Excel files"""
    
    def __init__(self):
        self.extracted_data = {}
        self.raw_text = ""
    
    def extract_pdf_text(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_excel_data(self, excel_file) -> Dict:
        """Extract data from Excel file"""
        try:
            # Read all sheets
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            processed_data = {}
            
            for sheet_name, df in excel_data.items():
                # Convert DataFrame to text for processing
                text_data = df.to_string()
                processed_data[sheet_name] = {
                    'dataframe': df,
                    'text': text_data
                }
            
            return processed_data
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return {}
    
    def extract_financial_metrics(self, text: str) -> Dict:
        """Extract common financial metrics from text"""
        metrics = {}
        
        # Common financial terms and patterns
        patterns = {
            'revenue': r'(?:revenue|sales|income)[\s:$]*([0-9,]+(?:\.[0-9]+)?)',
            'expenses': r'(?:expenses|costs)[\s:$]*([0-9,]+(?:\.[0-9]+)?)',
            'profit': r'(?:profit|net income|earnings)[\s:$]*([0-9,]+(?:\.[0-9]+)?)',
            'assets': r'(?:total assets|assets)[\s:$]*([0-9,]+(?:\.[0-9]+)?)',
            'liabilities': r'(?:total liabilities|liabilities)[\s:$]*([0-9,]+(?:\.[0-9]+)?)',
            'equity': r'(?:equity|shareholders equity)[\s:$]*([0-9,]+(?:\.[0-9]+)?)'
        }
        
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                # Take the first match and clean it
                value = matches[0].replace(',', '')
                try:
                    metrics[metric] = float(value)
                except ValueError:
                    metrics[metric] = value
        
        return metrics
    
    def process_document(self, file, file_type: str):
        """Process uploaded document"""
        if file_type == "pdf":
            self.raw_text = self.extract_pdf_text(file)
        elif file_type == "excel":
            excel_data = self.extract_excel_data(file)
            # Combine all sheet texts
            all_text = ""
            for sheet_data in excel_data.values():
                all_text += sheet_data['text'] + "\n"
            self.raw_text = all_text
            self.extracted_data['excel_sheets'] = excel_data
        
        # Extract financial metrics
        self.extracted_data['metrics'] = self.extract_financial_metrics(self.raw_text)
        self.extracted_data['raw_text'] = self.raw_text
        
        return self.extracted_data

class OllamaClient:
    """Client for interacting with Ollama local LLM"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response using Ollama"""
        try:
            # Prepare the full prompt with context
            full_prompt = f"""
            Context from financial document:
            {context[:2000]}  # Limit context to avoid token limits
            
            User Question: {prompt}
            
            Please answer the question based on the financial document context provided above. 
            Be specific and use numbers when available. If you cannot find the information, 
            say so clearly.
            """
            
            payload = {
                "model": "gemma:2b",  # You can change this to your preferred model
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Could not connect to Ollama (Status: {response.status_code})"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Please make sure Ollama is running on localhost:11434"
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.title("📊 Financial Document Q&A Assistant")
    st.markdown("Upload your financial documents and ask questions about the data!")
    
    # Initialize session state
    if 'document_data' not in st.session_state:
        st.session_state.document_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a financial document",
            type=['pdf', 'xlsx', 'xls'],
            help="Upload PDF or Excel files containing financial statements"
        )
        
        if uploaded_file is not None:
            file_type = "pdf" if uploaded_file.name.endswith('.pdf') else "excel"
            
            with st.spinner("Processing document..."):
                processor = DocumentProcessor()
                document_data = processor.process_document(uploaded_file, file_type)
                st.session_state.document_data = document_data
                
            st.success("✅ Document processed successfully!")
            
            # Display extracted metrics
            if document_data.get('metrics'):
                st.subheader("📈 Extracted Financial Metrics")
                for metric, value in document_data['metrics'].items():
                    st.metric(metric.title(), f"${value:,.2f}" if isinstance(value, float) else value)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions About Your Document")
        
        if st.session_state.document_data is None:
            st.info("👈 Please upload a financial document to get started!")
        else:
            # Initialize Ollama client
            ollama_client = OllamaClient()
            
            # Chat interface
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {question[:50]}..."):
                        st.write(f"**Question:** {question}")
                        st.write(f"**Answer:** {answer}")
            
            # Question input
            question = st.text_input(
                "Ask a question about your financial document:",
                placeholder="e.g., What was the total revenue? What are the main expenses?"
            )
            
            if st.button("Get Answer") and question:
                with st.spinner("Generating answer..."):
                    context = st.session_state.document_data['raw_text']
                    answer = ollama_client.generate_response(question, context)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display answer
                    st.success("**Answer:**")
                    st.write(answer)
            
            # Quick questions
            st.subheader("🚀 Quick Questions")
            quick_questions = [
                "What was the total revenue?",
                "What were the main expenses?",
                "What was the net profit?",
                "What are the total assets?",
                "Show me the key financial ratios"
            ]
            
            for quick_q in quick_questions:
                if st.button(quick_q):
                    with st.spinner("Generating answer..."):
                        context = st.session_state.document_data['raw_text']
                        answer = ollama_client.generate_response(quick_q, context)
                        st.session_state.chat_history.append((quick_q, answer))
                        st.success("**Answer:**")
                        st.write(answer)
    
    with col2:
        st.header("📄 Document Summary")
        
        if st.session_state.document_data:
            # Show document stats
            st.subheader("Document Statistics")
            raw_text = st.session_state.document_data['raw_text']
            st.write(f"**Characters:** {len(raw_text):,}")
            st.write(f"**Words:** {len(raw_text.split()):,}")
            
            # Show raw text preview
            with st.expander("Preview Document Text"):
                st.text_area("Document Content", raw_text[:1000] + "...", height=200)
            
            # Excel sheets info
            if 'excel_sheets' in st.session_state.document_data:
                st.subheader("Excel Sheets")
                for sheet_name, sheet_data in st.session_state.document_data['excel_sheets'].items():
                    with st.expander(f"Sheet: {sheet_name}"):
                        st.dataframe(sheet_data['dataframe'].head())
        
        # Instructions
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. Upload a financial document (PDF or Excel)
        2. Wait for processing to complete
        3. Ask questions about your financial data
        4. Use quick questions or type custom queries
        
        **Example Questions:**
        - What was the revenue last quarter?
        - Show me the expense breakdown
        - What's the profit margin?
        - Compare assets vs liabilities
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Note:** Make sure Ollama is running locally with a language model installed. "
        "Run `ollama serve` and `ollama pull llama2` in your terminal."
    )

if __name__ == "__main__":
    main()