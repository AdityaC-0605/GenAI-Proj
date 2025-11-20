"""Multi-page Streamlit Dashboard for Cross-Lingual Question Answering."""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Cross-Lingual QA",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-row {
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing results history
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

# Sidebar navigation
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Question Answering", "📊 Results Analytics"],
    label_visibility="collapsed"
)

# Helper functions
def check_api_status():
    """Check if API server is reachable."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_openai_status():
    """Check if OpenAI API key is configured."""
    return bool(os.getenv('OPENAI_API_KEY'))

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🌍 Cross-Lingual Question Answering</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multilingual AI-Powered Question Answering System</p>', unsafe_allow_html=True)
    
    # System status
    st.markdown("### 🔧 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if check_api_status():
            st.success("✅ API Server Online")
        else:
            st.error("❌ API Server Offline")
    
    with col2:
        st.info(f"📊 Results History: {len(st.session_state.results_history)}")
    
    with col3:
        st.info("🌍 Ready to Answer")
    
    st.markdown("---")
    
    # Welcome section
    st.markdown("## 👋 Welcome!")
    st.markdown("""
    This dashboard provides an interactive interface for testing cross-lingual question answering capabilities 
    using state-of-the-art multilingual models.
    """)
    
    # Features
    st.markdown("## ✨ Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔵 mBERT Model
        - Multilingual BERT architecture
        - Finds exact text spans in context
        - Fast inference (~100-200ms)
        - High precision for factual questions
        - Supports 100+ languages
        """)
    
    with col2:
        st.markdown("""
        ### 🟢 mT5 Model
        - Multilingual T5 architecture
        - Generates natural language answers
        - Flexible response generation
        - Better for explanatory questions
        - Supports 100+ languages
        """)
    
    st.markdown("---")
    
    # Supported languages
    st.markdown("## 🌐 Supported Languages")
    
    languages = {
        "🇬🇧 English": "en",
        "🇪🇸 Spanish": "es",
        "🇫🇷 French": "fr",
        "🇩🇪 German": "de",
        "🇨🇳 Chinese": "zh",
        "🇸🇦 Arabic": "ar"
    }
    
    cols = st.columns(3)
    for idx, (lang_name, lang_code) in enumerate(languages.items()):
        with cols[idx % 3]:
            st.info(f"{lang_name}")
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start")
    
    st.markdown("""
    1. **Navigate** to the Question Answering page using the sidebar
    2. **Enter** your question and provide context
    3. **Select** languages for question and context
    4. **Choose** one or both models to compare
    5. **Click** "Get Answers" to see results
    6. **View** analytics in the Results Analytics page
    """)
    
    # Statistics
    if st.session_state.results_history:
        st.markdown("---")
        st.markdown("## 📈 Recent Activity")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", len(st.session_state.results_history))
        
        with col2:
            avg_conf = sum(r.get('confidence', 0) for r in st.session_state.results_history) / len(st.session_state.results_history)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        with col3:
            avg_time = sum(r.get('processing_time_ms', 0) for r in st.session_state.results_history) / len(st.session_state.results_history)
            st.metric("Avg Response Time", f"{avg_time:.0f}ms")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Cross-Lingual Question Answering System | mBERT & mT5 Models</p>
        <p style='font-size: 0.9rem;'>Built with Streamlit • Powered by Transformers</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# QUESTION ANSWERING PAGE
# ============================================================================
elif page == "🔍 Question Answering":
    # Import the QA functionality
    from datetime import datetime
    
    def validate_inputs(question, context, use_mbert, use_mt5):
        """Validate user inputs before submission."""
        if not question or not question.strip():
            return False, "❌ Please provide a question"
        if not context or not context.strip():
            return False, "❌ Please provide a context passage"
        if not use_mbert and not use_mt5:
            return False, "❌ Please select at least one model"
        return True, ""
    
    def send_prediction_request(question, context, q_lang, c_lang, model_name):
        """Send prediction request to API server."""
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "question": question,
                "context": context,
                "question_language": q_lang,
                "context_language": c_lang,
                "model_name": model_name
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def handle_api_error(error, model_name):
        """Handle API errors and display appropriate messages."""
        if isinstance(error, requests.exceptions.ConnectionError):
            st.error(f"❌ Cannot connect to API server for {model_name}. Please start the API server first.")
            st.code("python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000")
        elif isinstance(error, requests.exceptions.Timeout):
            st.error(f"❌ Request timed out for {model_name}. The model might be loading or the query is too complex.")
        elif isinstance(error, requests.exceptions.HTTPError):
            st.error(f"❌ API Error for {model_name}: {error.response.status_code} - {error.response.text}")
        else:
            st.error(f"❌ Error with {model_name}: {str(error)}")
    
    def get_confidence_color(confidence):
        """Get color coding for confidence score."""
        if confidence >= 0.85:
            return "green", "Excellent"
        elif confidence >= 0.70:
            return "yellow", "Good"
        else:
            return "orange", "Low"
    
    def display_single_result(model_name, result):
        """Display result for a single model."""
        st.markdown(f"### {model_name}")
        
        answer = result['answer']
        confidence = result['confidence']
        processing_time = result['processing_time_ms']
        
        st.markdown(f"**Answer:** {answer}")
        
        color, label = get_confidence_color(confidence)
        
        col1, col2 = st.columns(2)
        with col1:
            if color == "green":
                st.success(f"Confidence: {confidence:.2%} - {label}")
            elif color == "yellow":
                st.info(f"Confidence: {confidence:.2%} - {label}")
            else:
                st.warning(f"Confidence: {confidence:.2%} - {label}")
        
        with col2:
            st.metric("Processing Time", f"{processing_time:.0f}ms")
    
    def display_comparison_results(mbert_result, mt5_result):
        """Display side-by-side comparison of both models."""
        st.markdown("## 📊 Results Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 mBERT")
            st.markdown(f"**Answer:** {mbert_result['answer']}")
            
            color, label = get_confidence_color(mbert_result['confidence'])
            if color == "green":
                st.success(f"Confidence: {mbert_result['confidence']:.2%} - {label}")
            elif color == "yellow":
                st.info(f"Confidence: {mbert_result['confidence']:.2%} - {label}")
            else:
                st.warning(f"Confidence: {mbert_result['confidence']:.2%} - {label}")
            
            st.caption(f"⏱️ {mbert_result['processing_time_ms']:.0f}ms")
        
        with col2:
            st.markdown("### 🟢 mT5")
            st.markdown(f"**Answer:** {mt5_result['answer']}")
            
            color, label = get_confidence_color(mt5_result['confidence'])
            if color == "green":
                st.success(f"Confidence: {mt5_result['confidence']:.2%} - {label}")
            elif color == "yellow":
                st.info(f"Confidence: {mt5_result['confidence']:.2%} - {label}")
            else:
                st.warning(f"Confidence: {mt5_result['confidence']:.2%} - {label}")
            
            st.caption(f"⏱️ {mt5_result['processing_time_ms']:.0f}ms")
        
        st.markdown("---")
        st.markdown("### 📈 Comparison Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            conf_diff = mt5_result['confidence'] - mbert_result['confidence']
            better_model = "mT5" if conf_diff > 0 else "mBERT"
            st.metric(
                "Confidence Difference",
                f"{abs(conf_diff):.2%}",
                delta=f"{better_model} higher"
            )
        
        with col2:
            time_diff = mt5_result['processing_time_ms'] - mbert_result['processing_time_ms']
            faster_model = "mBERT" if time_diff > 0 else "mT5"
            st.metric(
                "Speed Difference",
                f"{abs(time_diff):.0f}ms",
                delta=f"{faster_model} faster"
            )
        
        with col3:
            if mbert_result['answer'].strip().lower() == mt5_result['answer'].strip().lower():
                st.info("✓ Answers Match")
            else:
                st.warning("✗ Answers Differ")
    
    # Main QA interface
    st.markdown('<h1 class="main-header">🔍 Question Answering</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions and get answers from both models</p>', unsafe_allow_html=True)
    
    # System status
    col1, col2 = st.columns(2)
    with col1:
        if check_api_status():
            st.success("✅ API Server Online")
        else:
            st.error("❌ API Server Offline")
    with col2:
        st.info("🌍 Cross-Lingual QA Ready")
    
    st.markdown("---")
    
    # Language options
    LANGUAGES = {
        "en": "🇬🇧 English",
        "es": "🇪🇸 Spanish",
        "fr": "🇫🇷 French",
        "de": "🇩🇪 German",
        "zh": "🇨🇳 Chinese",
        "ar": "🇸🇦 Arabic"
    }
    
    # Input interface
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📝 Input")
        
        question = st.text_input(
            "Question",
            placeholder="Enter your question here...",
            help="Type the question you want to ask"
        )
        
        context = st.text_area(
            "Context",
            placeholder="Paste the context passage that contains the answer...",
            height=150,
            help="Provide the text passage containing the answer"
        )
    
    with col_right:
        st.markdown("### ⚙️ Configuration")
        
        q_lang = st.selectbox(
            "Question Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            index=0,
            help="Select the language of your question"
        )
        
        c_lang = st.selectbox(
            "Context Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            index=0,
            help="Select the language of your context"
        )
        
        st.markdown("---")
        st.markdown("**Model Selection**")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            use_mbert = st.checkbox(
                "mBERT",
                value=True,
                help="Multilingual BERT model"
            )
        
        with col_b:
            use_mt5 = st.checkbox(
                "mT5",
                value=True,
                help="Multilingual T5 model"
            )
        
        st.caption("💡 Select one or both models to compare")
    
    st.markdown("---")
    
    # Submit button
    if st.button("🔍 Get Answers", type="primary", use_container_width=True):
        is_valid, error_msg = validate_inputs(question, context, use_mbert, use_mt5)
        
        if not is_valid:
            st.error(error_msg)
        else:
            # Check for language mismatch with comprehensive detection
            import re
            
            # Detect actual language of context
            detected_lang = None
            context_lower = context.lower()
            
            # Check for non-Latin scripts first (most obvious)
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', context))
            if chinese_chars > len(context) * 0.15:
                detected_lang = "zh"
            
            arabic_chars = len(re.findall(r'[\u0600-\u06ff]', context))
            if arabic_chars > len(context) * 0.15:
                detected_lang = "ar"
            
            # Check for Latin script languages by common words and patterns
            if not detected_lang:
                # German indicators
                german_words = ['der', 'die', 'das', 'und', 'ist', 'von', 'wurde', 'als', 'einem', 'einer']
                german_chars = ['ä', 'ö', 'ü', 'ß']
                german_score = sum(1 for word in german_words if f' {word} ' in f' {context_lower} ')
                german_score += sum(1 for char in german_chars if char in context_lower) * 2
                
                # Spanish indicators
                spanish_words = ['el', 'la', 'de', 'que', 'es', 'en', 'por', 'como', 'del', 'los']
                spanish_chars = ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
                spanish_score = sum(1 for word in spanish_words if f' {word} ' in f' {context_lower} ')
                spanish_score += sum(1 for char in spanish_chars if char in context_lower) * 2
                
                # French indicators
                french_words = ['le', 'la', 'de', 'et', 'est', 'dans', 'pour', 'que', 'une', 'des']
                french_chars = ['à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û']
                french_score = sum(1 for word in french_words if f' {word} ' in f' {context_lower} ')
                french_score += sum(1 for char in french_chars if char in context_lower) * 2
                
                # Determine language based on scores
                scores = {
                    'de': german_score,
                    'es': spanish_score,
                    'fr': french_score
                }
                
                max_score = max(scores.values())
                if max_score >= 3:  # Threshold for detection
                    detected_lang = max(scores, key=scores.get)
            
            # Show warning if mismatch detected
            if detected_lang and detected_lang != c_lang:
                lang_names = {
                    'zh': '🇨🇳 Chinese',
                    'ar': '🇸🇦 Arabic',
                    'de': '🇩🇪 German',
                    'es': '🇪🇸 Spanish',
                    'fr': '🇫🇷 French'
                }
                st.warning(
                    f"⚠️ **Language Mismatch Detected!**\n\n"
                    f"Context appears to be in **{lang_names.get(detected_lang, detected_lang.upper())}**, "
                    f"but you selected **{LANGUAGES[c_lang]}**.\n\n"
                    f"Please select the correct language for accurate results. "
                    f"Confidence scores may be reduced due to this mismatch."
                )
            
            results = {}
            
            if use_mbert:
                with st.spinner("🔵 Testing mBERT..."):
                    try:
                        results['mbert'] = send_prediction_request(
                            question, context, q_lang, c_lang, "mbert"
                        )
                    except Exception as e:
                        handle_api_error(e, "mBERT")
            
            if use_mt5:
                with st.spinner("🟢 Testing mT5..."):
                    try:
                        results['mt5'] = send_prediction_request(
                            question, context, q_lang, c_lang, "mt5"
                        )
                    except Exception as e:
                        handle_api_error(e, "mT5")
            
            if results:
                st.markdown("---")
                
                # Store results in history
                for model_name, result in results.items():
                    st.session_state.results_history.append({
                        'timestamp': datetime.now(),
                        'model': model_name,
                        'question': question,
                        'answer': result['answer'],
                        'confidence': result['confidence'],
                        'processing_time_ms': result['processing_time_ms'],
                        'q_lang': q_lang,
                        'c_lang': c_lang
                    })
                
                if 'mbert' in results and 'mt5' in results:
                    display_comparison_results(results['mbert'], results['mt5'])
                elif 'mbert' in results:
                    st.markdown("## 📊 Results")
                    display_single_result("🔵 mBERT", results['mbert'])
                elif 'mt5' in results:
                    st.markdown("## 📊 Results")
                    display_single_result("🟢 mT5", results['mt5'])
    
    st.markdown("---")
    
    # Info sections
    with st.expander("ℹ️ About the Models"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔵 mBERT**
            - Multilingual BERT model
            - Finds exact text spans in context
            - Fast and precise
            - Best for factual questions
            """)
        
        with col2:
            st.markdown("""
            **🟢 mT5**
            - Multilingual T5 model
            - Creates natural language answers
            - More fluent responses
            - Best for explanatory questions
            """)
    
    with st.expander("🌍 Cross-Lingual Testing Tips"):
        st.markdown("""
        **Test cross-lingual capabilities:**
        
        1. Keep question in English
        2. Change context language to Spanish/French/German/etc.
        3. See if models can still answer correctly!
        
        **Example:**
        - Question (English): "What is the capital of France?"
        - Context (Spanish): "París es la capital de Francia..."
        - Expected: Models should answer "Paris" or "París"
        
        💡 Check `TEST_QUESTIONS.md` for ready-to-use test questions!
        """)

# ============================================================================
# RESULTS ANALYTICS PAGE
# ============================================================================
elif page == "📊 Results Analytics":
    st.markdown('<h1 class="main-header">📊 Results Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visualize and analyze model performance</p>', unsafe_allow_html=True)
    
    if not st.session_state.results_history:
        st.info("📭 No results yet. Go to the Question Answering page to generate some results!")
    else:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.results_history)
        
        # Summary metrics
        st.markdown("## 📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(df))
        
        with col2:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            avg_time = df['processing_time_ms'].mean()
            st.metric("Avg Response Time", f"{avg_time:.0f}ms")
        
        with col4:
            unique_langs = df['c_lang'].nunique()
            st.metric("Languages Used", unique_langs)
        
        st.markdown("---")
        
        # Model comparison
        st.markdown("## 🔄 Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence by model
            fig_conf = px.box(
                df,
                x='model',
                y='confidence',
                color='model',
                title='Confidence Distribution by Model',
                labels={'model': 'Model', 'confidence': 'Confidence Score'},
                color_discrete_map={'mbert': '#1f77b4', 'mt5': '#2ca02c'}
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Processing time by model
            fig_time = px.box(
                df,
                x='model',
                y='processing_time_ms',
                color='model',
                title='Processing Time by Model',
                labels={'model': 'Model', 'processing_time_ms': 'Time (ms)'},
                color_discrete_map={'mbert': '#1f77b4', 'mt5': '#2ca02c'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        st.markdown("---")
        
        # Language analysis
        st.markdown("## 🌐 Language Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Language distribution
            lang_counts = df['c_lang'].value_counts()
            fig_lang = px.pie(
                values=lang_counts.values,
                names=lang_counts.index,
                title='Context Language Distribution'
            )
            st.plotly_chart(fig_lang, use_container_width=True)
        
        with col2:
            # Confidence by language
            fig_lang_conf = px.bar(
                df.groupby('c_lang')['confidence'].mean().reset_index(),
                x='c_lang',
                y='confidence',
                title='Average Confidence by Language',
                labels={'c_lang': 'Language', 'confidence': 'Avg Confidence'},
                color='confidence',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_lang_conf, use_container_width=True)
        
        st.markdown("---")
        
        # Timeline
        st.markdown("## ⏱️ Performance Over Time")
        
        df_sorted = df.sort_values('timestamp')
        
        fig_timeline = go.Figure()
        
        for model in df['model'].unique():
            model_data = df_sorted[df_sorted['model'] == model]
            fig_timeline.add_trace(go.Scatter(
                x=model_data['timestamp'],
                y=model_data['confidence'],
                mode='lines+markers',
                name=model.upper(),
                line=dict(width=2)
            ))
        
        fig_timeline.update_layout(
            title='Confidence Scores Over Time',
            xaxis_title='Timestamp',
            yaxis_title='Confidence',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("---")
        
        # Recent results table
        st.markdown("## 📋 Recent Results")
        
        display_df = df[['timestamp', 'model', 'question', 'answer', 'confidence', 'processing_time_ms']].tail(10)
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['processing_time_ms'] = display_df['processing_time_ms'].apply(lambda x: f"{x:.0f}ms")
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Clear history button
        st.markdown("---")
        if st.button("🗑️ Clear Results History", type="secondary"):
            st.session_state.results_history = []
            st.rerun()
