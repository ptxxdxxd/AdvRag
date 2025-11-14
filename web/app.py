"""
Streamlit Web Interface for Advanced RAG System
Provides comprehensive dashboard for prompt engineering and generative AI portfolio
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Import our core modules
from core.rag_pipeline import AdvancedRAGPipeline, QueryResult
from core.fine_tuning_engine import FineTuningEngine

# Page configuration
st.set_page_config(
    page_title="Advanced RAG System - Portfolio Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fef3c7;
        color: #92400e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'fine_tuning_engine' not in st.session_state:
    st.session_state.fine_tuning_engine = None
if 'demo_data_loaded' not in st.session_state:
    st.session_state.demo_data_loaded = False


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced RAG System</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Portfolio Project**: Demonstrating Prompt Engineering and Generative AI Engineering Skills
    
    This system combines custom model fine-tuning, intelligent retrieval, and prompt optimization 
    to create a production-ready AI assistant.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a section:", [
        "🏠 Dashboard",
        "💬 Query System", 
        "📊 Analytics",
        "🔧 Fine-tuning",
        "⚙️ System Config"
    ])
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "💬 Query System":
        show_query_system()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "🔧 Fine-tuning":
        show_fine_tuning()
    elif page == "⚙️ System Config":
        show_system_config()


def show_dashboard():
    """Display the main dashboard with system status and key metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Active", "Ready for queries")
    
    with col2:
        if st.session_state.rag_pipeline:
            st.metric("Documents Loaded", len(st.session_state.rag_pipeline.documents), "+ this session")
        else:
            st.metric("Documents Loaded", "0", "Load documents to begin")
    
    with col3:
        if st.session_state.rag_pipeline:
            st.metric("Total Queries", st.session_state.rag_pipeline.performance_metrics["total_queries"])
        else:
            st.metric("Total Queries", "0", "No queries yet")
    
    with col4:
        if st.session_state.rag_pipeline:
            avg_cost = st.session_state.rag_pipeline.performance_metrics["avg_cost_per_query"]
            st.metric("Avg Cost/Query", f"${avg_cost:.4f}")
        else:
            st.metric("Avg Cost/Query", "$0.0000")
    
    # Initialize pipeline if not done
    if st.session_state.rag_pipeline is None:
        if st.button("🚀 Initialize RAG System"):
            with st.spinner("Initializing AI models..."):
                st.session_state.rag_pipeline = AdvancedRAGPipeline()
                success = st.session_state.rag_pipeline.initialize_models()
                
                if success:
                    st.success("✅ RAG Pipeline initialized successfully!")
                    
                    # Load demo data if available
                    load_demo_data()
                else:
                    st.error("❌ Failed to initialize RAG Pipeline")
    
    # Quick demo section
    st.markdown('<h2 class="sub-header">🎯 Quick Demo</h2>', unsafe_allow_html=True)
    
    if st.session_state.rag_pipeline and st.session_state.rag_pipeline.documents:
        
        # Pre-defined demo queries
        demo_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain the benefits of fine-tuning models",
            "What are the key challenges in AI development?",
            "How can AI improve business processes?"
        ]
        
        selected_query = st.selectbox("Try a demo query:", demo_queries)
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Run Query", type="primary"):
                with st.spinner("Processing query..."):
                    result = st.session_state.rag_pipeline.generate_response(selected_query)
                    
                    # Store result for display
                    st.session_state.last_query_result = result
        
        with col2:
            if 'last_query_result' in st.session_state:
                result = st.session_state.last_query_result
                
                # Display result in a nice format
                with st.container():
                    st.markdown("### 📝 Response")
                    st.write(result.response)
                    
                    if result.sources:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result.sources[:3], 1):
                            st.write(f"**Source {i}** ({source['source']}): Score {source['score']:.3f}")
                            st.write(source['text'])
                            
                            if i >= 2:  # Show top 2 sources
                                break
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retrieval Score", f"{result.retrieval_score:.3f}")
                    with col2:
                        st.metric("Generation Time", f"{result.generation_time:.2f}s")
                    with col3:
                        st.metric("Cost", f"${result.cost:.4f}")
    
    else:
        st.markdown("""
        ### 📁 Load Demo Data
        
        To get started with the demo, you need to load some documents. 
        The system works best with technical documents, research papers, or business content.
        """)
        
        if st.button("📁 Load Demo Documents"):
            load_demo_data()
            
            if st.session_state.rag_pipeline and st.session_state.rag_pipeline.documents:
                st.success("✅ Demo documents loaded successfully!")
            else:
                st.error("❌ Failed to load demo documents")


def load_demo_data():
    """Load demo documents for testing"""
    try:
        # Create demo documents
        demo_documents = [
            {
                "content": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create 
                intelligent machines that work and react like humans. AI encompasses several key areas: 
                Machine Learning algorithms that learn from data patterns, Natural Language Processing 
                for understanding human language, Computer Vision for interpreting visual information, 
                and Robotics for physical world interactions. The field has evolved significantly since 
                the 1950s, with modern AI systems capable of complex tasks like image recognition, 
                language translation, and strategic game playing. Deep learning, a subset of machine 
                learning, has been particularly successful in achieving human-level performance in 
                many domains. However, AI systems still face challenges including bias in training 
                data, explainability of decisions, and alignment with human values.
                """,
                "source": "ai_introduction.txt",
                "metadata": {"topic": "AI", "type": "educational"}
            },
            {
                "content": """
                Fine-tuning is a transfer learning technique where a pre-trained model is adapted 
                to a specific task or domain by training it on a smaller, task-specific dataset. 
                This approach offers several advantages: faster training times, reduced data requirements, 
                and improved performance compared to training from scratch. The process involves taking 
                a model trained on a large, general dataset (like ImageNet for computer vision or 
                Wikipedia for language models) and continuing training on domain-specific data. 
                Popular fine-tuning techniques include LoRA (Low-Rank Adaptation), which reduces 
                computational costs, and prompt engineering, which optimizes input text without 
                changing model weights. Fine-tuning is essential for creating specialized AI systems 
                that perform well on specific tasks while maintaining the general capabilities of 
                the base model.
                """,
                "source": "fine_tuning_guide.txt",
                "metadata": {"topic": "ML", "type": "technical"}
            },
            {
                "content": """
                Prompt engineering is the practice of designing and optimizing input prompts to 
                achieve desired outputs from large language models. Effective prompts are clear, 
                specific, and provide sufficient context for the model to generate accurate responses. 
                Key techniques include: Chain-of-thought prompting, which guides the model through 
                reasoning steps; Few-shot learning, providing examples to establish patterns; and 
                Role prompting, assigning specific roles to the AI system. Best practices involve 
                using concrete examples, specifying output formats, and including relevant constraints. 
                Prompt optimization can significantly improve model performance on specific tasks, 
                making it a crucial skill for AI practitioners. Advanced techniques include prompt 
                chaining, where complex tasks are broken into simpler steps, and prompt templates 
                that ensure consistency across multiple interactions.
                """,
                "source": "prompt_engineering.txt", 
                "metadata": {"topic": "AI", "type": "technical"}
            },
            {
                "content": """
                Business applications of AI have transformed numerous industries through automation, 
                optimization, and enhanced decision-making capabilities. In customer service, AI chatbots 
                provide 24/7 support while reducing operational costs. In finance, AI systems detect 
                fraud, assess credit risk, and optimize trading strategies. Healthcare applications 
                include diagnostic assistance, drug discovery acceleration, and personalized treatment 
                recommendations. Retail companies use AI for inventory management, recommendation 
                systems, and demand forecasting. Manufacturing benefits from predictive maintenance, 
                quality control automation, and supply chain optimization. The key to successful AI 
                implementation lies in identifying appropriate use cases, ensuring data quality, and 
                maintaining human oversight. ROI from AI investments typically includes cost reduction, 
                revenue growth through improved services, and competitive advantages through faster 
                decision-making and enhanced customer experiences.
                """,
                "source": "business_ai_applications.txt",
                "metadata": {"topic": "Business", "type": "application"}
            }
        ]
        
        # Initialize pipeline and load documents
        if not st.session_state.rag_pipeline:
            st.session_state.rag_pipeline = AdvancedRAGPipeline()
            st.session_state.rag_pipeline.initialize_models()
        
        # Load documents
        st.session_state.rag_pipeline.documents = demo_documents
        
        # Create chunks
        success = st.session_state.rag_pipeline.create_chunks() and st.session_state.rag_pipeline.create_embeddings()
        
        st.session_state.demo_data_loaded = success
        
        return success
        
    except Exception as e:
        st.error(f"Failed to load demo data: {str(e)}")
        return False


def show_query_system():
    """Display the interactive query system"""
    
    st.markdown('<h2 class="sub-header">💬 Interactive Query System</h2>', unsafe_allow_html=True)
    
    if not st.session_state.rag_pipeline:
        st.warning("Please initialize the RAG system first from the Dashboard.")
        return
    
    # Query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your question:", placeholder="Ask anything about AI, fine-tuning, prompt engineering...")
    
    with col2:
        retrieval_method = st.selectbox("Retrieval Method:", ["hybrid", "embeddings", "tfidf"])
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔍 Submit Query", type="primary"):
            if query.strip():
                with st.spinner("Processing your query..."):
                    result = st.session_state.rag_pipeline.generate_response(query, retrieval_method)
                    st.session_state.current_result = result
            else:
                st.warning("Please enter a query first.")
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.rag_pipeline.query_history = []
            st.session_state.rag_pipeline.performance_metrics = {
                "total_queries": 0,
                "avg_retrieval_time": 0,
                "avg_generation_time": 0,
                "avg_cost_per_query": 0,
                "total_cost": 0
            }
            st.success("Query history cleared!")
    
    with col3:
        if st.button("📊 Export Results"):
            if hasattr(st.session_state, 'current_result'):
                export_results(st.session_state.current_result)
    
    # Display current result
    if hasattr(st.session_state, 'current_result'):
        display_query_result(st.session_state.current_result)
    
    # Query history
    if st.session_state.rag_pipeline.query_history:
        st.markdown("### 📚 Recent Query History")
        
        for i, result in enumerate(reversed(st.session_state.rag_pipeline.query_history[-10:])):
            with st.expander(f"Query {len(st.session_state.rag_pipeline.query_history) - i}: {result.query[:50]}..."):
                st.write(f"**Response**: {result.response[:200]}...")
                st.write(f"**Sources**: {len(result.sources)} documents")
                st.write(f"**Time**: {result.generation_time:.2f}s | **Cost**: ${result.cost:.4f}")


def display_query_result(result: QueryResult):
    """Display a query result in a formatted way"""
    
    # Response section
    st.markdown("### 🤖 AI Response")
    st.write(result.response)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Retrieval Score", f"{result.retrieval_score:.3f}")
    with col2:
        st.metric("Generation Time", f"{result.generation_time:.3f}s")
    with col3:
        st.metric("Query Cost", f"${result.cost:.5f}")
    with col4:
        st.metric("Context Chunks", len(result.sources))
    
    # Sources section
    if result.sources:
        st.markdown("### 📚 Retrieved Sources")
        
        for i, source in enumerate(result.sources, 1):
            with st.expander(f"Source {i}: {source['source']} (Score: {source['score']:.3f})"):
                st.write(source['text'])
        
        # Source analysis
        source_scores = [s['score'] for s in result.sources]
        source_df = pd.DataFrame({
            'Source': [s['source'] for s in result.sources],
            'Score': [s['score'] for s in result.sources]
        })
        
        fig = px.bar(source_df, x='Source', y='Score', title="Source Relevance Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    # Metadata
    with st.expander("🔧 Technical Details"):
        st.json(result.metadata)


def export_results(result: QueryResult):
    """Export query results to files"""
    
    # Create export data
    export_data = {
        "query": result.query,
        "response": result.response,
        "sources": result.sources,
        "metrics": {
            "retrieval_score": result.retrieval_score,
            "generation_time": result.generation_time,
            "cost": result.cost
        },
        "metadata": result.metadata,
        "timestamp": datetime.now().isoformat()
    }
    
    # Convert to JSON
    export_json = json.dumps(export_data, indent=2)
    
    # Download button
    st.download_button(
        label="💾 Download Results (JSON)",
        data=export_json,
        file_name=f"query_result_{int(time.time())}.json",
        mime="application/json"
    )


def show_analytics():
    """Display comprehensive analytics and performance metrics"""
    
    st.markdown('<h2 class="sub-header">📊 Performance Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.rag_pipeline or not st.session_state.rag_pipeline.query_history:
        st.info("No query data available for analytics. Start making queries to see performance metrics.")
        return
    
    # Performance overview
    metrics = st.session_state.rag_pipeline.performance_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", metrics["total_queries"])
    with col2:
        st.metric("Avg Generation Time", f"{metrics['avg_generation_time']:.3f}s")
    with col3:
        st.metric("Total Cost", f"${metrics['total_cost']:.4f}")
    with col4:
        st.metric("Cost per Query", f"${metrics['avg_cost_per_query']:.5f}")
    
    # Query analysis charts
    if st.session_state.rag_pipeline.query_history:
        query_df = pd.DataFrame([
            {
                "timestamp": q.generation_time,  # This would be actual timestamp
                "response_time": q.generation_time,
                "cost": q.cost,
                "retrieval_score": q.retrieval_score,
                "query": q.query[:50] + "..." if len(q.query) > 50 else q.query
            }
            for q in st.session_state.rag_pipeline.query_history
        ])
        
        if not query_df.empty:
            # Response time trend
            fig_time = px.line(
                query_df, 
                x=query_df.index, 
                y="response_time",
                title="Response Time Over Queries",
                labels={"index": "Query Number", "response_time": "Time (seconds)"}
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Cost analysis
            fig_cost = px.line(
                query_df,
                x=query_df.index,
                y="cost",
                title="Query Cost Over Time",
                labels={"index": "Query Number", "cost": "Cost ($)"}
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            
            # Retrieval score distribution
            fig_score = px.histogram(
                query_df,
                x="retrieval_score",
                title="Distribution of Retrieval Scores",
                nbins=20
            )
            st.plotly_chart(fig_score, use_container_width=True)
            
            # Performance correlation
            fig_corr = px.scatter(
                query_df,
                x="response_time",
                y="retrieval_score",
                color="cost",
                title="Response Time vs Retrieval Score",
                labels={"response_time": "Response Time (s)", "retrieval_score": "Retrieval Score"}
            )
            st.plotly_chart(fig_corr, use_container_width=True)


def show_fine_tuning():
    """Display fine-tuning interface and tools"""
    
    st.markdown('<h2 class="sub-header">🔧 Custom Model Fine-tuning</h2>', unsafe_allow_html=True)
    
    # Initialize fine-tuning engine if not done
    if st.session_state.fine_tuning_engine is None:
        st.session_state.fine_tuning_engine = FineTuningEngine()
    
    # Fine-tuning options
    st.markdown("### 🎯 Fine-tuning Options")
    
    fine_tuning_method = st.selectbox(
        "Choose fine-tuning approach:",
        ["Supervised Learning (Classification)", "Contrastive Learning", "Hyperparameter Optimization"]
    )
    
    if fine_tuning_method == "Supervised Learning (Classification)":
        show_supervised_fine_tuning()
    elif fine_tuning_method == "Contrastive Learning":
        show_contrastive_fine_tuning()
    elif fine_tuning_method == "Hyperparameter Optimization":
        show_hyperparameter_optimization()


def show_supervised_fine_tuning():
    """Interface for supervised fine-tuning"""
    
    st.markdown("### 📚 Supervised Fine-tuning")
    st.write("Fine-tune embeddings on labeled data for classification tasks.")
    
    # Generate or load training data
    st.markdown("#### Data Preparation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.radio("Training data source:", ["Generate Synthetic", "Upload Custom"])
    
    with col2:
        if data_source == "Generate Synthetic":
            num_samples = st.slider("Number of samples", 100, 5000, 1000)
            domain = st.selectbox("Domain:", ["general", "technical", "medical", "legal"])
            generate_button = st.button("🧬 Generate Data")
            
            if generate_button:
                with st.spinner("Generating synthetic training data..."):
                    texts, labels = st.session_state.fine_tuning_engine.generate_synthetic_data(
                        num_samples, domain
                    )
                    st.session_state.training_data = (texts, labels)
                    st.success(f"Generated {len(texts)} training samples!")
    
    # Fine-tuning configuration
    if 'training_data' in st.session_state:
        st.markdown("#### Fine-tuning Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            learning_rate = st.slider("Learning Rate", 1e-6, 1e-3, 2e-5, format="%.1e")
        with col2:
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64])
        with col3:
            num_epochs = st.slider("Epochs", 1, 10, 3)
        with col4:
            max_seq_length = st.selectbox("Max Length", [256, 512, 1024])
        
        # Start fine-tuning
        if st.button("🚀 Start Fine-tuning", type="primary"):
            with st.spinner("Fine-tuning model..."):
                try:
                    # Configure training
                    st.session_state.fine_tuning_engine.training_config.learning_rate = learning_rate
                    st.session_state.fine_tuning_engine.training_config.batch_size = batch_size
                    st.session_state.fine_tuning_engine.training_config.num_epochs = num_epochs
                    st.session_state.fine_tuning_engine.training_config.max_seq_length = max_seq_length
                    
                    # Initialize model
                    st.session_state.fine_tuning_engine.initialize_model()
                    
                    # Run fine-tuning
                    texts, labels = st.session_state.training_data
                    results = st.session_state.fine_tuning_engine.fine_tune_embeddings(
                        texts, labels, save_model=True
                    )
                    
                    st.session_state.fine_tuning_results = results
                    
                    # Display results
                    display_fine_tuning_results(results)
                    
                except Exception as e:
                    st.error(f"Fine-tuning failed: {str(e)}")


def show_contrastive_fine_tuning():
    """Interface for contrastive learning"""
    
    st.markdown("### 🔄 Contrastive Fine-tuning")
    st.write("Fine-tune using query-document pairs for improved retrieval.")
    
    st.info("This feature requires query-document pairs with relevance labels. In a production system, this would use actual user interaction data or expert annotations.")
    
    # Generate demo contrastive data
    if st.button("🎲 Generate Demo Contrastive Data"):
        with st.spinner("Creating contrastive training pairs..."):
            # Create demo query-document pairs
            demo_pairs = [
                ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.", 1),
                ("What is machine learning?", "The weather is nice today.", 0),
                ("Explain deep learning", "Deep learning is a part of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.", 1),
                ("Explain deep learning", "I like pizza for dinner.", 0),
                ("How does AI work?", "AI systems work by processing large amounts of data, identifying patterns, and making predictions or decisions based on that analysis.", 1),
                ("How does AI work?", "The computer has a blue screen.", 0),
            ]
            
            st.session_state.contrastive_data = demo_pairs
            st.success(f"Created {len(demo_pairs)} contrastive pairs!")
    
    # Start contrastive fine-tuning
    if 'contrastive_data' in st.session_state and st.button("🚀 Start Contrastive Fine-tuning", type="primary"):
        with st.spinner("Running contrastive fine-tuning..."):
            try:
                st.session_state.fine_tuning_engine.initialize_model()
                
                results = st.session_state.fine_tuning_engine.contrastive_fine_tune(
                    st.session_state.contrastive_data, save_model=True
                )
                
                st.session_state.fine_tuning_results = results
                display_fine_tuning_results(results)
                
            except Exception as e:
                st.error(f"Contrastive fine-tuning failed: {str(e)}")


def show_hyperparameter_optimization():
    """Interface for hyperparameter optimization"""
    
    st.markdown("### 🎯 Hyperparameter Optimization")
    st.write("Automatically find the best hyperparameters for your model.")
    
    st.warning("This is computationally intensive and will test multiple parameter combinations.")
    
    # Parameter grid
    st.markdown("#### Parameter Grid")
    
    param_grid = {
        "learning_rate": [1e-5, 2e-5, 5e-5],
        "batch_size": [8, 16, 32],
        "num_epochs": [2, 3, 5]
    }
    
    st.json(param_grid)
    
    if st.button("🔍 Start Hyperparameter Search", type="primary"):
        with st.spinner("Testing parameter combinations..."):
            try:
                # Generate data if not available
                if 'training_data' not in st.session_state:
                    texts, labels = st.session_state.fine_tuning_engine.generate_synthetic_data(500)
                    st.session_state.training_data = (texts, labels)
                
                st.session_state.fine_tuning_engine.initialize_model()
                
                texts, labels = st.session_state.training_data
                best_results, best_params = st.session_state.fine_tuning_engine.optimize_hyperparameters(
                    texts, labels, param_grid
                )
                
                st.session_state.fine_tuning_results = best_results
                
                st.success("Hyperparameter optimization completed!")
                st.markdown(f"**Best Parameters**: {best_params}")
                
                display_fine_tuning_results(best_results)
                
            except Exception as e:
                st.error(f"Hyperparameter optimization failed: {str(e)}")


def display_fine_tuning_results(results):
    """Display fine-tuning results in a comprehensive way"""
    
    st.markdown("### 📊 Fine-tuning Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if results.eval_accuracy:
            st.metric("Final Accuracy", f"{results.eval_accuracy[-1]:.4f}")
    
    with col2:
        if results.eval_f1:
            st.metric("Final F1 Score", f"{results.eval_f1[-1]:.4f}")
    
    with col3:
        st.metric("Training Time", f"{results.training_time:.2f}s")
    
    with col4:
        if results.model_path:
            st.metric("Model Saved", "✅ Yes")
        else:
            st.metric("Model Saved", "❌ No")
    
    # Training curves
    if results.training_loss:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Loss', 'Accuracy', 'F1 Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        fig.add_trace(go.Scatter(y=results.training_loss, name="Train Loss", line=dict(color="blue")), row=1, col=1)
        
        # Validation loss
        fig.add_trace(go.Scatter(y=results.validation_loss, name="Val Loss", line=dict(color="red")), row=1, col=2)
        
        # Accuracy
        if results.eval_accuracy:
            fig.add_trace(go.Scatter(y=results.eval_accuracy, name="Accuracy", line=dict(color="green")), row=2, col=1)
        
        # F1 score
        if results.eval_f1:
            fig.add_trace(go.Scatter(y=results.eval_f1, name="F1 Score", line=dict(color="orange")), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True, title_text="Fine-tuning Progress")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    with st.expander("📋 Detailed Results"):
        st.json(results.final_metrics)


def show_system_config():
    """Display system configuration and settings"""
    
    st.markdown('<h2 class="sub-header">⚙️ System Configuration</h2>', unsafe_allow_html=True)
    
    # Pipeline configuration
    st.markdown("### 🤖 RAG Pipeline Settings")
    
    if st.session_state.rag_pipeline:
        # Current configuration
        config = {
            "Embedding Model": st.session_state.rag_pipeline.embedding_model_name,
            "Generation Model": st.session_state.rag_pipeline.generation_model_name,
            "Chunk Size": st.session_state.rag_pipeline.chunk_size,
            "Chunk Overlap": st.session_state.rag_pipeline.chunk_overlap,
            "Top-K Retrieval": st.session_state.rag_pipeline.top_k_retrieval,
            "Documents Loaded": len(st.session_state.rag_pipeline.documents),
            "Chunks Created": len(st.session_state.rag_pipeline.chunks),
            "Embeddings Ready": st.session_state.rag_pipeline.chunk_embeddings is not None
        }
        
        st.json(config)
        
        # Configuration options
        st.markdown("#### Adjust Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_chunk_size = st.slider("Chunk Size", 256, 2048, st.session_state.rag_pipeline.chunk_size)
            new_top_k = st.slider("Top-K Retrieval", 1, 20, st.session_state.rag_pipeline.top_k_retrieval)
        
        with col2:
            new_chunk_overlap = st.slider("Chunk Overlap", 0, 200, st.session_state.rag_pipeline.chunk_overlap)
        
        if st.button("🔄 Update Configuration"):
            st.session_state.rag_pipeline.chunk_size = new_chunk_size
            st.session_state.rag_pipeline.chunk_overlap = new_chunk_overlap
            st.session_state.rag_pipeline.top_k_retrieval = new_top_k
            
            # Recreate chunks and embeddings
            if st.session_state.rag_pipeline.documents:
                with st.spinner("Updating embeddings..."):
                    st.session_state.rag_pipeline.create_chunks()
                    st.session_state.rag_pipeline.create_embeddings()
                    st.success("Configuration updated successfully!")
        
        # System actions
        st.markdown("#### System Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save State"):
                if st.session_state.rag_pipeline:
                    save_path = "./rag_state.json"
                    st.session_state.rag_pipeline.save_state(save_path)
                    st.success(f"System state saved to {save_path}")
        
        with col2:
            if st.button("📊 Export Performance Data"):
                if st.session_state.rag_pipeline and st.session_state.rag_pipeline.query_history:
                    export_performance_data()
                else:
                    st.warning("No performance data to export")
        
        with col3:
            if st.button("🔄 Reset System"):
                st.session_state.rag_pipeline = None
                st.session_state.fine_tuning_engine = None
                st.session_state.demo_data_loaded = False
                st.success("System reset successfully!")
                st.rerun()
    
    else:
        st.info("RAG system not initialized. Go to Dashboard to initialize.")
    
    # Documentation
    st.markdown("### 📚 System Documentation")
    
    with st.expander("📖 How this system demonstrates AI engineering skills"):
        st.markdown("""
        **Generative AI Engineering Skills Demonstrated:**
        - ✅ Custom model fine-tuning and optimization
        - ✅ End-to-end ML pipeline development  
        - ✅ Performance monitoring and analytics
        - ✅ Production-ready system architecture
        - ✅ Cost optimization and scalability considerations
        
        **Prompt Engineering Skills Demonstrated:**
        - ✅ Intelligent prompt optimization
        - ✅ Context-aware response generation
        - ✅ Multi-modal retrieval strategies
        - ✅ Quality evaluation and metrics
        - ✅ User experience optimization
        
        **Technical Implementation:**
        - ✅ Modern ML frameworks (PyTorch, Transformers, SentenceTransformers)
        - ✅ Vector databases and similarity search
        - ✅ Real-time web interface with Streamlit
        - ✅ Comprehensive testing and evaluation
        - ✅ Model deployment and serving
        """)
    
    with st.expander("🔬 Research and Evaluation Methods"):
        st.markdown("""
        **Evaluation Metrics Used:**
        - Precision, Recall, F1-score for retrieval quality
        - NDCG (Normalized Discounted Cumulative Gain)
        - MRR (Mean Reciprocal Rank)
        - Latency and throughput analysis
        - Cost-effectiveness analysis
        
        **Fine-tuning Approaches:**
        - Supervised learning with custom datasets
        - Contrastive learning for improved retrieval
        - Hyperparameter optimization with grid search
        - Cross-validation and model selection
        """)


def export_performance_data():
    """Export performance data for analysis"""
    
    try:
        # Create comprehensive performance report
        report = {
            "system_info": {
                "pipeline_config": {
                    "embedding_model": st.session_state.rag_pipeline.embedding_model_name,
                    "chunk_size": st.session_state.rag_pipeline.chunk_size,
                    "top_k": st.session_state.rag_pipeline.top_k_retrieval
                },
                "data_info": {
                    "documents_loaded": len(st.session_state.rag_pipeline.documents),
                    "chunks_created": len(st.session_state.rag_pipeline.chunks)
                }
            },
            "performance_metrics": st.session_state.rag_pipeline.performance_metrics,
            "query_history": [
                {
                    "query": q.query,
                    "response_preview": q.response[:200],
                    "retrieval_score": q.retrieval_score,
                    "generation_time": q.generation_time,
                    "cost": q.cost,
                    "sources_count": len(q.sources)
                }
                for q in st.session_state.rag_pipeline.query_history
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert to JSON and provide download
        report_json = json.dumps(report, indent=2)
        
        st.download_button(
            label="💾 Download Performance Report",
            data=report_json,
            file_name=f"rag_performance_report_{int(time.time())}.json",
            mime="application/json"
        )
        
        st.success("Performance report generated!")
        
    except Exception as e:
        st.error(f"Failed to export performance data: {str(e)}")


if __name__ == "__main__":
    main()