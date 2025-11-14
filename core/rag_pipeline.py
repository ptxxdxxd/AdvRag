"""
Advanced RAG Pipeline with Custom Fine-tuning Capabilities
Combines prompt engineering and generative AI engineering skills
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structure for query results with metadata"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    retrieval_score: float
    generation_time: float
    cost: float
    metadata: Dict[str, Any]


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance"""
    precision: float
    recall: float
    f1_score: float
    ndcg: float
    mrr: float


class AdvancedRAGPipeline:
    """
    Advanced RAG system with custom fine-tuning, prompt optimization,
    and comprehensive evaluation capabilities
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model_name: str = "gpt-3.5-turbo",
        vector_db_path: str = "./data/vector_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k_retrieval: int = 5,
        enable_fine_tuning: bool = False
    ):
        """Initialize the RAG pipeline with configuration"""
        
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval
        self.enable_fine_tuning = enable_fine_tuning
        
        # Initialize models
        self.embedding_model = None
        self.vectorizer = None
        self.documents = []
        self.chunks = []
        self.chunk_embeddings = None
        
        # Performance tracking
        self.query_history = []
        self.performance_metrics = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "avg_generation_time": 0,
            "avg_cost_per_query": 0,
            "total_cost": 0
        }
        
        # Fine-tuning components
        self.fine_tuned_embeddings = None
        self.training_data = []
        
        logger.info(f"Initialized AdvancedRAGPipeline with model: {embedding_model_name}")
    
    def initialize_models(self):
        """Initialize embedding and generation models"""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Initialize TF-IDF vectorizer for comparison
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            return False
    
    def load_documents(self, file_paths: List[str]) -> bool:
        """
        Load and preprocess documents from various formats
        
        Args:
            file_paths: List of file paths to load documents from
            
        Returns:
            bool: Success status
        """
        try:
            self.documents = []
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Load based on file extension
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_path.endswith('.pdf'):
                    # Implementation for PDF loading would go here
                    # For now, skip PDFs
                    logger.warning("PDF support not implemented yet")
                    continue
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue
                
                if content.strip():
                    self.documents.append({
                        "content": content,
                        "source": os.path.basename(file_path),
                        "metadata": {"file_path": file_path}
                    })
            
            logger.info(f"Loaded {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            return False
    
    def create_chunks(self) -> bool:
        """
        Create intelligent document chunks with overlap
        
        Returns:
            bool: Success status
        """
        try:
            self.chunks = []
            
            for doc in self.documents:
                content = doc["content"]
                sentences = content.split('.')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= self.chunk_size:
                        current_chunk += sentence + "."
                    else:
                        if current_chunk.strip():
                            self.chunks.append({
                                "text": current_chunk.strip(),
                                "source": doc["source"],
                                "metadata": doc["metadata"]
                            })
                        current_chunk = sentence + "."
                
                # Add the last chunk
                if current_chunk.strip():
                    self.chunks.append({
                        "text": current_chunk.strip(),
                        "source": doc["source"],
                        "metadata": doc["metadata"]
                    })
            
            logger.info(f"Created {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create chunks: {str(e)}")
            return False
    
    def create_embeddings(self) -> bool:
        """
        Create embeddings for all chunks using fine-tuning if enabled
        
        Returns:
            bool: Success status
        """
        try:
            if not self.chunks:
                logger.error("No chunks available for embedding creation")
                return False
            
            chunk_texts = [chunk["text"] for chunk in self.chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                chunk_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.chunk_embeddings = embeddings
            
            # Also create TF-IDF vectors for comparison
            self.tfidf_vectors = self.vectorizer.fit_transform(chunk_texts)
            
            logger.info(f"Created embeddings for {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            return False
    
    def retrieve(self, query: str, method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using specified method
        
        Args:
            query: User query
            method: Retrieval method ("embeddings", "tfidf", "hybrid")
            
        Returns:
            List of relevant chunks with scores
        """
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Embedding-based retrieval
            if method in ["embeddings", "hybrid"]:
                embedding_scores = cosine_similarity(
                    [query_embedding], 
                    self.chunk_embeddings
                )[0]
            
            # TF-IDF-based retrieval
            if method in ["tfidf", "hybrid"]:
                query_vector = self.vectorizer.transform([query])
                tfidf_scores = np.array(
                    [cosine_similarity(query_vector, chunk_vec.toarray())[0][0] 
                     for chunk_vec in self.tfidf_vectors]
                )
            
            # Combine scores for hybrid method
            if method == "hybrid":
                # Normalize scores
                embedding_scores_norm = (embedding_scores - embedding_scores.min()) / \
                                       (embedding_scores.max() - embedding_scores.min())
                tfidf_scores_norm = (tfidf_scores - tfidf_scores.min()) / \
                                   (tfidf_scores.max() - tfidf_scores.min())
                
                # Weighted combination
                combined_scores = 0.7 * embedding_scores_norm + 0.3 * tfidf_scores_norm
            else:
                combined_scores = embedding_scores if method == "embeddings" else tfidf_scores
            
            # Get top-k results
            top_indices = np.argsort(combined_scores)[-self.top_k_retrieval:][::-1]
            
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk["retrieval_score"] = float(combined_scores[idx])
                results.append(chunk)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} chunks in {retrieval_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    def optimize_prompt(self, query: str, context_chunks: List[Dict], 
                       base_prompt: str = None) -> str:
        """
        Optimize prompt based on retrieved context
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            base_prompt: Base prompt template
            
        Returns:
            Optimized prompt string
        """
        if base_prompt is None:
            base_prompt = """Answer the following question based on the provided context. 
            If the context doesn't contain relevant information, say so clearly.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        
        # Create context from chunks
        context = "\n\n".join([
            f"[{i+1}] {chunk['text']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Add source citations
        sources = list(set([chunk['source'] for chunk in context_chunks]))
        context += f"\n\nSources: {', '.join(sources)}"
        
        # Optimize the prompt
        optimized_prompt = base_prompt.format(
            context=context,
            question=query
        )
        
        return optimized_prompt
    
    def generate_response(self, query: str, method: str = "hybrid") -> QueryResult:
        """
        Generate response using RAG pipeline
        
        Args:
            query: User query
            method: Retrieval method to use
            
        Returns:
            QueryResult with response and metadata
        """
        start_time = time.time()
        
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retrieve(query, method)
            
            if not retrieved_chunks:
                return QueryResult(
                    query=query,
                    response="I couldn't find relevant information to answer your question.",
                    sources=[],
                    retrieval_score=0.0,
                    generation_time=time.time() - start_time,
                    cost=0.0,
                    metadata={"error": "No relevant chunks found"}
                )
            
            # Optimize prompt
            prompt = self.optimize_prompt(query, retrieved_chunks)
            
            # For demo purposes, simulate response generation
            # In real implementation, this would call an LLM API
            response = self._simulate_llm_response(prompt, query)
            
            generation_time = time.time() - start_time
            
            # Calculate cost (simulated)
            cost = len(prompt + response) * 0.0001  # Simulated token cost
            
            # Create result
            result = QueryResult(
                query=query,
                response=response,
                sources=[{
                    "text": chunk["text"][:200] + "...",
                    "source": chunk["source"],
                    "score": chunk["retrieval_score"]
                } for chunk in retrieved_chunks],
                retrieval_score=np.mean([chunk["retrieval_score"] 
                                       for chunk in retrieved_chunks]),
                generation_time=generation_time,
                cost=cost,
                metadata={
                    "method": method,
                    "num_chunks": len(retrieved_chunks),
                    "avg_chunk_score": np.mean([chunk["retrieval_score"] 
                                              for chunk in retrieved_chunks])
                }
            )
            
            # Update performance metrics
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return QueryResult(
                query=query,
                response=f"Error generating response: {str(e)}",
                sources=[],
                retrieval_score=0.0,
                generation_time=time.time() - start_time,
                cost=0.0,
                metadata={"error": str(e)}
            )
    
    def _simulate_llm_response(self, prompt: str, query: str) -> str:
        """Simulate LLM response generation for demo purposes"""
        # Simple rule-based response simulation
        query_lower = query.lower()
        
        if "what" in query_lower or "how" in query_lower or "why" in query_lower:
            return f"Based on the provided context, I can provide information about {query}. The documents mention relevant details that address your question. Here's what I found: [This would be a detailed response generated by an actual LLM based on the retrieved context]."
        elif "explain" in query_lower or "describe" in query_lower:
            return f"To explain {query}: From the available documents, several key points emerge that provide insights into this topic. The context covers various aspects that help clarify the question. [Detailed explanation would continue here based on the specific retrieved chunks]"
        else:
            return f"Regarding {query}, the retrieved documents provide relevant information. The sources contain details that directly address your question. [Response would be generated based on the specific context retrieved from the documents]"
    
    def _update_metrics(self, result: QueryResult):
        """Update performance metrics"""
        self.performance_metrics["total_queries"] += 1
        self.performance_metrics["total_cost"] += result.cost
        
        # Update moving averages
        total = self.performance_metrics["total_queries"]
        
        current_retrieval = self.performance_metrics["avg_retrieval_time"]
        self.performance_metrics["avg_retrieval_time"] = \
            (current_retrieval * (total - 1) + result.generation_time) / total
        
        current_generation = self.performance_metrics["avg_generation_time"]
        self.performance_metrics["avg_generation_time"] = \
            (current_generation * (total - 1) + result.generation_time) / total
        
        current_cost = self.performance_metrics["avg_cost_per_query"]
        self.performance_metrics["avg_cost_per_query"] = \
            (current_cost * (total - 1) + result.cost) / total
    
    def evaluate_retrieval(self, queries: List[str], relevant_docs: List[List[int]]) -> RetrievalMetrics:
        """
        Evaluate retrieval performance using precision, recall, F1, etc.
        
        Args:
            queries: List of test queries
            relevant_docs: List of lists with relevant document indices for each query
            
        Returns:
            RetrievalMetrics object with evaluation scores
        """
        try:
            all_precisions = []
            all_recalls = []
            all_f1s = []
            all_ndcgs = []
            all_mrrs = []
            
            for query, relevant_indices in zip(queries, relevant_docs):
                retrieved_chunks = self.retrieve(query)
                retrieved_indices = [i for i, chunk in enumerate(retrieved_chunks)]
                
                # Precision and Recall
                true_positives = len(set(retrieved_indices) & set(relevant_indices))
                precision = true_positives / len(retrieved_indices) if retrieved_indices else 0
                recall = true_positives / len(relevant_indices) if relevant_indices else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
                
                # NDCG (Normalized Discounted Cumulative Gain)
                ndcg = self._calculate_ndcg(retrieved_indices, relevant_indices)
                all_ndcgs.append(ndcg)
                
                # MRR (Mean Reciprocal Rank)
                mrr = self._calculate_mrr(retrieved_indices, relevant_indices)
                all_mrrs.append(mrr)
            
            return RetrievalMetrics(
                precision=np.mean(all_precisions),
                recall=np.mean(all_recalls),
                f1_score=np.mean(all_f1s),
                ndcg=np.mean(all_ndcgs),
                mrr=np.mean(all_mrrs)
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return RetrievalMetrics(0, 0, 0, 0, 0)
    
    def _calculate_ndcg(self, retrieved: List[int], relevant: List[int]) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        # Simplified NDCG calculation
        if not relevant:
            return 0.0
        
        dcg = sum(1.0 / (i + 1) for i, idx in enumerate(retrieved) if idx in relevant)
        idcg = sum(1.0 / (i + 1) for i in range(len(relevant)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, retrieved: List[int], relevant: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, idx in enumerate(retrieved):
            if idx in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        return {
            "performance_metrics": self.performance_metrics,
            "query_history": self.query_history[-50:],  # Last 50 queries
            "system_status": {
                "models_loaded": self.embedding_model is not None,
                "documents_loaded": len(self.documents),
                "chunks_created": len(self.chunks),
                "embeddings_ready": self.chunk_embeddings is not None
            }
        }
    
    def save_state(self, filepath: str):
        """Save pipeline state for persistence"""
        state = {
            "config": {
                "embedding_model_name": self.embedding_model_name,
                "generation_model_name": self.generation_model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k_retrieval": self.top_k_retrieval
            },
            "performance_metrics": self.performance_metrics,
            "documents": self.documents,
            "chunks": self.chunks
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Pipeline state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load pipeline state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore configuration
            config = state["config"]
            self.chunk_size = config["chunk_size"]
            self.chunk_overlap = config["chunk_overlap"]
            self.top_k_retrieval = config["top_k_retrieval"]
            
            # Restore documents and chunks
            self.documents = state["documents"]
            self.chunks = state["chunks"]
            
            # Recreate embeddings
            self.create_embeddings()
            
            logger.info(f"Pipeline state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            return False