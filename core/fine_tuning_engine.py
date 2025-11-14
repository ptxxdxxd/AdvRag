"""
Custom Fine-tuning Engine for RAG System
Demonstrates generative AI engineering skills with model training and optimization
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel, AutoTokenizer, get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.util import cos_sim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_seq_length: int = 512
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./models/fine_tuned"
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"


@dataclass
class FineTuningResults:
    """Results from fine-tuning process"""
    training_loss: List[float]
    validation_loss: List[float]
    eval_accuracy: List[float]
    eval_f1: List[float]
    learning_rates: List[float]
    final_metrics: Dict[str, float]
    training_time: float
    model_path: str


class CustomDataset(Dataset):
    """Custom dataset for fine-tuning"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning"""
    
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], tokenizer):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]
        
        encoding1 = self.tokenizer(
            text1, truncation=True, padding='max_length', max_length=512, return_tensors='pt'
        )
        encoding2 = self.tokenizer(
            text2, truncation=True, padding='max_length', max_length=512, return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': encoding1['input_ids'].flatten(),
            'anchor_attention_mask': encoding1['attention_mask'].flatten(),
            'positive_input_ids': encoding2['input_ids'].flatten(),
            'positive_attention_mask': encoding2['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class FineTuningEngine:
    """
    Advanced fine-tuning engine for custom model training
    Supports multiple training strategies and optimization techniques
    """
    
    def __init__(self, base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize fine-tuning engine"""
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.training_config = TrainingConfig()
        self.training_history = []
        self.current_epoch = 0
        
        logger.info(f"Initialized FineTuningEngine with base model: {base_model_name}")
    
    def initialize_model(self, model_path: Optional[str] = None):
        """Initialize the model for fine-tuning"""
        try:
            if model_path:
                self.model = SentenceTransformer(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"Loaded model from {model_path}")
            else:
                self.model = SentenceTransformer(self.base_model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                logger.info(f"Loaded base model: {self.base_model_name}")
            
            # Set max sequence length
            if hasattr(self.tokenizer, 'model_max_length'):
                self.tokenizer.model_max_length = self.training_config.max_seq_length
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def prepare_embedding_training_data(
        self, 
        documents: List[str], 
        queries: List[str] = None, 
        relevant_docs: List[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Prepare training data for embedding fine-tuning
        
        Args:
            documents: List of document texts
            queries: Optional list of queries for contrastive learning
            relevant_docs: Optional list of relevant document indices for each query
            
        Returns:
            Tuple of (texts, labels) for supervised fine-tuning
        """
        try:
            training_texts = []
            training_labels = []
            
            # Method 1: Simple classification based on document length/quality
            for i, doc in enumerate(documents):
                # Simple heuristic: longer documents are "high quality"
                quality_score = len(doc) / 1000  # Normalize by 1000 characters
                
                # Create binary classification: high quality (>0.5) vs low quality (≤0.5)
                label = 1 if quality_score > 0.5 else 0
                
                training_texts.append(doc)
                training_labels.append(label)
            
            # Method 2: Query-document pairs for contrastive learning
            if queries and relevant_docs:
                for query, relevant_indices in zip(queries, relevant_docs):
                    for idx in relevant_indices:
                        if idx < len(documents):
                            training_texts.append(documents[idx])
                            training_labels.append(1)  # Relevant
                    
                    # Add some non-relevant examples
                    for _ in range(min(3, len(documents) - len(relevant_indices))):
                        # Random non-relevant document
                        non_relevant_idx = np.random.choice(
                            [i for i in range(len(documents)) if i not in relevant_indices],
                            1
                        )[0]
                        
                        training_texts.append(documents[non_relevant_idx])
                        training_labels.append(0)  # Not relevant
            
            logger.info(f"Prepared {len(training_texts)} training examples")
            return training_texts, training_labels
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            return [], []
    
    def fine_tune_embeddings(
        self,
        training_texts: List[str],
        training_labels: List[int],
        validation_split: float = 0.2,
        save_model: bool = True
    ) -> FineTuningResults:
        """
        Fine-tune the embedding model using supervised learning
        
        Args:
            training_texts: List of training texts
            training_labels: List of corresponding labels
            validation_split: Fraction of data to use for validation
            save_model: Whether to save the fine-tuned model
            
        Returns:
            FineTuningResults with training metrics
        """
        start_time = time.time()
        
        try:
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                training_texts, training_labels, 
                test_size=validation_split, 
                stratify=training_labels,
                random_state=42
            )
            
            logger.info(f"Training on {len(train_texts)} examples, validating on {len(val_texts)}")
            
            # Create datasets
            train_dataset = CustomDataset(train_texts, train_labels, self.tokenizer, 
                                        self.training_config.max_seq_length)
            val_dataset = CustomDataset(val_texts, val_labels, self.tokenizer, 
                                      self.training_config.max_seq_length)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=self.training_config.output_dir,
                num_train_epochs=self.training_config.num_epochs,
                per_device_train_batch_size=self.training_config.batch_size,
                per_device_eval_batch_size=self.training_config.batch_size,
                warmup_steps=self.training_config.warmup_steps,
                weight_decay=self.training_config.weight_decay,
                logging_dir=f"{self.training_config.output_dir}/logs",
                logging_steps=self.training_config.logging_steps,
                evaluation_strategy=self.training_config.eval_strategy,
                eval_steps=self.training_config.eval_steps,
                save_strategy="steps",
                save_steps=self.training_config.save_steps,
                load_best_model_at_end=self.training_config.load_best_model_at_end,
                metric_for_best_model=self.training_config.metric_for_best_model,
                greater_is_better=True,
                learning_rate=self.training_config.learning_rate,
                report_to=[],  # Disable wandb
            )
            
            # Define compute metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return {
                    'accuracy': accuracy_score(labels, predictions),
                    'f1': f1_score(labels, predictions, average='weighted'),
                }
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )
            
            # Train the model
            logger.info("Starting fine-tuning...")
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            
            training_time = time.time() - start_time
            
            # Save model if requested
            model_path = None
            if save_model:
                output_dir = Path(self.training_config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(str(output_dir))
                model_path = str(output_dir)
                logger.info(f"Model saved to {model_path}")
            
            # Create results object
            results = FineTuningResults(
                training_loss=trainer.state.log_history["train_loss"] if "train_loss" in str(trainer.state.log_history) else [],
                validation_loss=[eval_results.get("eval_loss", 0)],
                eval_accuracy=[eval_results.get("eval_accuracy", 0)],
                eval_f1=[eval_results.get("eval_f1", 0)],
                learning_rates=[self.training_config.learning_rate],
                final_metrics=eval_results,
                training_time=training_time,
                model_path=model_path or ""
            )
            
            logger.info(f"Fine-tuning completed in {training_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            raise
    
    def contrastive_fine_tune(
        self,
        query_document_pairs: List[Tuple[str, str, int]],
        save_model: bool = True
    ) -> FineTuningResults:
        """
        Fine-tune using contrastive learning (query-document pairs)
        
        Args:
            query_document_pairs: List of (query, document, label) where label=1 for relevant
            save_model: Whether to save the fine-tuned model
            
        Returns:
            FineTuningResults with training metrics
        """
        start_time = time.time()
        
        try:
            # Separate pairs and labels
            pairs = [(pair[0], pair[1]) for pair in query_document_pairs]
            labels = [pair[2] for pair in query_document_pairs]
            
            # Split data
            train_pairs, val_pairs, train_labels, val_labels = train_test_split(
                pairs, labels, test_size=0.2, stratify=labels, random_state=42
            )
            
            logger.info(f"Contrastive training on {len(train_pairs)} pairs")
            
            # Create contrastive dataset
            train_dataset = ContrastiveDataset(train_pairs, train_labels, self.tokenizer)
            val_dataset = ContrastiveDataset(val_pairs, val_labels, self.tokenizer)
            
            # Create DataLoader for contrastive learning
            train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.training_config.batch_size, shuffle=False)
            
            # Training loop for contrastive learning
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            optimizer = AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
            criterion = nn.BCELoss()
            
            training_history = {
                "train_loss": [],
                "val_loss": [],
                "accuracy": [],
                "f1": []
            }
            
            for epoch in range(self.training_config.num_epochs):
                self.model.train()
                epoch_loss = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    # Get embeddings
                    anchor_emb = self.model.encode(batch['anchor_input_ids'], 
                                                 convert_to_numpy=False)
                    positive_emb = self.model.encode(batch['positive_input_ids'], 
                                                   convert_to_numpy=False)
                    
                    # Calculate contrastive loss
                    similarity = torch.mm(anchor_emb, positive_emb.t())
                    similarity = torch.sigmoid(similarity)
                    
                    loss = criterion(similarity.squeeze(), batch['labels'])
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        anchor_emb = self.model.encode(batch['anchor_input_ids'], 
                                                     convert_to_numpy=False)
                        positive_emb = self.model.encode(batch['positive_input_ids'], 
                                                       convert_to_numpy=False)
                        
                        similarity = torch.mm(anchor_emb, positive_emb.t())
                        similarity = torch.sigmoid(similarity)
                        
                        loss = criterion(similarity.squeeze(), batch['labels'])
                        val_loss += loss.item()
                        
                        val_predictions.extend((similarity.squeeze() > 0.5).cpu().numpy())
                        val_targets.extend(batch['labels'].cpu().numpy())
                
                # Calculate metrics
                train_loss = epoch_loss / len(train_loader)
                val_loss = val_loss / len(val_loader)
                accuracy = accuracy_score(val_targets, val_predictions)
                f1 = f1_score(val_targets, val_predictions)
                
                training_history["train_loss"].append(train_loss)
                training_history["val_loss"].append(val_loss)
                training_history["accuracy"].append(accuracy)
                training_history["f1"].append(f1)
                
                logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            training_time = time.time() - start_time
            
            # Save model if requested
            model_path = None
            if save_model:
                output_dir = Path(self.training_config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(str(output_dir))
                model_path = str(output_dir)
            
            results = FineTuningResults(
                training_loss=training_history["train_loss"],
                validation_loss=training_history["val_loss"],
                eval_accuracy=training_history["accuracy"],
                eval_f1=training_history["f1"],
                learning_rates=[self.training_config.learning_rate] * len(training_history["train_loss"]),
                final_metrics={
                    "final_accuracy": accuracy,
                    "final_f1": f1,
                    "final_train_loss": training_history["train_loss"][-1],
                    "final_val_loss": training_history["val_loss"][-1]
                },
                training_time=training_time,
                model_path=model_path or ""
            )
            
            logger.info(f"Contrastive fine-tuning completed in {training_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Contrastive fine-tuning failed: {str(e)}")
            raise
    
    def optimize_hyperparameters(
        self,
        training_texts: List[str],
        training_labels: List[int],
        param_grid: Dict[str, List[Any]]
    ) -> Tuple[FineTuningResults, Dict[str, Any]]:
        """
        Hyperparameter optimization using grid search
        
        Args:
            training_texts: Training texts
            training_labels: Training labels
            param_grid: Dictionary of hyperparameters to optimize
            
        Returns:
            Best results and best parameters
        """
        try:
            from itertools import product
            
            best_score = 0
            best_params = {}
            best_results = None
            
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())
            
            logger.info(f"Testing {len(param_combinations)} parameter combinations")
            
            for i, param_combo in enumerate(param_combinations):
                # Set parameters
                params = dict(zip(param_names, param_combo))
                logger.info(f"Testing combination {i + 1}/{len(param_combinations)}: {params}")
                
                # Update training config
                self.training_config.learning_rate = params.get("learning_rate", 2e-5)
                self.training_config.batch_size = params.get("batch_size", 16)
                self.training_config.num_epochs = params.get("num_epochs", 3)
                
                # Run fine-tuning
                results = self.fine_tune_embeddings(training_texts, training_labels, 
                                                  validation_split=0.2, save_model=False)
                
                # Check score
                current_score = results.final_metrics.get("eval_accuracy", 0)
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = params
                    best_results = results
                
                logger.info(f"Score: {current_score:.4f}, Best: {best_score:.4f}")
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {best_score:.4f}")
            
            return best_results, best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise
    
    def compare_models(
        self,
        test_texts: List[str],
        test_labels: List[int],
        model_paths: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple fine-tuned models on test data
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            model_paths: List of paths to different models
            
        Returns:
            DataFrame with comparison results
        """
        try:
            results = []
            
            for model_path in model_paths:
                logger.info(f"Evaluating model: {model_path}")
                
                # Load model
                test_model = SentenceTransformer(model_path)
                test_tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Create test dataset
                test_dataset = CustomDataset(test_texts, test_labels, test_tokenizer)
                
                # Set up temporary model for evaluation
                self.model = test_model
                self.tokenizer = test_tokenizer
                
                # Evaluate (simplified evaluation)
                accuracy, f1 = self._evaluate_model(test_dataset)
                
                results.append({
                    "model_path": model_path,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "model_size": os.path.getsize(f"{model_path}/config.json") if os.path.exists(f"{model_path}/config.json") else 0
                })
            
            results_df = pd.DataFrame(results)
            logger.info("Model comparison completed")
            return results_df
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            raise
    
    def _evaluate_model(self, test_dataset) -> Tuple[float, float]:
        """Simplified model evaluation"""
        # This is a simplified evaluation - in practice, you'd use proper validation
        test_loader = DataLoader(test_dataset, batch_size=self.training_config.batch_size)
        
        all_predictions = []
        all_labels = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                # Get embeddings and predict
                embeddings = self.model.encode(batch['input_ids'], convert_to_numpy=False)
                
                # Simple classification (this would be replaced with actual classifier)
                predictions = torch.sigmoid(embeddings.mean(dim=1) * 1000).round()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return accuracy, f1
    
    def plot_training_history(self, results: FineTuningResults, save_path: str = None):
        """Plot training history"""
        plt.figure(figsize=(15, 10))
        
        # Training loss
        plt.subplot(2, 2, 1)
        plt.plot(results.training_loss, label='Training Loss')
        plt.plot(results.validation_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(results.eval_accuracy, label='Validation Accuracy', marker='o')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # F1 Score
        plt.subplot(2, 2, 3)
        plt.plot(results.eval_f1, label='Validation F1', marker='s')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Learning Rate
        plt.subplot(2, 2, 4)
        plt.plot(results.learning_rates, label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plot saved to {save_path}")
        
        plt.show()
    
    def generate_synthetic_data(self, num_samples: int = 1000, domain: str = "general") -> Tuple[List[str], List[int]]:
        """
        Generate synthetic training data for fine-tuning
        
        Args:
            num_samples: Number of samples to generate
            domain: Domain for synthetic data ("general", "technical", "medical", "legal")
            
        Returns:
            Tuple of (texts, labels)
        """
        synthetic_texts = []
        synthetic_labels = []
        
        # Define templates for different domains
        domain_templates = {
            "general": {
                "high_quality": [
                    "The comprehensive analysis demonstrates the fundamental principles governing the subject matter.",
                    "In depth exploration reveals critical insights and nuanced perspectives.",
                    "The detailed investigation provides substantial evidence supporting the conclusions.",
                    "Thorough examination uncovers significant correlations and patterns."
                ],
                "low_quality": [
                    "This is about the topic.",
                    "Here's something related.",
                    "This mentions the subject.",
                    "This refers to the thing."
                ]
            },
            "technical": {
                "high_quality": [
                    "The implementation leverages advanced algorithms to optimize performance and scalability.",
                    "The architecture design incorporates microservices patterns and container orchestration.",
                    "The system employs distributed computing paradigms with fault tolerance mechanisms.",
                    "The algorithm implements efficient data structures and algorithmic complexity optimizations."
                ],
                "low_quality": [
                    "This uses code to do stuff.",
                    "The program works somehow.",
                    "It has some features.",
                    "This makes the thing work."
                ]
            }
        }
        
        templates = domain_templates.get(domain, domain_templates["general"])
        
        for i in range(num_samples):
            # Randomly choose quality level (60% high quality, 40% low quality)
            is_high_quality = np.random.random() < 0.6
            
            if is_high_quality:
                template = np.random.choice(templates["high_quality"])
                label = 1
            else:
                template = np.random.choice(templates["low_quality"])
                label = 0
            
            # Add some variation
            if np.random.random() < 0.3:
                template += f" Additional context regarding {domain} matters."
            
            synthetic_texts.append(template)
            synthetic_labels.append(label)
        
        logger.info(f"Generated {num_samples} synthetic samples for domain: {domain}")
        return synthetic_texts, synthetic_labels