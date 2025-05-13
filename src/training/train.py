"""
Training Script for SocialLLM

This script provides a command-line interface for training the SocialLLM model
on social media data.
"""

import os
import argparse
import logging
import torch
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer

from src.models.social_llm.model import SocialLLM, SocialLLMConfig
from src.data.data_collector import SocialMediaDataCollector, SocialMediaProcessor
from src.training.trainer import SocialLLMTrainer, SocialLLMTrainingArguments

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parses command-line arguments for training a SocialLLM model."""
    parser = argparse.ArgumentParser(description="Train SocialLLM model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with data files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for model and tokenizer")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Pretrained model name or path")
    parser.add_argument("--vocab_size", type=int, default=50265, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads")
    
    # Social media specific arguments
    parser.add_argument("--num_hashtag_types", type=int, default=1000, help="Number of hashtag types")
    parser.add_argument("--num_emoji_types", type=int, default=500, help="Number of emoji types")
    parser.add_argument("--use_curriculum_learning", action="store_true", help="Use curriculum learning")
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint saving steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="social-llm", help="W&B project name")
    
    return parser.parse_args()


def main():
    # Parse arguments
    """Main function to train a SocialLLM model using provided arguments and
    configurations.
    
    The function handles the entire pipeline from data preparation, model
    initialization, training, evaluation, and saving the final model and metrics.
    It also supports resuming training from checkpoints and handles interruptions
    gracefully.
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Initialize tokenizer
    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # Use a default tokenizer if no model path provided
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model configuration
    config = SocialLLMConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_hashtag_types=args.num_hashtag_types,
        num_emoji_types=args.num_emoji_types
    )
    
    # Initialize model
    logger.info("Initializing SocialLLM model")
    model = SocialLLM(config)
    
    # Load pretrained weights if provided
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        logger.info(f"Loading pretrained weights from {args.model_name_or_path}")
        model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "model.pt")))
    
    # Data collection and processing
    logger.info("Loading and processing data")
    processor = SocialMediaProcessor(
        max_length=config.max_position_embeddings,
        hashtag_min_freq=5,
        emoji_min_freq=5,
        mention_min_freq=10
    )
    
    # Initialize data collector
    collector = SocialMediaDataCollector(
        output_dir=os.path.join(args.output_dir, "processed_data"),
        processor=processor,
        apply_augmentation=True
    )
    
    # Load data from CSV or JSONL files in data directory
    data_files = []
    for file in os.listdir(args.data_dir):
        if file.endswith('.csv'):
            data_files.append(os.path.join(args.data_dir, file))
        elif file.endswith('.jsonl'):
            data_files.append(os.path.join(args.data_dir, file))
    
    # Check if data files were found
    if not data_files:
        logger.error(f"No CSV or JSONL files found in {args.data_dir}")
        return
    
    # Process each data file
    for file_path in data_files:
        if file_path.endswith('.csv'):
            logger.info(f"Processing CSV file: {file_path}")
            collector.collect_from_csv(
                file_path=file_path,
                text_column="text",
                id_column="post_id",
                user_column="user_id",
                timestamp_column="timestamp",
                likes_column="likes",
                shares_column="shares",
                comments_column="comments",
                media_column="has_media",
                language_column="language"
            )
        elif file_path.endswith('.jsonl'):
            logger.info(f"Processing JSONL file: {file_path}")
            collector.collect_from_jsonl(file_path=file_path)
    
    # Clean and validate data
    collector.validate_and_clean()
    logger.info(f"Collected {len(collector.posts)} posts after cleaning")
    
    # Save processed data for future use
    processed_data_path = os.path.join(args.output_dir, "processed_data", "processed_posts.json")
    collector.save_processed_data(processed_data_path)
    
    # Prepare datasets for each task
    tasks = [
        "generation",               # Language modeling
        "sentiment_analysis",       # Sentiment classification
        "engagement_prediction",    # Predict engagement metrics
        "hashtag_suggestion",       # Suggest relevant hashtags
        "viral_potential",          # Predict viral score
        "topic_classification"      # Classify post topics
    ]
    
    # Prepare datasets for training and evaluation
    train_datasets = {}
    eval_datasets = {}
    
    for task in tasks:
        logger.info(f"Preparing dataset for task: {task}")
        try:
            task_datasets = collector.prepare_dataset(
                tokenizer=tokenizer,
                task=task,
                test_size=0.1,
                valid_size=0.1,
                seed=args.seed
            )
            
            # Only add datasets if they have samples
            if len(task_datasets["train"]) > 0:
                train_datasets[task] = task_datasets["train"]
                eval_datasets[task] = task_datasets["validation"]
                
                # Log dataset sizes
                logger.info(f"Task {task}: {len(task_datasets['train'])} training samples, "
                          f"{len(task_datasets['validation'])} validation samples")
            else:
                logger.warning(f"Task {task}: No valid samples available, skipping")
                
        except Exception as e:
            logger.warning(f"Error preparing dataset for task {task}: {str(e)}")
            continue
    
    # Create training arguments
    training_args = SocialLLMTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_curriculum_learning=args.use_curriculum_learning,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        fp16=args.fp16,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Initialize trainer
    trainer = SocialLLMTrainer(
        model=model,
        args=training_args,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer
    )
    
    # Check for existing checkpoints
    checkpoints = []
    if os.path.exists(args.output_dir):
        for dirname in os.listdir(args.output_dir):
            if dirname.startswith("checkpoint-"):
                checkpoints.append(os.path.join(args.output_dir, dirname))
    
    # Resume from latest checkpoint if available
    if checkpoints:
        # Sort by step number
        def get_step(checkpoint_dir):
            """Extracts and returns the step number from a checkpoint directory name."""
            dir_name = os.path.basename(checkpoint_dir)
            try:
                step = int(dir_name.split("-")[1].split("-")[0])
                return step
            except (IndexError, ValueError):
                return 0
        
        checkpoints.sort(key=get_step, reverse=True)
        latest_checkpoint = checkpoints[0]
        logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.load_checkpoint(latest_checkpoint)
    
    # Start training
    logger.info("Starting training")
    try:
        # Train the model
        metrics = trainer.train()
        
        # Log final metrics
        logger.info("Training completed")
        logger.info("Final metrics:")
        for task, task_metrics in metrics.items():
            for k, v in task_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
        
        # Save the final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        # Save model weights
        torch.save(model.state_dict(), os.path.join(final_model_path, "model.pt"))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save tokenizer
        if tokenizer:
            tokenizer.save_pretrained(final_model_path)
        
        # Save model config
        config_path = os.path.join(final_model_path, "config.json")
        with open(config_path, "w") as f:
            import json
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
            json.dump(config_dict, f, indent=2)
        
        # Run final evaluation on test set
        logger.info("Running final evaluation on test set")
        test_datasets = {task: task_datasets["test"] for task, task_datasets in collector.prepare_dataset(
            tokenizer=tokenizer,
            test_size=0.1,
            valid_size=0.1,
            seed=args.seed
        ).items()}
        
        # Create evaluation trainer with test datasets
        eval_trainer = SocialLLMTrainer(
            model=model,
            args=training_args,
            train_datasets={},  # No training, just evaluation
            eval_datasets=test_datasets,
            tokenizer=tokenizer
        )
        
        # Run evaluation on each task
        test_metrics = {}
        for task in test_datasets:
            logger.info(f"Evaluating task: {task}")
            task_metrics = eval_trainer.evaluate()
            test_metrics[task] = task_metrics
            
            # Log metrics
            for k, v in task_metrics.items():
                if k.startswith(task):
                    logger.info(f"  {k}: {v:.4f}")
        
        # Save test metrics
        test_metrics_path = os.path.join(final_model_path, "test_metrics.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        logger.info(f"Test metrics saved to {test_metrics_path}")
        
        return final_model_path
        
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        # Save interrupted checkpoint
        interrupted_path = os.path.join(args.output_dir, "checkpoint-interrupted")
        trainer.save_checkpoint("interrupted")
        logger.info(f"Interrupted checkpoint saved to {interrupted_path}")
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()

