"""
SocialLLM Trainer

This module provides training functionality for the SocialLLM model with
multi-task learning, engagement prediction, and social media specific metrics.
"""

import os
import json
import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class SocialLLMTrainingArguments:
    """
    Arguments for SocialLLM training
    """
    output_dir: str
    
    # Basic training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Curriculum learning and task mixing
    task_weights: Dict[str, float] = field(default_factory=dict)
    use_curriculum_learning: bool = True
    curriculum_steps: List[int] = field(default_factory=list)
    
    # Optimization parameters
    warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: Optional[int] = 3
    
    # Other parameters
    seed: int = 42
    fp16: bool = False
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    def __post_init__(self):
        # Set default task weights if not provided
        if not self.task_weights:
            self.task_weights = {
                "generation": 1.0,
                "sentiment_analysis": 0.5,
                "engagement_prediction": 0.5,
                "hashtag_suggestion": 0.3,
                "viral_potential": 0.3,
                "audience_targeting": 0.3,
                "topic_classification": 0.3
            }
        
        # Set default curriculum steps if not provided
        if not self.curriculum_steps and self.use_curriculum_learning:
            self.curriculum_steps = [1000, 2000, 3000]


class SocialLLMTrainer:
    """
    Trainer for the SocialLLM model with multi-task learning support
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: SocialLLMTrainingArguments,
        train_datasets: Dict[str, Dataset],
        eval_datasets: Optional[Dict[str, Dataset]] = None,
        tokenizer = None,
        optimizer: Optional[Optimizer] = None
    ):
        self.model = model
        self.args = args
        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets
        self.tokenizer = tokenizer
        
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize WandB if available and requested
        if args.use_wandb and has_wandb:
            wandb_project = args.wandb_project or "social-llm"
            wandb.init(project=wandb_project)
            wandb.config.update(vars(args))
        
        # Initialize task-specific dataloaders
        self.train_dataloaders = {}
        for task, dataset in train_datasets.items():
            if len(dataset) > 0:  # Only create dataloader if dataset has samples
                self.train_dataloaders[task] = DataLoader(
                    dataset,
                    batch_size=args.per_device_train_batch_size,
                    shuffle=True,
                    drop_last=True
                )
        
        # Initialize evaluation dataloaders if provided
        self.eval_dataloaders = {}
        if eval_datasets:
            for task, dataset in eval_datasets.items():
                if len(dataset) > 0:  # Only create dataloader if dataset has samples
                    self.eval_dataloaders[task] = DataLoader(
                        dataset,
                        batch_size=args.per_device_eval_batch_size
                    )
        
        # Create optimizer and scheduler
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
            
        # Setup learning rate scheduler
        self.scheduler = self._create_lr_scheduler()
        
        # Current training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        
        # Dictionary to track metrics
        self.metrics = {task: {} for task in train_datasets.keys()}
        
        # Mixed precision training
        self.fp16 = args.fp16
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Task curriculum state
        self.active_tasks = list(train_datasets.keys())
        self.current_curriculum_step = 0
        
        # Log initialization status
        logger.info(f"Initialized trainer with {len(self.train_dataloaders)} training tasks")
        logger.info(f"Model will train on {self.device}")
                
        logger.info(f"Initialized trainer with {len(train_datasets)} training tasks")
        logger.info(f"Model will train on {self.device}")
    
    def _create_optimizer(self) -> AdamW:
        # Separate parameters that should have weight decay applied
        """Creates an AdamW optimizer with grouped parameters.
        
        This function separates model parameters into two groups: those that should
        have weight decay applied and those that should not. It then initializes and
        returns an AdamW optimizer configured with the specified learning rate, betas,
        and epsilon values from the `self.args` configuration.
        
        Args:
            self: The instance of the class containing model parameters and args.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon
        )
    
    def _create_lr_scheduler(self) -> LambdaLR:
        # Calculate total training steps
        """Create a learning rate scheduler using a lambda function."""
        num_steps_per_epoch = sum(len(dl) for dl in self.train_dataloaders.values())
        total_steps = num_steps_per_epoch * self.args.num_train_epochs
        
        def lr_lambda(current_step: int):
            """Calculate learning rate lambda based on current step."""
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - self.args.warmup_steps)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _update_curriculum(self):
        """Updates the curriculum step based on global training steps.
        
        This method checks if the current global step meets or exceeds any defined
        curriculum steps. If so, it advances to the next curriculum step and updates
        the active tasks accordingly. The function logs the advancement of curriculum
        steps and the updated list of active tasks.
        """
        if not self.args.use_curriculum_learning:
            return
        
        # Check if we need to advance to the next curriculum step
        for i, step in enumerate(self.args.curriculum_steps):
            if self.global_step >= step and self.current_curriculum_step == i:
                self.current_curriculum_step = i + 1
                logger.info(f"Advancing to curriculum step {self.current_curriculum_step}")
                
                # Update active tasks based on curriculum
                if self.current_curriculum_step == 1:
                    # Start with only language modeling
                    self.active_tasks = ["generation"]
                elif self.current_curriculum_step == 2:
                    # Add sentiment analysis and engagement prediction
                    self.active_tasks = ["generation", "sentiment_analysis", "engagement_prediction"]
                else:
                    # Enable all tasks
                    self.active_tasks = list(self.train_datasets.keys())
                
                logger.info(f"Active tasks: {self.active_tasks}")
    
    def train(self) -> Dict[str, float]:
        """Train a model using multiple dataloaders with optional curriculum learning and
        mixed precision training.
        
        This function sets up training tracking, iterates through epochs, and processes
        batches from each task's dataloader. It includes forward passes, loss
        calculations, backward passes, parameter updates, and logging of metrics.
        Additionally, it handles curriculum updates, evaluation at specified intervals,
        and saving checkpoints.
        
        Returns:
            Dict[str, float]: A dictionary containing final training metrics.
        
        Raises:
            ValueError: If the model does not return a loss for any task.
        """
        logger.info("Starting training")
        
        # Setup training tracking
        total_steps = sum(len(dl) for dl in self.train_dataloaders.values()) * self.args.num_train_epochs
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        # Training loop
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch+1}/{self.args.num_train_epochs}")
            
            # Track metrics for this epoch
            epoch_losses = {task: [] for task in self.train_dataloaders.keys()}
            
            # Iterate through all tasks
            for task, dataloader in self.train_dataloaders.items():
                # Skip task if not active in current curriculum
                if self.args.use_curriculum_learning and task not in self.active_tasks:
                    continue
                
                task_weight = self.args.task_weights.get(task, 1.0)
                
                # Process all batches for this task
                for step, batch in enumerate(dataloader):
                    # Update curriculum if needed
                    self._update_curriculum()
                    
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass with automatic mixed precision if enabled
                    if self.fp16:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                hashtag_ids=batch.get("hashtag_ids"),
                                emoji_ids=batch.get("emoji_ids"),
                                mention_ids=batch.get("mention_ids"),
                                url_flags=batch.get("url_flags"),
                                task=task,
                                labels=batch.get("labels")
                            )
                            
                            # Check if loss was returned
                            loss = outputs.get("loss")
                            if loss is None:
                                raise ValueError(f"Model did not return loss for task {task}")
                            
                            # Apply task weight
                            weighted_loss = loss * task_weight
                    else:
                        # Standard forward pass
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            hashtag_ids=batch.get("hashtag_ids"),
                            emoji_ids=batch.get("emoji_ids"),
                            mention_ids=batch.get("mention_ids"),
                            url_flags=batch.get("url_flags"),
                            task=task,
                            labels=batch.get("labels")
                        )
                        
                        # Check if loss was returned
                        loss = outputs.get("loss")
                        if loss is None:
                            raise ValueError(f"Model did not return loss for task {task}")
                        
                        # Apply task weight
                        weighted_loss = loss * task_weight
                    
                    # Backward pass
                    if self.fp16:
                        self.scaler.scale(weighted_loss).backward()
                    else:
                        weighted_loss.backward()
                    
                    # Track loss for this task
                    epoch_losses[task].append(loss.item())
                    
                    # Update parameters
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update tracking
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Log progress if needed
                    if self.global_step % self.args.logging_steps == 0:
                        # Log task-specific metrics
                        task_metrics = {f"{task}_loss": np.mean(epoch_losses[task][-100:])}
                        
                        # Log to console
                        log_msg = f"Step: {self.global_step}, {task} Loss: {task_metrics[f'{task}_loss']:.4f}"
                        logger.info(log_msg)
                        
                        # Log to WandB if enabled
                        if self.args.use_wandb and has_wandb:
                            wandb.log(task_metrics, step=self.global_step)
                    
                    # Evaluate if needed
                    if self.eval_dataloaders and self.global_step % self.args.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        # Log evaluation metrics
                        if self.args.use_wandb and has_wandb:
                            wandb.log(eval_metrics, step=self.global_step)
                    
                    # Save checkpoint if needed
                    if self.global_step % self.args.save_steps == 0:
                        self.save_checkpoint()
            
            # Calculate epoch-level metrics
            for task in epoch_losses:
                if epoch_losses[task]:  # Only calculate if we have losses for this task
                    avg_loss = np.mean(epoch_losses[task])
                    logger.info(f"Epoch {epoch+1}, Task {task}, Avg Loss: {avg_loss:.4f}")
            
            # Always evaluate at the end of an epoch
            if self.eval_dataloaders:
                eval_metrics = self.evaluate()
                
                # Log evaluation metrics for the epoch
                if self.args.use_wandb and has_wandb:
                    wandb.log(eval_metrics, step=self.global_step)
            
            # Save checkpoint at the end of each epoch
            # Save checkpoint at the end of each epoch
            self.save_checkpoint(f"epoch_{epoch+1}")
            
        progress_bar.close()
        logger.info("Training completed")
        
        # Return final metrics
        return self.metrics
    
    def save_checkpoint(self, checkpoint_suffix: str = "") -> str:
        # Create checkpoint directory
        """Saves the model and training state as a checkpoint."""
        checkpoint_dir = os.path.join(
            self.args.output_dir, 
            f"checkpoint-{self.global_step}{'-' + checkpoint_suffix if checkpoint_suffix else ''}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer and scheduler states
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save tokenizer if provided
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "metrics": self.metrics,
            "args": vars(self.args)
        }
        
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Clean up old checkpoints if save_total_limit is set
        if self.args.save_total_limit:
            self._cleanup_checkpoints()
            
        logger.info(f"Model checkpoint saved to {checkpoint_dir}")
        return checkpoint_dir
    
    def _cleanup_checkpoints(self) -> None:
        """Clean up excess checkpoints based on the save limit."""
        if not self.args.save_total_limit:
            return
            
        # List all checkpoint directories
        checkpoints = []
        for dirname in os.listdir(self.args.output_dir):
            if dirname.startswith("checkpoint-"):
                checkpoint_dir = os.path.join(self.args.output_dir, dirname)
                if os.path.isdir(checkpoint_dir):
                    checkpoints.append(checkpoint_dir)
        
        # Sort by global step (extraction from directory name)
        def get_step(checkpoint_dir: str) -> int:
            """Extracts and returns the step number from the checkpoint directory name."""
            dir_name = os.path.basename(checkpoint_dir)
            step_str = dir_name.split("-")[1].split("-")[0]  # Extract step number
            try:
                return int(step_str)
            except ValueError:
                return 0
                
        checkpoints.sort(key=get_step)
        
        # Remove oldest checkpoints if exceeding limit
        num_to_remove = max(0, len(checkpoints) - self.args.save_total_limit)
        if num_to_remove > 0:
            to_remove = checkpoints[:num_to_remove]
            for checkpoint_dir in to_remove:
                logger.info(f"Removing old checkpoint: {checkpoint_dir}")
                import shutil
                shutil.rmtree(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        # Load model weights
        """Loads model weights, optimizer state, scheduler state, and training state from
        a specified checkpoint directory.
        
        This function attempts to load the model weights, optimizer state, scheduler
        state, and training state from files located in the given checkpoint directory.
        If any of these files are missing, it logs a warning. The function uses
        `torch.load` to load the model, optimizer, and scheduler states, and reads the
        training state from a JSON file. It updates the class instance's attributes
        with the loaded values.
        
        Args:
            checkpoint_dir (str): The directory containing the checkpoint files.
        """
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning(f"Model weights not found at {model_path}")
        
        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info(f"Loaded optimizer state from {optimizer_path}")
        else:
            logger.warning(f"Optimizer state not found at {optimizer_path}")
        
        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            logger.info(f"Loaded scheduler state from {scheduler_path}")
        else:
            logger.warning(f"Scheduler state not found at {scheduler_path}")
        
        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                training_state = json.load(f)
                
            self.global_step = training_state.get("global_step", 0)
            self.epoch = training_state.get("epoch", 0)
            self.best_metric = training_state.get("best_metric", float("inf"))
            self.metrics = training_state.get("metrics", {})
            
            logger.info(f"Restored training state from {state_path}")
            logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")
        else:
            logger.warning(f"Training state not found at {state_path}")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on multiple tasks using provided datasets.
        
        This function iterates over each task, processes the input data, and computes
        various metrics such as accuracy, F1 score, mean squared error, and mean
        absolute error depending on the task type. It logs the task-specific metrics
        and updates the trainer's state with these metrics. Additionally, it tracks the
        best model based on specified metrics.
        
        Returns:
            dict: A dictionary containing all computed evaluation metrics.
        """
        if not self.eval_dataloaders:
            logger.warning("No evaluation datasets provided")
            return {}
            
        logger.info("Starting evaluation")
        self.model.eval()
        
        # Track metrics for all tasks
        eval_metrics = {}
        
        # Evaluate on each task
        for task, dataloader in self.eval_dataloaders.items():
            # Skip if no task-specific evaluation data
            if not dataloader:
                continue
                
            # Initialize task-specific metrics
            task_loss = 0.0
            task_samples = 0
            
            # Task-specific prediction tracking
            task_preds = []
            task_labels = []
            
            # Process all batches for this task
            with torch.no_grad():
                for step, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {task}")):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        hashtag_ids=batch.get("hashtag_ids"),
                        emoji_ids=batch.get("emoji_ids"),
                        mention_ids=batch.get("mention_ids"),
                        url_flags=batch.get("url_flags"),
                        task=task,
                        labels=batch.get("labels")
                    )
                    
                    # Get loss
                    loss = outputs.get("loss")
                    if loss is not None:
                        task_loss += loss.item() * batch["input_ids"].size(0)
                        task_samples += batch["input_ids"].size(0)
                    
                    # Get predictions
                    logits = outputs.get("logits")
                    if logits is not None and batch.get("labels") is not None:
                        # Handle different task types
                        if task == "generation":
                            # For generation, we don't compute metrics on validation
                            # Just track the loss
                            pass
                        elif task in ["sentiment_analysis", "topic_classification", "audience_targeting"]:
                            # For classification tasks
                            preds = torch.argmax(logits, dim=-1).cpu().numpy()
                            labels = batch["labels"].cpu().numpy()
                            task_preds.extend(preds.flatten())
                            task_labels.extend(labels.flatten())
                        elif task == "hashtag_suggestion":
                            # For multi-label classification
                            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                            labels = batch["labels"].cpu().numpy()
                            task_preds.append(preds)
                            task_labels.append(labels)
                        elif task == "viral_potential":
                            # For regression task
                            preds = logits.cpu().numpy()
                            labels = batch["labels"].cpu().numpy()
                            task_preds.extend(preds.flatten())
                            task_labels.extend(labels.flatten())
                        elif task == "engagement_prediction":
                            # For multi-output regression task
                            preds = logits.cpu().numpy()
                            labels = batch["labels"].cpu().numpy()
                            # Store array format for engagement metrics
                            task_preds.append(preds)
                            task_labels.append(labels)
                        
            # Calculate task-specific metrics
            if task_samples > 0:
                avg_loss = task_loss / task_samples
                eval_metrics[f"{task}_loss"] = avg_loss
                
                # Classification metrics
                if task in ["sentiment_analysis", "topic_classification", "audience_targeting"] and task_preds:
                    accuracy = accuracy_score(task_labels, task_preds)
                    f1 = f1_score(task_labels, task_preds, average='weighted')
                    eval_metrics[f"{task}_accuracy"] = accuracy
                    eval_metrics[f"{task}_f1"] = f1
                
                # Hashtag suggestion metrics
                elif task == "hashtag_suggestion" and task_preds:
                    # Combine batches
                    all_preds = np.vstack(task_preds)
                    all_labels = np.vstack(task_labels)
                    
                    # Calculate metrics for multi-label classification
                    hashtag_f1 = f1_score(all_labels, all_preds, average='weighted')
                    hashtag_precision = self._calculate_hashtag_precision(all_labels, all_preds)
                    hashtag_recall = self._calculate_hashtag_recall(all_labels, all_preds)
                    
                    eval_metrics[f"{task}_f1"] = hashtag_f1
                    eval_metrics[f"{task}_precision"] = hashtag_precision
                    eval_metrics[f"{task}_recall"] = hashtag_recall
                
                # Viral potential metrics
                elif task == "viral_potential" and task_preds:
                    mse = mean_squared_error(task_labels, task_preds)
                    mae = mean_absolute_error(task_labels, task_preds)
                    eval_metrics[f"{task}_mse"] = mse
                    eval_metrics[f"{task}_mae"] = mae
                
                # Engagement prediction metrics
                elif task == "engagement_prediction" and task_preds:
                    # Combine batches
                    all_preds = np.vstack(task_preds)
                    all_labels = np.vstack(task_labels)
                    
                    # Calculate metrics for each engagement type
                    engagement_types = ["likes", "shares", "comments", "reach"]
                    for i, e_type in enumerate(engagement_types):
                        if i < all_preds.shape[1]:  # Check if this engagement type was predicted
                            e_preds = all_preds[:, i]
                            e_labels = all_labels[:, i]
                            
                            e_mse = mean_squared_error(e_labels, e_preds)
                            e_mae = mean_absolute_error(e_labels, e_preds)
                            
                            # Log RMSE for more interpretable metric
                            e_rmse = math.sqrt(e_mse)
                            
                            eval_metrics[f"{task}_{e_type}_rmse"] = e_rmse
                            eval_metrics[f"{task}_{e_type}_mae"] = e_mae
            
            # Log task-specific metrics
            task_metrics_str = ", ".join([f"{k.split('_', 1)[1]}: {v:.4f}" for k, v in eval_metrics.items() if k.startswith(task)])
            logger.info(f"Eval {task}: {task_metrics_str}")
        
        # Store metrics in trainer state
        for k, v in eval_metrics.items():
            task = k.split("_")[0]
            if task not in self.metrics:
                self.metrics[task] = {}
            self.metrics[task][k] = v
        
        # Track best metrics for checkpoint selection
        if "generation_loss" in eval_metrics:
            current_metric = eval_metrics["generation_loss"]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint("best")
                logger.info(f"New best model with generation_loss: {current_metric:.4f}")
        elif "engagement_prediction_likes_rmse" in eval_metrics:
            # Alternative metric for best model selection
            current_metric = eval_metrics["engagement_prediction_likes_rmse"]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint("best")
                logger.info(f"New best model with engagement_prediction_likes_rmse: {current_metric:.4f}")
        
        self.model.train()
        return eval_metrics
    
    def _calculate_hashtag_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate precision for each sample and then average
        """Calculate the precision of hashtag predictions.
        
        This method computes the precision for each sample by comparing true and
        predicted hashtags, then averages these values across all samples. Precision is
        defined as the ratio of true positives to the total number of positive
        predictions. Special cases are handled where no hashtags were expected or none
        were predicted.
        
        Args:
            y_true: A numpy array representing the ground truth hashtag labels.
            y_pred: A numpy array representing the predicted hashtag labels.
        
        Returns:
            The average precision as a float.
        """
        precisions = []
        for i in range(y_true.shape[0]):
            true_positives = np.sum(y_true[i] * y_pred[i])
            predicted_positives = np.sum(y_pred[i])
            
            if predicted_positives > 0:
                precisions.append(true_positives / predicted_positives)
            elif np.sum(y_true[i]) == 0:
                # If no hashtags were expected and none predicted, it's correct
                precisions.append(1.0)
            else:
                # If hashtags were expected but none predicted, precision is 0
                precisions.append(0.0)
                
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_hashtag_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate recall for each sample and then average
        """Calculates the average recall across multiple samples.
        
        This function computes the recall for each sample by comparing true and
        predicted hashtag vectors. It then averages these recalls to produce a single
        metric. If there are no actual positives in a sample but no predictions either,
        it considers the recall as 1.0. If there are no actual positives but some
        predicted, it also assigns a recall of 1.0.
        
        Args:
            y_true (np.ndarray): A binary array indicating true presence of hashtags.
            y_pred (np.ndarray): A binary array indicating predicted presence of hashtags.
        
        Returns:
            float: The average recall across all samples.
        """
        recalls = []
        for i in range(y_true.shape[0]):
            true_positives = np.sum(y_true[i] * y_pred[i])
            actual_positives = np.sum(y_true[i])
            
            if actual_positives > 0:
                recalls.append(true_positives / actual_positives)
            elif np.sum(y_pred[i]) == 0:
                # If no hashtags were expected and none predicted, it's correct
                recalls.append(1.0)
            else:
                # If no hashtags were expected but some predicted, recall is 1
                recalls.append(1.0)
                
        return np.mean(recalls) if recalls else 0.0
    
    def _calculate_social_media_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # This could be extended with more social media specific metrics
        """Calculate social media metrics including MAPE and viral prediction accuracy."""
        metrics = {}
        
        # Calculate engagement prediction error percentage
        if y_true.size > 0 and np.mean(y_true) > 0:
            # Mean Absolute Percentage Error for engagement
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
            metrics["engagement_mape"] = mape
            
            # Calculate viral threshold accuracy (% of content correctly predicted as "viral")
            # For this example, we consider "viral" as top 10% of engagement
            viral_threshold = np.percentile(y_true, 90)
            predicted_viral = y_pred >= viral_threshold
            actual_viral = y_true >= viral_threshold
            
            viral_accuracy = np.mean(predicted_viral == actual_viral) * 100
            metrics["viral_prediction_accuracy"] = viral_accuracy
            
            # Add logarithmic error metrics (often better for engagement metrics with wide value ranges)
            # Add small value to avoid log(0)
            y_true_log = np.log1p(y_true)
            y_pred_log = np.log1p(y_pred)
            log_mse = mean_squared_error(y_true_log, y_pred_log)
            metrics["log_mse"] = log_mse
        
        return metrics
    
    def generate_content(
        self, 
        prompt: str, 
        max_length: int = 100,
        do_sample: bool = True,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        # Encode the prompt
        """Generates text based on the given prompt using a language model."""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Generate text
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                top_p=0.95
            )
        
        # Decode the generated sequences
        generated_texts = []
        for output_sequence in output_sequences:
            text = self.tokenizer.decode(output_sequence, skip_special_tokens=True)
            generated_texts.append(text)
        
        # Set the model back to training mode
        self.model.train()
        
        return generated_texts
    
    def predict_engagement(self, text: str, has_media: bool = False) -> Dict[str, float]:
        # Encode the text
        """Predicts engagement metrics (likes, shares, comments, reach) for a given text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task="engagement_prediction"
            )
            
            # Get logits
            logits = outputs["logits"]
        
        # Convert to engagement values
        engagement_types = ["likes", "shares", "comments", "reach"]
        engagement_values = {
            engagement_type: round(float(value))
            for engagement_type, value in zip(engagement_types, logits[0].cpu().numpy())
        }
        
        # Set model back to training mode
        self.model.train()
        
        return engagement_values
    
    def suggest_hashtags(self, text: str, num_hashtags: int = 5) -> List[str]:
        # Encode the text
        """Suggests top hashtags for a given text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task="hashtag_suggestion"
            )
            
            # Get logits
            logits = outputs["logits"]
        
        # Get the top hashtags
        probs = torch.sigmoid(logits)
        top_indices = torch.topk(probs[0], min(num_hashtags, probs.size(1))).indices.cpu().numpy()
        
        # Map indices to hashtags
        # In a real implementation, you would have a mapping from indices to hashtag strings
        # For this example, we'll create placeholder hashtags
        hashtags = [f"#tag{idx}" for idx in top_indices]
        
        # Set model back to training mode
        self.model.train()
        
        return hashtags
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        # Encode the text
        """Analyze sentiment of the given text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task="sentiment_analysis"
            )
            
            # Get logits
            logits = outputs["logits"]
        
        # Get the sentiment class and probabilities
        probs = torch.softmax(logits, dim=-1)
        sentiment_class = torch.argmax(probs, dim=-1).item()
        
        # Map class index to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(sentiment_class, "neutral")
        
        # Get probabilities
        prob_values = probs[0].cpu().numpy()
        sentiment_probs = {
            "negative": float(prob_values[0]),
            "neutral": float(prob_values[1]),
            "positive": float(prob_values[2])
        }
        
        # Set model back to training mode
        self.model.train()
        
        return {
            "sentiment": sentiment,
            "probabilities": sentiment_probs,
            "score": float(prob_values[2] - prob_values[0])  # Score from -1 to 1
        }
