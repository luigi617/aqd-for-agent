"""
ACE (Agent-Curator-Environment) System
Main orchestrator class for training and testing with playbook-based learning.

This module coordinates three agents:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and tags bullets
- Curator: Updates the playbook based on feedback
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from src.aqd.schema import QueryNode
from src.config import Config
from src.ace.core import Generator, Reflector, Curator, BulletpointAnalyzer
from src.ace.playbook_utils import *
from src.ace.logger import *
from src.ace.utils import *


class ACE:
    """
    Main ACE system orchestrator.
    
    Manages the training loop where:
    1. Generator produces answers using playbook
    2. Reflector analyzes answers and tags bullets
    3. Curator updates playbook based on feedback
    
    """
    
    def __init__(
        self,
        config: Config,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90
    ):
        """
        Initialize the ACE system.
        
        Args:
            config,
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer (0-1)
        """
        self.config = config

        self.api_provider = "openai"
        api_client = initialize_clients(self.api_provider)
        # Initialize the three agents
        self.generator = Generator(api_client, self.api_provider, self.config)
        self.reflector = Reflector(api_client, self.api_provider, self.config)
        self.curator = Curator(api_client, self.api_provider, self.config)
        
        # Initialize bulletpoint analyzer if requested and available
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        
        if use_bulletpoint_analyzer:
            self.bulletpoint_analyzer = BulletpointAnalyzer(self.config)
            print(f"✓ BulletpointAnalyzer initialized (threshold={bulletpoint_analyzer_threshold})")
        else:
            self.bulletpoint_analyzer = None
        
        self.playbook = self._initialize_empty_playbook()
        
        self.best_playbook = self.playbook
        # Track global bullet ID
        self.next_global_id = 1

        self.max_num_rounds = 100
        
        self.num_epochs = 1
        
        # Extract additional parameters
        self.curator_frequency = 1
        self.task_name = "AQD"
        self.save_steps = 50
        self.use_json_mode = True
        self.test_workers = self.config.n_threads
        self.online_eval_frequency = 100
        self.no_ground_truth = True
    
    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    def _setup_paths(self, task_name: str, mode: str) -> Tuple[str, str]:
        """
        Setup logging paths and directories.
        
        Args:
            save_dir: Base path for saving results
            task_name: task name
            mode: 'offline', 'online', or 'eval_only'
            
        Returns:
            Tuple of (usage_log_path, playbook_dir)
        """

        save_dir = self.config.output_path.parent

        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"ace_run_{timestamp}_{task_name}_{mode}"
        save_path = os.path.join(save_dir, run_folder)
        
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        if mode == "eval_only":
            return save_path, log_dir

        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)
        
        return save_path, usage_log_path, playbook_dir, log_dir
    
    def run(
        self,
        mode: str,
        test_samples: List[QueryNode],
        data_processor = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for running ACE system in different modes.
        
        Args:
            mode: Run mode - 'online', or 'eval_only'
            test_samples: Test samples (required for online and eval_only modes)
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            
        Returns:
            Dictionary with results depending on the mode
        """
        # Validate inputs
        if mode not in ['online', 'eval_only']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'online', or 'eval_only'")
        
        
        # Setup paths based on mode
        if mode == 'eval_only':
            save_path, log_dir = self._setup_paths(self.task_name, mode)
            usage_log_path = None
            playbook_dir = None
        else:
            save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(self.task_name, mode)
        
        # Print initial banner
        print(f"\n{'='*60}")
        print(f"ACE SYSTEM - {mode.upper().replace('_', ' ')} MODE")
        print(f"{'='*60}")
        print(f"Task: {self.task_name}")
        if mode == 'online':
            print(f"Test samples (used for training and testing): {len(test_samples)}")
        else:  # eval_only
            print(f"Test samples: {len(test_samples)}")
        print(f"{'='*60}\n")
        
        # Execute based on mode
        results = {}
        
        if mode == 'online':
            # ONLINE MODE WORKFLOW
            # 1. Run initial test
            print(f"\n{'='*60}")
            print(f"INITIAL TEST (before training)")
            print(f"{'='*60}\n")
            initial_test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                log_dir=log_dir,
                save_path=save_path,
                prefix="initial"
            )
            results['initial_test_results'] = initial_test_results
            print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run online training and testing
            print(f"\n{'='*60}")
            print(f"STARTING ONLINE TRAIN AND TEST")
            print(f"{'='*60}\n")
            online_results = self._online_train_and_test(
                test_samples=test_samples,
                data_processor=data_processor,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir
            )
            results['online_test_results'] = online_results
        
        else:  # eval_only
            # EVAL ONLY MODE WORKFLOW
            print(f"\n{'='*60}")
            print(f"RUNNING TEST")
            print(f"{'='*60}\n")
            test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                log_dir=log_dir,
                save_path=save_path,
                prefix="test"
            )
            results['test_results'] = test_results
        
        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Mode: {mode.upper().replace('_', ' ')}")
        
        if mode == 'online':
            print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
            print(f"Final Test Accuracy: {results['online_test_results']['accuracy']:.3f}")
        else:  # eval_only
            print(f"Test Accuracy: {results['test_results']['accuracy']:.3f}")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _run_test(
        self,
        test_samples: List[QueryNode],
        data_processor,
        playbook: str,
        log_dir: str,
        save_path: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        
        
        test_results, test_error_log = evaluate_test_set(
            data_processor,
            self.generator,
            playbook,
            test_samples,
            log_dir,
            max_workers=self.config.n_threads,
            use_json_mode=self.use_json_mode
        )

        # Save test results
        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_results": test_results,
                "error_log": test_error_log,
            }, f, indent=2)
        
        return test_results
    
    def _train_single_sample(
        self,
        query_node: QueryNode,
        data_processor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        total_samples: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Train on a single sample with reflection and curation.
        
        Args:
            task_dict: Sample dictionary with question, context, target
            data_processor: Data processor for evaluation
            step_id: Identifier string for this step (e.g., "train_e_1_s_10" or "online_train_w_1_s_5")
            epoch: Current epoch number
            step: Current step number
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            total_samples: Total number of samples in dataset
            
        Returns:
            Tuple of (pre_train_answer, post_train_answer, tracking_dict)
        """
        
        
        # Extract sample data
        context = ""
        question = query_node.query
        
        # STEP 1: Initial generation (pre-train)
        print("Generating initial answer...")
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=self.use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir
        )
        
        # Extract answer and check correctness
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer)
        pre_train_answer = final_answer
        
        print(f"Correct: {is_correct}")
        
        # Log bullet usage
        log_bullet_usage(usage_log_path, epoch, step, query_node.to_dict(), bullet_ids,
                       playbook=self.playbook, is_correct=is_correct)
        
        # Track pre-train result
        tracking_dict = {
            "pre_train_result": {
                "final_answer": final_answer,
                "is_correct": is_correct,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook)
            }
        }
        
        reflection_content = "(empty)"
        
        # STEP 2: Reflection and regeneration
        if not is_correct:
            # For incorrect answers - iterate reflection rounds
            for round_num in range(self.max_num_rounds):
                print(f"Reflection round {round_num + 1}/{self.max_num_rounds}")
                
                # Get bullets for reflector
                playbook_bullets = extract_playbook_bullets(
                    self.playbook, bullet_ids
                )
                
                # Reflect on error
                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=None,
                    environment_feedback="Predicted answer does not match ground truth",
                    bullets_used=playbook_bullets,
                    use_ground_truth=not self.no_ground_truth,
                    use_json_mode=self.use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir
                )
                
                # Update bullet counts
                if bullet_tags:
                    self.playbook = update_bullet_counts(
                        self.playbook, bullet_tags
                    )
                
                # Regenerate with reflection
                gen_response, bullet_ids, _ = self.generator.generate(
                    question=question,
                    playbook=self.playbook,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=self.use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir
                )
                
                final_answer = extract_answer(gen_response)
                
                if data_processor.answer_is_correct(final_answer):
                    print(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break
        
        else:
            # For correct answers - still run reflector to tag helpful bullets
            playbook_bullets = extract_playbook_bullets(
                self.playbook, bullet_ids
            )
            
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question,
                reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=None,
                environment_feedback="Predicted answer matches ground truth",
                bullets_used=playbook_bullets,
                use_ground_truth=not self.no_ground_truth,
                use_json_mode=self.use_json_mode,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir
            )
            
            # Update bullet counts
            if bullet_tags:
                self.playbook = update_bullet_counts(
                    self.playbook, bullet_tags
                )
            
            # Log with reflection
            log_bullet_usage(usage_log_path, epoch, step, query_node.to_dict(), bullet_ids,
                           playbook=self.playbook, 
                           reflection_content=reflection_content,
                           is_correct=is_correct)
        
        # STEP 3: Curator - Periodically update playbook
        if step % self.curator_frequency == 0:
            print(f"\n--- Running Curator at step {step} ---")
            
            stats = get_playbook_stats(self.playbook)
            
            self.playbook, self.next_global_id, operations, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                question_context=context,
                current_step=step,
                total_samples=total_samples,
                token_budget=self.config.budget,
                playbook_stats=stats,
                use_ground_truth=not self.no_ground_truth,
                use_json_mode=self.use_json_mode,
                call_id=step_id,
                log_dir=log_dir,
                next_global_id=self.next_global_id
            )
            
            # Run bulletpoint analyzer if enabled
            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                print(f"  Running BulletpointAnalyzer (threshold={self.bulletpoint_analyzer_threshold})...")
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True
                )
        
        # STEP 4: Post-curator generation
        gen_response, _, _ = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=self.use_json_mode,
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir
        )
        
        final_answer = extract_answer(gen_response)
        post_train_answer = final_answer
        
        post_train_is_correct = data_processor.answer_is_correct(final_answer)
        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_train_is_correct,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook)
        }
        
        return pre_train_answer, post_train_answer, tracking_dict
    
    
    def test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run testing with the playbook (backward compatibility wrapper).
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to be used for generator
            config: Configuration dictionary
            
        Returns:
            Dictionary with test results
        """
        # Temporarily set the playbook
        old_playbook = self.playbook
        self.playbook = playbook
        
        # Use the run method
        results = self.run(
            mode='eval_only',
            test_samples=test_samples,
            data_processor=data_processor,
            config=config
        )
        
        # Restore old playbook
        self.playbook = old_playbook
        
        # Return in the old format for backward compatibility
        return {
            "test_results": results['test_results'],
            "error_log": results.get('test_error_log', {}),
            "playbook": playbook
        }
    
    def _online_train_and_test(
        self,
        test_samples: List[QueryNode],
        data_processor,
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Run online training and testing
        
        Args:
            test_samples: List of samples to train and test on
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results, test results, and final playbook
        """
        
        # Initialize tracking
        train_results = []
        pre_train_post_train_results = []
        
        # Test tracking - accumulate across all windows
        correct_count_sample_based = 0
        correct_count = 0
        total_count = 0
        all_test_errors = []
        window_test_results = []
        print(f"Total samples: {len(test_samples)}")
        print(f"Window size: {self.online_eval_frequency}")
        print(f"Number of windows: {(len(test_samples) + self.online_eval_frequency - 1) // self.online_eval_frequency}")
        print(f"Curator frequency: every {self.curator_frequency} steps")
        
        # Split samples into windows
        num_windows = (len(test_samples) + self.online_eval_frequency - 1) // self.online_eval_frequency
        
        epoch = 1  # Always 1 epoch
        global_step = 0
        
        for window_idx in range(num_windows):
            start_idx = window_idx * self.online_eval_frequency
            end_idx = min((window_idx + 1) * self.online_eval_frequency, len(test_samples))
            window_samples = test_samples[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"WINDOW {window_idx + 1}/{num_windows}")
            print(f"Samples {start_idx} to {end_idx - 1}")
            print(f"{'='*60}")
            
            # =================================================================
            # STEP 1: TEST on window with current playbook (before training)
            # =================================================================
            print(f"\n--- Testing window {window_idx + 1} with current playbook ---")
            
            # Use evaluate_test_set for parallel evaluation
            window_test_results_dict, window_test_error_log = evaluate_test_set(
                data_processor,
                self.generator,
                self.playbook,
                window_samples,
                self.max_tokens,
                log_dir,
                max_workers=self.config.n_threads,
                use_json_mode=self.use_json_mode
            )
            
            # Extract results
            window_accuracy = window_test_results_dict['accuracy']
            window_correct = window_test_results_dict['correct']
            window_total = window_test_results_dict['total']
            correct_count_sample_based += window_correct
            correct_count += window_accuracy * window_total
            total_count += window_total
            
            # Add errors with window and global index information
            for error in window_test_error_log['errors']:
                all_test_errors.append({
                    "window": window_idx + 1,
                    "global_index": start_idx + error['index'],
                    "prediction": error['prediction'],
                    "ground_truth": error['ground_truth']
                })
            
            window_test_results.append({
                "window": window_idx + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_accuracy": window_accuracy,
                "window_correct": window_correct,
                "window_total": window_total
            })
            
            # Calculate cumulative test accuracy so far
            cumulative_test_accuracy = correct_count / total_count
            
            print(f"Window {window_idx + 1} test accuracy: {window_accuracy:.3f}")
            print(f"Cumulative test accuracy so far: {cumulative_test_accuracy:.3f} "
                  f"({total_count} samples)")
            
            # =================================================================
            # STEP 2: TRAIN on window (same as offline_train)
            # =================================================================
            print(f"\n--- Training on window {window_idx + 1} ---")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            
            for local_step, task_dict in enumerate(window_samples):
                global_step += 1
                local_step += 1
                
                print(f"\n--- Window {window_idx + 1}, Step {local_step}/{len(window_samples)} "
                      f"(Global step {global_step}) ---")
                
                target = task_dict.get("target", "")
                
                # Use helper method for training single sample
                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    data_processor=data_processor,
                    step_id=f"online_train_s_{global_step}",
                    epoch=epoch,
                    step=global_step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    total_samples=len(test_samples)
                )
                
                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                
                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "window": window_idx + 1,
                    "global_step": global_step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook
                if global_step % self.save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"step_{global_step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
            
            # End of window - compute training accuracies for this window
            pre_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_pre_train, epoch_targets_pre_train
            )
            post_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_post_train, epoch_targets_post_train
            )
            
            window_train_result = {
                "window": window_idx + 1,
                "global_step": global_step,
                "train_result": {
                    "pre_train_accuracy": pre_train_accuracy,
                    "post_train_accuracy": post_train_accuracy
                },
                "cumulative_test_accuracy": cumulative_test_accuracy,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook),
                "playbook_stats": get_playbook_stats(self.playbook)
            }
            train_results.append(window_train_result)
            
            print(f"\nWindow {window_idx + 1} training complete:")
            print(f"  Pre-train accuracy: {pre_train_accuracy:.3f}")
            print(f"  Post-train accuracy: {post_train_accuracy:.3f}")
            
            # Save window playbook
            window_playbook_path = os.path.join(
                playbook_dir, f"window_{window_idx + 1}_final_playbook.txt"
            )
            with open(window_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # All windows complete
        print(f"\n{'='*60}")
        print(f"ONLINE TRAIN AND TEST COMPLETE")
        print(f"{'='*60}")
        
        # Calculate final cumulative test accuracy
        assert total_count == len(test_samples)
        final_test_accuracy = correct_count / total_count
        
        test_results = {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
            "window_results": window_test_results
        }
        
        test_error_log = {
            "accuracy": final_test_accuracy,
            "errors": all_test_errors
        }

        # Save test results
        test_results_path = os.path.join(save_path, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_accuracy": final_test_accuracy,
                "test_results": test_results,
                "test_error_log": test_error_log
            }, f, indent=2)
        
        # Save training results (per window)
        train_results_path = os.path.join(save_path, "train_results.json")
        with open(train_results_path, "w") as f:
            json.dump({"train_results": train_results}, f, indent=2)
        
        # Save pre-train/post-train results
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        
        print(f"\n{'='*60}")
        print(f"ONLINE TRAINING AND TESTING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Test Accuracy: {final_test_accuracy:.3f}")
        print(f"{'='*60}\n")
        
        return {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
        }