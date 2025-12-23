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

from src.aqd.prompt_search.refine_prompt import PROMPT_REFINEMENT_CONTEXT
from src.aqd.schema import QueryNode, PromptNode, TreeNode
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
        self.no_ground_truth = True
        self.max_tokens = 4096

        self.environment_feedback = "Generated prompt does not reveal agent weakness and can't generate user query that elicit unpreferred response"
        self.instruction = "Generate a prompt to generate a user query so that the answer by agent is unpreferred"
    
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
        test_samples: List[TreeNode],
        data_processor = None,
    ) -> Dict[str, Any]:
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
        
        # Execute based on mode
        results = {}
        
        if mode == 'online':
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
        
        
        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Mode: {mode.upper().replace('_', ' ')}")

        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _train_single_sample(
        self,
        prompt_tree_node: TreeNode,
        data_processor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        total_samples: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        
        # Extract sample data
        context = PROMPT_REFINEMENT_CONTEXT
        

        
        # STEP 1: Initial generation (pre-train)
        print("Generating initial answer...")
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=self.instruction,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=self.use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir
        )
        
        # Extract answer and check correctness
        final_prompt = extract_answer(gen_response)
        prompt_node = PromptNode(final_prompt)
        score = data_processor.compute_score(self.config, prompt_node)
        prompt_tree_node.value = prompt_node
        pre_train_answer = final_prompt
        
        print(f"Pretrain Score: {score}")
        
        # Log bullet usage
        log_bullet_usage(usage_log_path, epoch, step, prompt_node.to_dict(), bullet_ids,
                       playbook=self.playbook, score=score)
        
        # Track pre-train result
        tracking_dict = {
            "pre_train_result": {
                "final_prompt": final_prompt,
                "score": score,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook)
            }
        }
        
        reflection_content = "(empty)"
        
        for round_num in range(self.max_num_rounds):
            print(f"Reflection round {round_num + 1}/{self.max_num_rounds}")
            
            # Get bullets for reflector
            playbook_bullets = extract_playbook_bullets(
                self.playbook, bullet_ids
            )
            
            # Reflect on error
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=self.instruction,
                reasoning_trace=gen_response,
                predicted_answer=final_prompt,
                ground_truth=None,
                environment_feedback=self.environment_feedback,
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
                question=self.instruction,
                playbook=self.playbook,
                context=context,
                reflection=reflection_content,
                use_json_mode=self.use_json_mode,
                call_id=f"{step_id}_post_reflect_round_{round_num}",
                log_dir=log_dir
            )
            
            final_prompt = extract_answer(gen_response)
            prompt_node = PromptNode(final_prompt)
            score = data_processor.compute_score(self.config, prompt_node)
            new_tree_node = TreeNode(prompt_node)
            prompt_tree_node.children.append(new_tree_node)
            new_tree_node.parent = prompt_tree_node

            print(f"reflection round {round_num + 1}!")
     
        
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
            question=self.instruction,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=self.use_json_mode,
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir
        )

        final_prompt = extract_answer(gen_response)
        prompt_node = PromptNode(final_prompt)
        post_train_score = data_processor.compute_score(self.config, prompt_node)
        new_tree_node = TreeNode(prompt_node)
        prompt_tree_node.children.append(new_tree_node)
        new_tree_node.parent = prompt_tree_node
        
        post_train_answer = final_prompt

        post_train_score = data_processor.compute_score(self.config, final_prompt)
        tracking_dict["post_train_result"] = {
            "final_prompt": final_prompt,
            "score": post_train_score,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook)
        }
        
        return pre_train_answer, post_train_answer, tracking_dict
    
    
    def test(
        self,
        test_samples: List[PromptNode],
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
        test_samples: List[TreeNode],
        data_processor,
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
      
        
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
        print(f"Curator frequency: every {self.curator_frequency} steps")
        
        # Split samples into windows
        
        epoch = 1  # Always 1 epoch
        global_step = 0
        
        for local_step, prompt_tree_node in enumerate(test_samples):
            global_step += 1
            local_step += 1
            
            
            
            # Use helper method for training single sample
            pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                prompt_tree_node=prompt_tree_node,
                data_processor=data_processor,
                step_id=f"online_train_s_{global_step}",
                epoch=epoch,
                step=global_step,
                usage_log_path=usage_log_path,
                log_dir=log_dir,
                total_samples=len(test_samples)
            )
            
            # Collect answers for accuracy calculation
            
            # Track pre-train and post-train results
            pre_train_post_train_result = {
                "global_step": global_step,
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
        
        
        window_train_result = {
            "global_step": global_step,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook),
            "playbook_stats": get_playbook_stats(self.playbook)
        }
        train_results.append(window_train_result)
        
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