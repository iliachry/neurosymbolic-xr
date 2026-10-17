"""
VR Training Assistant with Procedural Reasoning

Combines hand/gaze tracking with symbolic procedure validation for
interactive VR training scenarios.

This is the third of four neurosymbolic AI systems.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import time

import numpy as np
import torch

try:
    from pyswip import Prolog
except ImportError:
    print("Warning: PySwip not installed. Install with: pip install pyswip")
    Prolog = None


class StepStatus(Enum):
    """Status of a procedure step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class HandGesture(Enum):
    """Recognized hand gestures."""
    IDLE = "idle"
    POINT = "point"
    GRAB = "grab"
    PINCH = "pinch"
    RELEASE = "release"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OPEN_PALM = "open_palm"


@dataclass
class HandPose:
    """Hand pose with joint positions and gesture."""
    
    timestamp: float
    hand_type: str  # 'left' or 'right'
    joints: np.ndarray  # (21, 3) joint positions in world space
    gesture: HandGesture
    confidence: float
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'timestamp': self.timestamp,
            'hand_type': self.hand_type,
            'joints': self.joints.tolist(),
            'gesture': self.gesture.value,
            'confidence': self.confidence
        }


@dataclass
class GazePose:
    """Gaze direction and fixation point."""
    
    timestamp: float
    origin: np.ndarray  # (3,) eye position
    direction: np.ndarray  # (3,) gaze direction (normalized)
    fixation_point: Optional[np.ndarray]  # (3,) 3D fixation point
    fixation_object: Optional[str]  # Object ID being looked at
    confidence: float
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'timestamp': self.timestamp,
            'origin': self.origin.tolist(),
            'direction': self.direction.tolist(),
            'fixation_point': self.fixation_point.tolist() if self.fixation_point is not None else None,
            'fixation_object': self.fixation_object,
            'confidence': self.confidence
        }


@dataclass
class ProcedureStep:
    """Single step in a training procedure."""
    
    id: str
    description: str
    action: str  # Action type (e.g., 'grab', 'place', 'press')
    target_object: Optional[str]  # Target object ID
    preconditions: List[str]  # Logical preconditions
    postconditions: List[str]  # Expected outcomes
    constraints: List[str]  # Constraints to satisfy
    timeout: float  # Maximum time allowed (seconds)
    hints: List[str]  # Hints for the user
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'description': self.description,
            'action': self.action,
            'target_object': self.target_object,
            'preconditions': self.preconditions,
            'postconditions': self.postconditions,
            'constraints': self.constraints,
            'timeout': self.timeout,
            'hints': self.hints,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


@dataclass
class TrainingProcedure:
    """Complete training procedure with steps."""
    
    id: str
    name: str
    description: str
    steps: List[ProcedureStep]
    current_step_idx: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def get_current_step(self) -> Optional[ProcedureStep]:
        """Get current step."""
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None
    
    def advance_step(self) -> bool:
        """Advance to next step. Returns True if successful."""
        if self.current_step_idx < len(self.steps) - 1:
            self.current_step_idx += 1
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status == StepStatus.COMPLETED for step in self.steps)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps],
            'current_step_idx': self.current_step_idx,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class TrainingAssistant:
    """
    VR Training Assistant with procedural reasoning.
    
    Features:
    - Hand tracking and gesture recognition
    - Gaze tracking and attention monitoring
    - Procedural step validation with Prolog
    - Real-time feedback and hints
    - Performance analytics
    - Adaptive difficulty
    
    Args:
        device: Computation device
        use_prolog: Enable Prolog reasoning engine
    """
    
    def __init__(
        self,
        device: str = "cuda",
        use_prolog: bool = True
    ):
        self.device = device
        self.use_prolog = use_prolog
        
        # Initialize Prolog reasoning engine
        if use_prolog and Prolog is not None:
            self.prolog = Prolog()
            self._initialize_prolog_kb()
        else:
            self.prolog = None
        
        # Tracking state
        self.left_hand: Optional[HandPose] = None
        self.right_hand: Optional[HandPose] = None
        self.gaze: Optional[GazePose] = None
        
        # Current procedure
        self.current_procedure: Optional[TrainingProcedure] = None
        
        # World state (facts about the environment)
        self.world_state: Dict[str, any] = {}
        
        # Performance metrics
        self.metrics = {
            'total_time': 0.0,
            'step_times': [],
            'errors': 0,
            'hints_used': 0,
            'completion_rate': 0.0
        }
        
        # Callbacks for feedback
        self.feedback_callbacks: List[Callable] = []
    
    def start_procedure(self, procedure: TrainingProcedure) -> None:
        """
        Start a training procedure.
        
        Args:
            procedure: TrainingProcedure to execute
        """
        self.current_procedure = procedure
        procedure.start_time = time.time()
        
        # Initialize first step
        first_step = procedure.get_current_step()
        if first_step:
            first_step.status = StepStatus.IN_PROGRESS
            first_step.start_time = time.time()
            
            # Provide initial feedback
            self._provide_feedback(f"Starting: {procedure.name}")
            self._provide_feedback(f"Step 1: {first_step.description}")
    
    def update(
        self,
        left_hand: Optional[HandPose] = None,
        right_hand: Optional[HandPose] = None,
        gaze: Optional[GazePose] = None,
        world_state: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Update training assistant with current tracking data.
        
        Args:
            left_hand: Left hand pose
            right_hand: Right hand pose
            gaze: Gaze pose
            world_state: Current world state facts
            
        Returns:
            Dict with validation results and feedback
        """
        # Update tracking state
        if left_hand:
            self.left_hand = left_hand
        if right_hand:
            self.right_hand = right_hand
        if gaze:
            self.gaze = gaze
        if world_state:
            self.world_state.update(world_state)
        
        # Check if procedure is active
        if self.current_procedure is None:
            return {'status': 'no_procedure', 'feedback': []}
        
        current_step = self.current_procedure.get_current_step()
        if current_step is None:
            return {'status': 'procedure_complete', 'feedback': []}
        
        # Validate current step
        validation_result = self._validate_step(current_step)
        
        # Check for step completion
        if validation_result['completed']:
            self._complete_step(current_step)
            
            # Advance to next step
            if self.current_procedure.advance_step():
                next_step = self.current_procedure.get_current_step()
                next_step.status = StepStatus.IN_PROGRESS
                next_step.start_time = time.time()
                self._provide_feedback(f"Step {self.current_procedure.current_step_idx + 1}: {next_step.description}")
            else:
                # Procedure complete
                self._complete_procedure()
        
        # Check for errors
        if validation_result['errors']:
            self.metrics['errors'] += len(validation_result['errors'])
            for error in validation_result['errors']:
                self._provide_feedback(f"Error: {error}", level='error')
        
        # Check for timeout
        if current_step.start_time and (time.time() - current_step.start_time) > current_step.timeout:
            self._provide_feedback("Taking too long. Here's a hint:", level='warning')
            if current_step.hints:
                self._provide_feedback(current_step.hints[0], level='hint')
                self.metrics['hints_used'] += 1
        
        return {
            'status': 'in_progress',
            'current_step': current_step.to_dict(),
            'validation': validation_result,
            'feedback': validation_result.get('feedback', [])
        }
    
    def _validate_step(self, step: ProcedureStep) -> Dict[str, any]:
        """
        Validate if step requirements are satisfied.
        
        Args:
            step: ProcedureStep to validate
            
        Returns:
            Dict with validation results
        """
        result = {
            'completed': False,
            'errors': [],
            'warnings': [],
            'feedback': []
        }
        
        # Check preconditions
        preconditions_met = self._check_preconditions(step.preconditions)
        if not preconditions_met:
            result['errors'].append("Preconditions not satisfied")
            return result
        
        # Validate action based on type
        if step.action == 'grab':
            action_valid = self._validate_grab_action(step)
        elif step.action == 'place':
            action_valid = self._validate_place_action(step)
        elif step.action == 'press':
            action_valid = self._validate_press_action(step)
        elif step.action == 'look_at':
            action_valid = self._validate_look_at_action(step)
        else:
            action_valid = False
            result['warnings'].append(f"Unknown action type: {step.action}")
        
        # Check constraints
        constraints_satisfied = self._check_constraints(step.constraints)
        if not constraints_satisfied:
            result['errors'].append("Constraints violated")
        
        # Check postconditions for completion
        if action_valid and constraints_satisfied:
            postconditions_met = self._check_postconditions(step.postconditions)
            if postconditions_met:
                result['completed'] = True
                result['feedback'].append("Step completed successfully!")
        
        return result
    
    def _validate_grab_action(self, step: ProcedureStep) -> bool:
        """Validate grab action."""
        # Check if either hand is in grab gesture
        if self.left_hand and self.left_hand.gesture == HandGesture.GRAB:
            hand = self.left_hand
        elif self.right_hand and self.right_hand.gesture == HandGesture.GRAB:
            hand = self.right_hand
        else:
            return False
        
        # Check if hand is near target object
        if step.target_object:
            target_pos = self.world_state.get(f"{step.target_object}_position")
            if target_pos is not None:
                hand_pos = hand.joints[0]  # Palm position
                distance = np.linalg.norm(hand_pos - np.array(target_pos))
                return distance < 0.15  # 15cm threshold
        
        return True
    
    def _validate_place_action(self, step: ProcedureStep) -> bool:
        """Validate place action."""
        # Check if hand is in release gesture
        if self.left_hand and self.left_hand.gesture == HandGesture.RELEASE:
            hand = self.left_hand
        elif self.right_hand and self.right_hand.gesture == HandGesture.RELEASE:
            hand = self.right_hand
        else:
            return False
        
        # Check if hand is near target location
        if step.target_object:
            target_pos = self.world_state.get(f"{step.target_object}_position")
            if target_pos is not None:
                hand_pos = hand.joints[0]
                distance = np.linalg.norm(hand_pos - np.array(target_pos))
                return distance < 0.2  # 20cm threshold
        
        return True
    
    def _validate_press_action(self, step: ProcedureStep) -> bool:
        """Validate press action."""
        # Check if hand is in point or pinch gesture
        if self.left_hand and self.left_hand.gesture in [HandGesture.POINT, HandGesture.PINCH]:
            hand = self.left_hand
        elif self.right_hand and self.right_hand.gesture in [HandGesture.POINT, HandGesture.PINCH]:
            hand = self.right_hand
        else:
            return False
        
        # Check if fingertip is near target
        if step.target_object:
            target_pos = self.world_state.get(f"{step.target_object}_position")
            if target_pos is not None:
                fingertip_pos = hand.joints[8]  # Index fingertip
                distance = np.linalg.norm(fingertip_pos - np.array(target_pos))
                return distance < 0.05  # 5cm threshold
        
        return True
    
    def _validate_look_at_action(self, step: ProcedureStep) -> bool:
        """Validate look_at action."""
        if self.gaze is None:
            return False
        
        # Check if looking at target object
        if step.target_object:
            if self.gaze.fixation_object == step.target_object:
                return True
            
            # Alternative: check if gaze ray intersects target
            target_pos = self.world_state.get(f"{step.target_object}_position")
            if target_pos is not None and self.gaze.fixation_point is not None:
                distance = np.linalg.norm(self.gaze.fixation_point - np.array(target_pos))
                return distance < 0.3  # 30cm threshold
        
        return False
    
    def _check_preconditions(self, preconditions: List[str]) -> bool:
        """Check if preconditions are satisfied using Prolog."""
        if not preconditions:
            return True
        
        if self.prolog is None:
            # Fallback: simple string matching
            return all(self._check_fact(cond) for cond in preconditions)
        
        # Use Prolog for logical reasoning
        for condition in preconditions:
            try:
                query_result = list(self.prolog.query(condition))
                if not query_result:
                    return False
            except Exception as e:
                print(f"Prolog query error: {e}")
                return False
        
        return True
    
    def _check_postconditions(self, postconditions: List[str]) -> bool:
        """Check if postconditions are satisfied."""
        if not postconditions:
            return True
        
        if self.prolog is None:
            return all(self._check_fact(cond) for cond in postconditions)
        
        for condition in postconditions:
            try:
                query_result = list(self.prolog.query(condition))
                if not query_result:
                    return False
            except Exception as e:
                print(f"Prolog query error: {e}")
                return False
        
        return True
    
    def _check_constraints(self, constraints: List[str]) -> bool:
        """Check if constraints are satisfied."""
        if not constraints:
            return True
        
        # Simplified constraint checking
        for constraint in constraints:
            if not self._check_fact(constraint):
                return False
        
        return True
    
    def _check_fact(self, fact: str) -> bool:
        """Check if a fact is true in world state (simplified)."""
        # Parse fact string and check against world_state
        # Simplified implementation
        return fact in self.world_state and self.world_state[fact]
    
    def _complete_step(self, step: ProcedureStep) -> None:
        """Mark step as completed and update metrics."""
        step.status = StepStatus.COMPLETED
        step.end_time = time.time()
        
        if step.start_time:
            step_time = step.end_time - step.start_time
            self.metrics['step_times'].append(step_time)
        
        self._provide_feedback("✓ Step completed!", level='success')
    
    def _complete_procedure(self) -> None:
        """Mark procedure as completed and compute final metrics."""
        if self.current_procedure:
            self.current_procedure.end_time = time.time()
            
            if self.current_procedure.start_time:
                self.metrics['total_time'] = self.current_procedure.end_time - self.current_procedure.start_time
            
            completed_steps = sum(1 for step in self.current_procedure.steps if step.status == StepStatus.COMPLETED)
            self.metrics['completion_rate'] = completed_steps / len(self.current_procedure.steps)
            
            self._provide_feedback(f"🎉 Procedure completed!", level='success')
            self._provide_feedback(f"Total time: {self.metrics['total_time']:.1f}s", level='info')
            self._provide_feedback(f"Errors: {self.metrics['errors']}", level='info')
    
    def _provide_feedback(self, message: str, level: str = 'info') -> None:
        """Provide feedback to user through callbacks."""
        feedback = {
            'message': message,
            'level': level,
            'timestamp': time.time()
        }
        
        for callback in self.feedback_callbacks:
            callback(feedback)
    
    def _initialize_prolog_kb(self) -> None:
        """Initialize Prolog knowledge base with common rules."""
        if self.prolog is None:
            return
        
        # Define common predicates
        rules = [
            # Spatial relationships
            "near(X, Y) :- distance(X, Y, D), D < 0.5.",
            "far(X, Y) :- distance(X, Y, D), D > 2.0.",
            "touching(X, Y) :- distance(X, Y, D), D < 0.05.",
            
            # Hand states
            "holding(Hand, Object) :- hand_gesture(Hand, grab), near(Hand, Object).",
            "pointing_at(Hand, Object) :- hand_gesture(Hand, point), near(Hand, Object).",
            
            # Gaze
            "looking_at(Object) :- gaze_target(Object).",
            "attending_to(Object) :- looking_at(Object), gaze_duration(Object, D), D > 1.0.",
            
            # Procedure logic
            "can_grab(Object) :- not(holding(_, Object)), reachable(Object).",
            "can_place(Object, Location) :- holding(_, Object), reachable(Location).",
        ]
        
        for rule in rules:
            try:
                self.prolog.assertz(rule)
            except Exception as e:
                print(f"Error adding Prolog rule: {e}")
    
    def add_feedback_callback(self, callback: Callable) -> None:
        """Add callback for receiving feedback."""
        self.feedback_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, any]:
        """Get performance metrics."""
        return self.metrics.copy()
    
    def reset(self) -> None:
        """Reset training assistant state."""
        self.current_procedure = None
        self.left_hand = None
        self.right_hand = None
        self.gaze = None
        self.world_state.clear()
        self.metrics = {
            'total_time': 0.0,
            'step_times': [],
            'errors': 0,
            'hints_used': 0,
            'completion_rate': 0.0
        }


def create_training_assistant(config: dict) -> TrainingAssistant:
    """
    Factory function to create Training Assistant.
    
    Args:
        config: Configuration dict with keys:
            - device: str
            - use_prolog: bool
            
    Returns:
        Configured TrainingAssistant
    """
    return TrainingAssistant(
        device=config.get('device', 'cuda'),
        use_prolog=config.get('use_prolog', True)
    )


# Example procedure definition
def create_example_procedure() -> TrainingProcedure:
    """Create an example training procedure."""
    return TrainingProcedure(
        id="assembly_001",
        name="Basic Assembly Task",
        description="Learn to assemble a simple device",
        steps=[
            ProcedureStep(
                id="step_1",
                description="Pick up the base component",
                action="grab",
                target_object="base_component",
                preconditions=["reachable(base_component)"],
                postconditions=["holding(hand, base_component)"],
                constraints=["not(holding(hand, _))"],
                timeout=30.0,
                hints=["Look at the blue base component and grab it with your hand"]
            ),
            ProcedureStep(
                id="step_2",
                description="Place the base on the work surface",
                action="place",
                target_object="work_surface",
                preconditions=["holding(hand, base_component)"],
                postconditions=["on(base_component, work_surface)"],
                constraints=[],
                timeout=30.0,
                hints=["Move your hand to the work surface and release"]
            ),
            ProcedureStep(
                id="step_3",
                description="Pick up the top component",
                action="grab",
                target_object="top_component",
                preconditions=["on(base_component, work_surface)", "reachable(top_component)"],
                postconditions=["holding(hand, top_component)"],
                constraints=[],
                timeout=30.0,
                hints=["Now grab the red top component"]
            ),
            ProcedureStep(
                id="step_4",
                description="Attach the top to the base",
                action="place",
                target_object="base_component",
                preconditions=["holding(hand, top_component)", "on(base_component, work_surface)"],
                postconditions=["on(top_component, base_component)", "assembled(device)"],
                constraints=[],
                timeout=45.0,
                hints=["Carefully place the top component on the base"]
            )
        ]
    )
